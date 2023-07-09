#include "renderer.h"
#include "../common.h"

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>

__global__ void calc_pixel_kernel(Renderer* renderer) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	renderer->update_pixel(row, col);
}

void Renderer::framebuffer_setup()
{
	//data alignment stuff
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
	glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

	//texture creation
	glGenTextures(1, &_texture_id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _texture_id);

	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGBA,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		0,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		_framebuffer
	);

	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//generates texture mipmap
	glGenerateMipmap(GL_TEXTURE_2D);

	//creates framebuffer and attaches previous texture to it
	glGenFramebuffers(1, &_framebuffer_id);
	glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer_id);
	glFramebufferTexture(
		GL_FRAMEBUFFER,
		GL_COLOR_ATTACHMENT0,
		_texture_id,
		0
	);
}

void Renderer::init()
{
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw new std::runtime_error("opengl init failed");
	}
}

Renderer::Renderer()
{
	_framebuffer_size = WINDOW_WIDTH * WINDOW_HEIGHT * NUM_CHANNELS;
	checkCudaErrors(cudaMallocManaged((void**)&_framebuffer, _framebuffer_size));
	_texture_id = 0;
	_framebuffer_id = 0;

	framebuffer_setup();
}

void Renderer::update_frame()
{
	dim3 blocks((WINDOW_HEIGHT / NUM_ROW_THREADS) + 1, (WINDOW_WIDTH / NUM_COL_THREADS) + 1);
	dim3 threads(NUM_ROW_THREADS, NUM_COL_THREADS);
	calc_pixel_kernel<<<blocks, threads>>>(this);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0,
		0,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		_framebuffer
	);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, _framebuffer_id);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glBlitFramebuffer(
		0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
		0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

__device__ void Renderer::update_pixel(uint32_t row, uint32_t col)
{
	if (row >= WINDOW_HEIGHT || col >= WINDOW_WIDTH) return;

	float red = float(row) / WINDOW_HEIGHT;
	float green = float(col) / WINDOW_WIDTH;
	float blue = 0.2f;
	float alpha = 0.99f;

	_framebuffer[row * WINDOW_WIDTH * NUM_CHANNELS + col * NUM_CHANNELS + 0] = static_cast<int>(256 * red);
	_framebuffer[row * WINDOW_WIDTH * NUM_CHANNELS + col * NUM_CHANNELS + 1] = static_cast<int>(256 * green);
	_framebuffer[row * WINDOW_WIDTH * NUM_CHANNELS + col * NUM_CHANNELS + 2] = static_cast<int>(256 * blue);
	_framebuffer[row * WINDOW_WIDTH * NUM_CHANNELS + col * NUM_CHANNELS + 3] = static_cast<int>(256 * alpha);
}
