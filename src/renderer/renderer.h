#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "../common.h"

#define NUM_ROW_THREADS 8
#define NUM_COL_THREADS 8

class Renderer : public CudaManaged{
private:
	GLubyte* _framebuffer;
	uint64_t _framebuffer_size;
	uint32_t _framebuffer_id;
	uint32_t _texture_id;

	void framebuffer_setup();

public:
	static void init();
	
	Renderer();

	void update_frame();
	__device__ void update_pixel(uint32_t row, uint32_t col);
};