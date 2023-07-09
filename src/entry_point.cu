#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "window/window.h"
#include "renderer/renderer.h"

int main() {
	
	Window::init();
	Window* window = new Window(WINDOW_WIDTH, WINDOW_HEIGHT, "LearnCUDA");

	Renderer::init();
	Renderer* renderer = new Renderer();

	while (window->is_alive()) {
		renderer->update_frame();
		window->update_frame();
	}

	Window::cleanup();

	return 0;
}