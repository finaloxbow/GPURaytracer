#include "window.h"

void Window::processInput()
{
	if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(_window, true);
	}
}

Window::Window(uint32_t width, uint32_t height, const char* title)
{
	_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	glfwMakeContextCurrent(_window);
	glfwSwapInterval(0);
}

Window::~Window()
{
	glfwDestroyWindow(_window);
}

void Window::init()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}

void Window::cleanup()
{
	glfwTerminate();
}

bool Window::is_alive()
{
    return !glfwWindowShouldClose(_window);
}

void Window::update_frame()
{
	processInput();
	glfwSwapBuffers(_window);
	glfwPollEvents();
}
