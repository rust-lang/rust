#include "../globals.h"
#include "sync.h"

void sync::yield() {
#ifdef __APPLE__
    pthread_yield_np();
#elif __WIN32__
    Sleep(1);
#else
    pthread_yield();
#endif
}

rust_thread::rust_thread() : _is_running(false) {
    // Nop.
}

#if defined(__WIN32__)
static DWORD WINAPI
#elif defined(__GNUC__)
static void *
#else
#error "Platform not supported"
#endif
rust_thread_start(void *ptr) {
    rust_thread *thread = (rust_thread *) ptr;
    thread->run();
    thread->thread = 0;
    return 0;
}

void
rust_thread::start() {
#if defined(__WIN32__)
   thread = CreateThread(NULL, 0, rust_thread_start, this, 0, NULL);
#else
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setstacksize(&attr, 1024 * 1024);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   pthread_create(&thread, &attr, rust_thread_start, (void *) this);
#endif
   _is_running = true;
}

void
rust_thread::join() {
#if defined(__WIN32__)
   WaitForSingleObject(thread, INFINITE);
#else
   pthread_join(thread, NULL);
#endif
   thread = 0;
   _is_running = false;
}

bool
rust_thread::is_running() {
    return _is_running;
}
