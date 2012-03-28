#include "rust_globals.h"
#include "rust_thread.h"

const size_t default_stack_sz = 1024*1024;

rust_thread::rust_thread() : thread(0), stack_sz(default_stack_sz) {
}

rust_thread::rust_thread(size_t stack_sz)
  : thread(0), stack_sz(stack_sz) {
}

rust_thread::~rust_thread() {
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
    return 0;
}

void
rust_thread::start() {
#if defined(__WIN32__)
   thread = CreateThread(NULL, stack_sz, rust_thread_start, this, 0, NULL);
#else
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setstacksize(&attr, stack_sz);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   pthread_create(&thread, &attr, rust_thread_start, (void *) this);
#endif
}

void
rust_thread::join() {
#if defined(__WIN32__)
   if (thread)
     WaitForSingleObject(thread, INFINITE);
#else
   if (thread)
     pthread_join(thread, NULL);
#endif
   thread = 0;
}

void
rust_thread::detach() {
#if !defined(__WIN32__)
    // Don't leak pthread resources.
    // http://crosstantine.blogspot.com/2010/01/pthreadcreate-memory-leak.html
    pthread_detach(thread);
#endif
}
