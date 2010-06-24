
#include "rust_internal.h"

// The mechanism in this file is very crude; every domain (thread) spawns its
// own secondary timer thread, and that timer thread *never idles*. It
// sleep-loops interrupting the domain.
//
// This will need replacement, particularly in order to achieve an actual
// state of idling when we're waiting on the outside world.  Though that might
// be as simple as making a secondary waitable start/stop-timer signalling
// system between the domain and its timer thread. We'll see.
//
// On the other hand, we don't presently have the ability to idle domains *at
// all*, and without the timer thread we're unable to otherwise preempt rust
// tasks. So ... one step at a time.
//
// The implementation here is "lockless" in the sense that it only involves
// one-directional signaling of one-shot events, so the event initiator just
// writes a nonzero word to a prederermined location and waits for the
// receiver to see it show up in their memory.

#if defined(__WIN32__)
static DWORD WINAPI
win32_timer_loop(void *ptr)
{
    // We were handed the rust_timer that owns us.
    rust_timer *timer = (rust_timer *)ptr;
    rust_dom &dom = timer->dom;
    dom.log(LOG_TIMER, "in timer 0x%" PRIxPTR, (uintptr_t)timer);
    while (!timer->exit_flag) {
        Sleep(TIME_SLICE_IN_MS);
        dom.log(LOG_TIMER,
                "timer 0x%" PRIxPTR
                " interrupting domain 0x%" PRIxPTR,
                (uintptr_t)timer,
                (uintptr_t)&dom);
        dom.interrupt_flag = 1;
    }
    ExitThread(0);
    return 0;
}

#elif defined(__GNUC__)
static void *
pthread_timer_loop(void *ptr)
{
    // We were handed the rust_timer that owns us.
    rust_timer *timer = (rust_timer *)ptr;
    rust_dom &dom(timer->dom);
    while (!timer->exit_flag) {
        usleep(TIME_SLICE_IN_MS * 1000);
        dom.interrupt_flag = 1;
    }
    pthread_exit(NULL);
    return 0;

}
#else
#error "Platform not supported"
#endif


rust_timer::rust_timer(rust_dom &dom) : dom(dom), exit_flag(0)
{
    dom.log(rust_log::TIMER, "creating timer for domain 0x%" PRIxPTR, &dom);
#if defined(__WIN32__)
    thread = CreateThread(NULL, 0, win32_timer_loop, this, 0, NULL);
    dom.win32_require("CreateThread", thread != NULL);
#else
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&thread, &attr, pthread_timer_loop, (void *)this);
#endif
}

rust_timer::~rust_timer()
{
    exit_flag = 1;
#if defined(__WIN32__)
    dom.win32_require("WaitForSingleObject",
                      WaitForSingleObject(thread, INFINITE)
                      == WAIT_OBJECT_0);
#else
    pthread_join(thread, NULL);
#endif
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
