#include "rust_internal.h"
#include "valgrind.h"

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
#elif defined(__GNUC__)
static void *
#else
#error "Platform not supported"
#endif
timer_loop(void *ptr) {
    // We were handed the rust_timer that owns us.
    rust_timer *timer = (rust_timer *)ptr;
    rust_scheduler *sched = timer->sched;
    DLOG(sched, timer, "in timer 0x%" PRIxPTR, (uintptr_t)timer);
    size_t ms = TIME_SLICE_IN_MS;

    while (!timer->exit_flag) {
#if defined(__WIN32__)
        Sleep(ms);
#else
        usleep(ms * 1000);
#endif
        DLOG(sched, timer, "timer 0x%" PRIxPTR
        " interrupting schedain 0x%" PRIxPTR, (uintptr_t) timer,
                 (uintptr_t) sched);
        sched->interrupt_flag = 1;
    }
#if defined(__WIN32__)
    ExitThread(0);
#else
    pthread_exit(NULL);
#endif
    return 0;
}

rust_timer::rust_timer(rust_scheduler *sched) :
    sched(sched), exit_flag(0) {
    DLOG(sched, timer, "creating timer for domain 0x%" PRIxPTR, sched);
#if defined(__WIN32__)
    thread = CreateThread(NULL, 0, timer_loop, this, 0, NULL);
    sched->kernel->win32_require("CreateThread", thread != NULL);
    if (RUNNING_ON_VALGRIND)
        Sleep(10);
#else
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&thread, &attr, timer_loop, (void *)this);
#endif
}

rust_timer::~rust_timer() {
    exit_flag = 1;
#if defined(__WIN32__)
    sched->kernel->win32_require("WaitForSingleObject",
                               WaitForSingleObject(thread, INFINITE) == 
                               WAIT_OBJECT_0);
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
