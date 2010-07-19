#include "../globals.h"
#include "spin_lock.h"

/*
 * Your average spin lock.
 */

// #define TRACE

spin_lock::spin_lock() {
    unlock();
}

spin_lock::~spin_lock() {
}

static inline unsigned xchg32(void *ptr, unsigned x) {
    __asm__ __volatile__("xchgl %0,%1"
                :"=r" ((unsigned) x)
                :"m" (*(volatile unsigned *)ptr), "0" (x)
                :"memory");
    return x;
}

void spin_lock::lock() {
    while (true) {
        if (!xchg32(&ticket, 1)) {
            return;
        }
        while (ticket) {
            pause();
        }
    }
#ifdef TRACE
    printf("  lock: %d", ticket);
#endif
}

void spin_lock::unlock() {
    ticket = 0;
#ifdef TRACE
    printf("unlock:");
#endif
}

void spin_lock::pause() {
    asm volatile("pause\n" : : : "memory");
}
