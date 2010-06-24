/*
 * This works well as long as the number of contending threads
 * is less than the number of processors. This is because of
 * the fair locking scheme. If the thread that is next in line
 * for acquiring the lock is not currently running, no other
 * thread can acquire the lock. This is terrible for performance,
 * and it seems that all fair locking schemes suffer from this
 * behavior.
 */

// #define TRACE

fair_ticket_lock::fair_ticket_lock() {
    next_ticket = now_serving = 0;
}

fair_ticket_lock::~fair_ticket_lock() {

}

void fair_ticket_lock::lock() {
    unsigned ticket = __sync_fetch_and_add(&next_ticket, 1);
    while (now_serving != ticket) {
        pause();
    }
#ifdef TRACE
    printf("locked   nextTicket: %d nowServing: %d",
            next_ticket, now_serving);
#endif
}

void fair_ticket_lock::unlock() {
    now_serving++;
#ifdef TRACE
    printf("unlocked nextTicket: %d nowServing: %d",
            next_ticket, now_serving);
#endif
}

void fair_ticket_lock::pause() {
    asm volatile("pause\n" : : : "memory");
}

