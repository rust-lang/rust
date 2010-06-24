#ifndef UNFAIR_TICKET_LOCK_H
#define UNFAIR_TICKET_LOCK_H

class spin_lock {
    unsigned ticket;
    void pause();
public:
    spin_lock();
    virtual ~spin_lock();
    void lock();
    void unlock();
};

#endif /* UNFAIR_TICKET_LOCK_H */
