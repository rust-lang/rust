#ifndef SPIN_LOCK_H
#define SPIN_LOCK_H

class spin_lock {
    unsigned ticket;
    void pause();
public:
    spin_lock();
    virtual ~spin_lock();
    void lock();
    void unlock();
};

#endif /* SPIN_LOCK_H */
