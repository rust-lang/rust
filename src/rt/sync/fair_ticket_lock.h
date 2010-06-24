#ifndef FAIR_TICKET_LOCK_H
#define FAIR_TICKET_LOCK_H

class fair_ticket_lock {
    unsigned next_ticket;
    unsigned now_serving;
    void pause();
public:
    fair_ticket_lock();
    virtual ~fair_ticket_lock();
    void lock();
    void unlock();
};

#endif /* FAIR_TICKET_LOCK_H */
