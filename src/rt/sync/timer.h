/*
 *  Utility class to measure time in a platform independent way.
 */

#ifndef TIMER_H
#define TIMER_H

class timer {
private:
    uint64_t _start;
    uint64_t _timeout;
    uint64_t get_time();
#if __WIN32__
    uint64_t _ticks_per_us;
#endif
public:
    timer();
    void reset(uint64_t timeout);
    uint64_t get_elapsed_time();
    int64_t get_timeout();
    bool has_timed_out();
    virtual ~timer();
};

#endif /* TIMER_H */
