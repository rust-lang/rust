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
    uint64_t _ticks_per_ns;
#endif
public:
    timer();
    void reset(uint64_t timeout);
    uint64_t get_elapsed_time();
    double get_elapsed_time_in_ms();
    int64_t get_timeout();
    bool has_timed_out();
    uint64_t nano_time();
    virtual ~timer();
};

#endif /* TIMER_H */
