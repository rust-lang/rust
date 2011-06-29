#include "../globals.h"
#include "timer.h"

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

timer::timer() {
#if __WIN32__
    uint64_t ticks_per_second;
    QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_second);
    _ticks_per_ns = ticks_per_second / 1000;
#endif
    reset(0);
}

void
timer::reset(uint64_t timeout) {
    _start = get_time();
    _timeout = timeout;
}

uint64_t
timer::get_elapsed_time() {
    return get_time() - _start;
}

double
timer::get_elapsed_time_in_ms() {
    return (double) get_elapsed_time() / 1000.0;
}

int64_t
timer::get_timeout() {
    return _timeout - get_elapsed_time();
}

bool
timer::has_timed_out() {
    return get_timeout() <= 0;
}

uint64_t
timer::nano_time() {
#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    return time_nano;
#elif __WIN32__
    uint64_t ticks;
    QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
    return ticks / _ticks_per_ns;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000000LL + ts.tv_nsec);
#endif
}

uint64_t
timer::get_time() {
    return nano_time() / 1000;
}

timer::~timer() {
    // Nop.
}
