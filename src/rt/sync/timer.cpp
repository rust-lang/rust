#include "../globals.h"
#include "timer.h"

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

uint64_t ns_per_s = 1000000000LL;

timer::timer() {
#if __WIN32__
    _ticks_per_s = 0LL;
    // FIXME: assert this works or have a workaround.
    QueryPerformanceFrequency((LARGE_INTEGER *)&_ticks_per_s);
    if (_ticks_per_s == 0LL) {
      _ticks_per_s = 1LL;
    }
#endif
    reset_us(0);
}

void
timer::reset_us(uint64_t timeout_us) {
    _start_us = time_us();
    _timeout_us = timeout_us;
}

uint64_t
timer::elapsed_us() {
    return time_us() - _start_us;
}

double
timer::elapsed_ms() {
    return (double) elapsed_us() / 1000.0;
}

int64_t
timer::remaining_us() {
    return _timeout_us - elapsed_us();
}

bool
timer::has_timed_out() {
    return remaining_us() <= 0;
}

uint64_t
timer::time_ns() {
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
    return ((ticks * ns_per_s) / _ticks_per_s);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * ns_per_s + ts.tv_nsec);
#endif
}

uint64_t
timer::time_us() {
    return time_ns() / 1000;
}

timer::~timer() {
}
