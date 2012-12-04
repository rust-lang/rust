// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 *  Utility class to measure time in a platform independent way.
 */

#ifndef TIMER_H
#define TIMER_H

class timer {
private:
    uint64_t _start_us;
    uint64_t _timeout_us;
    uint64_t time_us();
#if __WIN32__
    uint64_t _ticks_per_s;
#endif
public:
    timer();
    void reset_us(uint64_t timeout);
    uint64_t elapsed_us();
    double elapsed_ms();
    int64_t remaining_us();
    bool has_timed_out();
    uint64_t time_ns();
    virtual ~timer();
};

#endif /* TIMER_H */
