// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_LOG_H
#define RUST_LOG_H

#include "rust_globals.h"

const uint32_t log_err = 1;
const uint32_t log_warn = 2;
const uint32_t log_info = 3;
const uint32_t log_debug = 4;

void update_log_settings(void* crate_map, char* settings);

extern uint32_t log_rt_mem;
extern uint32_t log_rt_box;
extern uint32_t log_rt_comm;
extern uint32_t log_rt_task;
extern uint32_t log_rt_dom;
extern uint32_t log_rt_trace;
extern uint32_t log_rt_cache;
extern uint32_t log_rt_upcall;
extern uint32_t log_rt_timer;
extern uint32_t log_rt_gc;
extern uint32_t log_rt_stdlib;
extern uint32_t log_rt_kern;
extern uint32_t log_rt_backtrace;
extern uint32_t log_rt_callback;

#endif /* RUST_LOG_H */
