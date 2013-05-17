// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME #3921. This is unsafe because linenoise uses global mutable
// state without mutexes.

use core::prelude::*;

use core::libc::{c_char, c_int};

pub mod rustrt {
    use core::libc::{c_char, c_int};

    pub extern {
        pub unsafe fn linenoise(prompt: *c_char) -> *c_char;
        pub unsafe fn linenoiseHistoryAdd(line: *c_char) -> c_int;
        pub unsafe fn linenoiseHistorySetMaxLen(len: c_int) -> c_int;
        pub unsafe fn linenoiseHistorySave(file: *c_char) -> c_int;
        pub unsafe fn linenoiseHistoryLoad(file: *c_char) -> c_int;
        pub unsafe fn linenoiseSetCompletionCallback(callback: *u8);
        pub unsafe fn linenoiseAddCompletion(completions: *(), line: *c_char);
    }
}

/// Add a line to history
pub unsafe fn add_history(line: &str) -> bool {
    do str::as_c_str(line) |buf| {
        rustrt::linenoiseHistoryAdd(buf) == 1 as c_int
    }
}

/// Set the maximum amount of lines stored
pub unsafe fn set_history_max_len(len: int) -> bool {
    rustrt::linenoiseHistorySetMaxLen(len as c_int) == 1 as c_int
}

/// Save line history to a file
pub unsafe fn save_history(file: &str) -> bool {
    do str::as_c_str(file) |buf| {
        rustrt::linenoiseHistorySave(buf) == 1 as c_int
    }
}

/// Load line history from a file
pub unsafe fn load_history(file: &str) -> bool {
    do str::as_c_str(file) |buf| {
        rustrt::linenoiseHistoryLoad(buf) == 1 as c_int
    }
}

/// Print out a prompt and then wait for input and return it
pub unsafe fn read(prompt: &str) -> Option<~str> {
    do str::as_c_str(prompt) |buf| {
        let line = rustrt::linenoise(buf);

        if line.is_null() { None }
        else { Some(str::raw::from_c_str(line)) }
    }
}

pub type CompletionCb<'self> = @fn(~str, &'self fn(~str));

fn complete_key(_v: @CompletionCb) {}

/// Bind to the main completion callback
pub unsafe fn complete(cb: CompletionCb) {
    local_data::local_data_set(complete_key, @(cb));

    extern fn callback(line: *c_char, completions: *()) {
        unsafe {
            let cb = *local_data::local_data_get(complete_key)
                .get();

            do cb(str::raw::from_c_str(line)) |suggestion| {
                do str::as_c_str(suggestion) |buf| {
                    rustrt::linenoiseAddCompletion(completions, buf);
                }
            }
        }
    }

    rustrt::linenoiseSetCompletionCallback(callback);
}
