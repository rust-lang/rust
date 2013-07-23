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


use std::libc::{c_char, c_int};
use std::local_data;
use std::str;

pub mod rustrt {
    use std::libc::{c_char, c_int};

    extern {
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
    do line.as_c_str |buf| {
        rustrt::linenoiseHistoryAdd(buf) == 1 as c_int
    }
}

/// Set the maximum amount of lines stored
pub unsafe fn set_history_max_len(len: int) -> bool {
    rustrt::linenoiseHistorySetMaxLen(len as c_int) == 1 as c_int
}

/// Save line history to a file
pub unsafe fn save_history(file: &str) -> bool {
    do file.as_c_str |buf| {
        rustrt::linenoiseHistorySave(buf) == 1 as c_int
    }
}

/// Load line history from a file
pub unsafe fn load_history(file: &str) -> bool {
    do file.as_c_str |buf| {
        rustrt::linenoiseHistoryLoad(buf) == 1 as c_int
    }
}

/// Print out a prompt and then wait for input and return it
pub unsafe fn read(prompt: &str) -> Option<~str> {
    do prompt.as_c_str |buf| {
        let line = rustrt::linenoise(buf);

        if line.is_null() { None }
        else { Some(str::raw::from_c_str(line)) }
    }
}

pub type CompletionCb = @fn(~str, @fn(~str));

static complete_key: local_data::Key<@CompletionCb> = &local_data::Key;

/// Bind to the main completion callback
pub unsafe fn complete(cb: CompletionCb) {
    local_data::set(complete_key, @cb);

    extern fn callback(line: *c_char, completions: *()) {
        do local_data::get(complete_key) |cb| {
            let cb = **cb.unwrap();

            unsafe {
                do cb(str::raw::from_c_str(line)) |suggestion| {
                    do suggestion.as_c_str |buf| {
                        rustrt::linenoiseAddCompletion(completions, buf);
                    }
                }
}
        }
    }

    rustrt::linenoiseSetCompletionCallback(callback);
}
