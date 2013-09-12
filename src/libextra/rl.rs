// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::c_str::ToCStr;
use std::libc::{c_char, c_int};
use std::{local_data, str, rt};
use std::unstable::finally::Finally;

#[cfg(stage0)]
pub mod rustrt {
    use std::libc::{c_char, c_int};

    extern {
        fn linenoise(prompt: *c_char) -> *c_char;
        fn linenoiseHistoryAdd(line: *c_char) -> c_int;
        fn linenoiseHistorySetMaxLen(len: c_int) -> c_int;
        fn linenoiseHistorySave(file: *c_char) -> c_int;
        fn linenoiseHistoryLoad(file: *c_char) -> c_int;
        fn linenoiseSetCompletionCallback(callback: *u8);
        fn linenoiseAddCompletion(completions: *(), line: *c_char);

        fn rust_take_linenoise_lock();
        fn rust_drop_linenoise_lock();
    }
}

#[cfg(not(stage0))]
pub mod rustrt {
    use std::libc::{c_char, c_int};

    externfn!(fn linenoise(prompt: *c_char) -> *c_char)
    externfn!(fn linenoiseHistoryAdd(line: *c_char) -> c_int)
    externfn!(fn linenoiseHistorySetMaxLen(len: c_int) -> c_int)
    externfn!(fn linenoiseHistorySave(file: *c_char) -> c_int)
    externfn!(fn linenoiseHistoryLoad(file: *c_char) -> c_int)
    externfn!(fn linenoiseSetCompletionCallback(callback: extern "C" fn(*i8, *())))
    externfn!(fn linenoiseAddCompletion(completions: *(), line: *c_char))

    externfn!(fn rust_take_linenoise_lock())
    externfn!(fn rust_drop_linenoise_lock())
}

macro_rules! locked {
    ($expr:expr) => {
        unsafe {
            // FIXME #9105: can't use a static mutex in pure Rust yet.
            rustrt::rust_take_linenoise_lock();
            let x = $expr;
            rustrt::rust_drop_linenoise_lock();
            x
        }
    }
}

/// Add a line to history
pub fn add_history(line: &str) -> bool {
    do line.with_c_str |buf| {
        (locked!(rustrt::linenoiseHistoryAdd(buf))) == 1 as c_int
    }
}

/// Set the maximum amount of lines stored
pub fn set_history_max_len(len: int) -> bool {
    (locked!(rustrt::linenoiseHistorySetMaxLen(len as c_int))) == 1 as c_int
}

/// Save line history to a file
pub fn save_history(file: &str) -> bool {
    do file.with_c_str |buf| {
        // 0 on success, -1 on failure
        (locked!(rustrt::linenoiseHistorySave(buf))) == 0 as c_int
    }
}

/// Load line history from a file
pub fn load_history(file: &str) -> bool {
    do file.with_c_str |buf| {
        // 0 on success, -1 on failure
        (locked!(rustrt::linenoiseHistoryLoad(buf))) == 0 as c_int
    }
}

/// Print out a prompt and then wait for input and return it
pub fn read(prompt: &str) -> Option<~str> {
    do prompt.with_c_str |buf| {
        let line = locked!(rustrt::linenoise(buf));

        if line.is_null() { None }
        else {
            unsafe {
                do (|| {
                    Some(str::raw::from_c_str(line))
                }).finally {
                    // linenoise's return value is from strdup, so we
                    // better not leak it.
                    rt::global_heap::exchange_free(line);
                }
            }
        }
    }
}

pub type CompletionCb = @fn(~str, @fn(~str));

static complete_key: local_data::Key<CompletionCb> = &local_data::Key;

/// Bind to the main completion callback in the current task.
///
/// The completion callback should not call any `extra::rl` functions
/// other than the closure that it receives as its second
/// argument. Calling such a function will deadlock on the mutex used
/// to ensure that the calls are thread-safe.
pub fn complete(cb: CompletionCb) {
    local_data::set(complete_key, cb);

    extern fn callback(c_line: *c_char, completions: *()) {
        do local_data::get(complete_key) |opt_cb| {
            // only fetch completions if a completion handler has been
            // registered in the current task.
            match opt_cb {
                None => {},
                Some(cb) => {
                    let line = unsafe { str::raw::from_c_str(c_line) };
                    do (*cb)(line) |suggestion| {
                        do suggestion.with_c_str |buf| {
                            // This isn't locked, because `callback` gets
                            // called inside `rustrt::linenoise`, which
                            // *is* already inside the mutex, so
                            // re-locking would be a deadlock.
                            unsafe {
                                rustrt::linenoiseAddCompletion(completions, buf);
                            }
                        }
                    }
                }
            }
        }
    }

    locked!(rustrt::linenoiseSetCompletionCallback(callback));
}
