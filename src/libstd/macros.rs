// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];
#[doc(hidden)];

macro_rules! rterrln (
    ($($arg:tt)*) => ( {
        ::rt::util::dumb_println(format!($($arg)*));
    } )
)

// Some basic logging. Enabled by passing `--cfg rtdebug` to the libstd build.
macro_rules! rtdebug (
    ($($arg:tt)*) => ( {
        if cfg!(rtdebug) {
            rterrln!($($arg)*)
        }
    })
)

macro_rules! rtassert (
    ( $arg:expr ) => ( {
        if ::rt::util::ENFORCE_SANITY {
            if !$arg {
                rtabort!("assertion failed: {}", stringify!($arg));
            }
        }
    } )
)


macro_rules! rtabort(
    ($($msg:tt)*) => ( {
        ::rt::util::abort(format!($($msg)*));
    } )
)

macro_rules! assert_once_ever(
    ($($msg:tt)+) => ( {
        // FIXME(#8472) extra function should not be needed to hide unsafe
        fn assert_once_ever() {
            unsafe {
                static mut already_happened: int = 0;
                // Double-check lock to avoid a swap in the common case.
                if already_happened != 0 ||
                    ::unstable::intrinsics::atomic_xchg_relaxed(&mut already_happened, 1) != 0 {
                        fail2!("assert_once_ever happened twice: {}",
                               format!($($msg)+));
                }
            }
        }
        assert_once_ever();
    } )
)

#[cfg(test)]
mod tests {
    #[test]
    fn test_assert_once_ever_ok() {
        assert_once_ever!("help i'm stuck in an");
        assert_once_ever!("assertion error message");
    }

    #[test] #[ignore(cfg(windows))] #[should_fail]
    fn test_assert_once_ever_fail() {
        use task;

        fn f() { assert_once_ever!("if you're seeing this... good!") }

        // linked & watched, naturally
        task::spawn(f);
        task::spawn(f);
    }
}
