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

macro_rules! rterrln (
    ($( $arg:expr),+) => ( {
        ::rt::util::dumb_println(fmt!( $($arg),+ ));
    } )
)

// Some basic logging
macro_rules! rtdebug_ (
    ($( $arg:expr),+) => ( {
        rterrln!( $($arg),+ )
    } )
)

// An alternate version with no output, for turning off logging. An
// earlier attempt that did not call the fmt! macro was insufficient,
// as a case of the "let bind each variable" approach eventually
// failed without an error message describing the invocation site.
macro_rules! rtdebug (
    ($( $arg:expr),+) => ( {
        let _x = fmt!( $($arg),+ );
    })
)

macro_rules! rtassert (
    ( $arg:expr ) => ( {
        if !$arg {
            rtabort!("assertion failed: %s", stringify!($arg));
        }
    } )
)


macro_rules! rtabort(
    ($( $msg:expr),+) => ( {
        ::rt::util::abort(fmt!($($msg),+));
    } )
)

