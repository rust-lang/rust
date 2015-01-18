// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {
    E0001: r##"
    This error suggests that the expression arm corresponding to the noted pattern
    will never be reached as for all possible values of the expression being matched,
    one of the preceeding patterns will match.

    This means that perhaps some of the preceeding patterns are too general, this
    one is too specific or the ordering is incorrect.
"##,

    E0003: r##"
    Not-a-Number (NaN) values can not be compared for equality and hence can never match
    the input to a match expression. To match against NaN values, you should instead use
    the `is_nan` method in a guard, as in: x if x.is_nan() => ...
"##,

    E0004: r##"
    This error indicates that the compiler can not guarantee a matching pattern for one
    or more possible inputs to a match expression. Guaranteed matches are required in order
    to assign values to match expressions, or alternatively, determine the flow of execution.

    If you encounter this error you must alter your patterns so that every possible value of
    the input type is matched. For types with a small number of variants (like enums) you
    should probably cover all cases explicitly. Alternatively, the underscore `_` wildcard
    pattern can be added after all other patterns to match "anything else".
"##,

    // FIXME: Remove duplication here?
    E0005: r##"
    Patterns used to bind names must be irrefutable, that is, they must guarantee that a
    name will be extracted in all cases. If you encounter this error you probably need
    to use a `match` or `if let` to deal with the possibility of failure.
"##,

    E0006: r##"
    Patterns used to bind names must be irrefutable, that is, they must guarantee that a
    name will be extracted in all cases. If you encounter this error you probably need
    to use a `match` or `if let` to deal with the possibility of failure.
"##
}

register_diagnostics! {
    E0002,
    E0007,
    E0008,
    E0009,
    E0010,
    E0011,
    E0012,
    E0014,
    E0015,
    E0016,
    E0017,
    E0018,
    E0019,
    E0020,
    E0022,
    E0109,
    E0110,
    E0133,
    E0134,
    E0135,
    E0136,
    E0137,
    E0138,
    E0139,
    E0152,
    E0158,
    E0161,
    E0162,
    E0165,
    E0170
}

__build_diagnostic_array! { DIAGNOSTICS }

