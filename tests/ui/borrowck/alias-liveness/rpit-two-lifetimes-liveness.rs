//@ check-fail
//@ known-bug: unknown
//@ edition: 2024

// Both of these cases *should* compile, because neither `'a` nor `'b` outlive
// each other (and nothing outlives both of them), so neither lifetime should be
// live.

#![forbid(unsafe_code)]

use std::fmt::Debug;

struct Captured<'a> {
    _value: &'a u8,
}

fn no_common_outliving_arg<'a, 'b, T>(_: T) -> impl Debug + 'a + 'b {
    ()
}

fn bounds_are_not_contents<'a, 'b>(_: &'a u8, _: &'b u8) -> impl Debug + 'a + 'b {
    ()
}

fn main() {
    let escaped;
    {
        let short = 0;
        escaped = no_common_outliving_arg(Captured { _value: &short });
    }
    println!("escaped = {escaped:?}");

    let long = 0;
    let escaped;
    {
        let short = 0;
        escaped = bounds_are_not_contents(&long, &short);
    }
    println!("escaped = {escaped:?}");
}
