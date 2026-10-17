//@ compile-flags: -Znext-solver
//@ build-pass
//@ needs-sanitizer-kcfi
//@ no-prefer-dynamic
//@ compile-flags: -Cpanic=abort -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer

// KCFI sanitizer works with unsafe binders.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::wrap_binder;

pub fn take(_: unsafe<'a> &'a u8) {}

fn main() {
    let f: fn(unsafe<'a> &'a u8) = take;
    f(unsafe { wrap_binder!(&0) });
}
