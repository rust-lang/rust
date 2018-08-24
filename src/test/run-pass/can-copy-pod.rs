// pretty-expanded FIXME #23616

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

// Tests that type parameters with the `Copy` are implicitly copyable.

#![allow(dead_code)]

fn can_copy_copy<T:Copy>(v: T) {
    let _a = v;
    let _b = v;
}

pub fn main() {}
