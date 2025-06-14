//! Tests that type parameters with the `Copy` are implicitly copyable.

//@ run-pass

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(dead_code)]

fn can_copy_copy<T: Copy>(v: T) {
    let _a = v;
    let _b = v;
}

pub fn main() {}
