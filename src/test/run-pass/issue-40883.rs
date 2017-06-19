// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we don't have linear stack usage with multiple calls to `push`
// min-llvm-version 4.0

#![feature(test)]

extern crate test;
use std::mem;

fn meal() -> Big {
    if test::black_box(false) {
        panic!()
    }
    Big { drop_me: [
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
    ]}
}

pub struct Big {
    drop_me: [Option<Box<u8>>; 48],
}

#[inline]
fn push(out: &mut Vec<Big>) {
    out.push(meal());
}

#[inline(never)]
pub fn supersize_me(out: &mut Vec<Big>) {
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out); // 16 calls to `push`

    verify_stack_usage(out);

    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out);
    push(out); // 16 calls to `push`
}

#[inline(never)]
fn verify_stack_usage(before_ptr: *mut Vec<Big>) {
    // to check stack usage, create locals before and after
    // and check the difference in addresses between them.
    let mut stack_var: Vec<Big> = vec![];
    test::black_box(&mut stack_var);
    let stack_usage = isize::abs(
        (&mut stack_var as *mut _ as isize) -
            (before_ptr as isize)) as usize;
    // give space for 2 copies of `Big` + 128 "misc" bytes.
    if stack_usage > mem::size_of::<Big>() * 2 + 128 {
        panic!("used {} bytes of stack, but `struct Big` is only {} bytes",
               stack_usage, mem::size_of::<Big>());
    }

}

pub fn main() {
    let mut v = vec![];
    test::black_box(&mut v);
    supersize_me(&mut v);
}
