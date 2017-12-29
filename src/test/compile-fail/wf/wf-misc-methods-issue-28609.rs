// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that misc. method calls are well-formed

use std::marker::PhantomData;
use std::ops::{Deref, Shl};

#[derive(Copy, Clone)]
struct S<'a, 'b: 'a> {
    marker: PhantomData<&'a &'b ()>,
    bomb: Option<&'b u32>
}

type S2<'a> = S<'a, 'a>;

impl<'a, 'b> S<'a, 'b> {
    fn transmute_inherent(&self, a: &'b u32) -> &'a u32 {
        a
    }
}

fn return_dangling_pointer_inherent(s: S2) -> &u32 {
    let s = s;
    s.transmute_inherent(&mut 42) //~ ERROR does not live long enough
}

impl<'a, 'b> Deref for S<'a, 'b> {
    type Target = &'a u32;
    fn deref(&self) -> &&'a u32 {
        self.bomb.as_ref().unwrap()
    }
}

fn return_dangling_pointer_coerce(s: S2) -> &u32 {
    let four = 4;
    let mut s = s;
    s.bomb = Some(&four); //~ ERROR does not live long enough
    &s
}

fn return_dangling_pointer_unary_op(s: S2) -> &u32 {
    let four = 4;
    let mut s = s;
    s.bomb = Some(&four); //~ ERROR does not live long enough
    &*s
}

impl<'a, 'b> Shl<&'b u32> for S<'a, 'b> {
    type Output = &'a u32;
    fn shl(self, t: &'b u32) -> &'a u32 { t }
}

fn return_dangling_pointer_binary_op(s: S2) -> &u32 {
    let s = s;
    s << &mut 3 //~ ERROR does not live long enough
}

fn return_dangling_pointer_method(s: S2) -> &u32 {
    let s = s;
    s.shl(&mut 3) //~ ERROR does not live long enough
}

fn return_dangling_pointer_ufcs(s: S2) -> &u32 {
    let s = s;
    S2::shl(s, &mut 3) //~ ERROR does not live long enough
}

fn main() {
    let s = S { marker: PhantomData, bomb: None };
    let _inherent_dp = return_dangling_pointer_inherent(s);
    let _coerce_dp = return_dangling_pointer_coerce(s);
    let _unary_dp = return_dangling_pointer_unary_op(s);
    let _binary_dp = return_dangling_pointer_binary_op(s);
    let _method_dp = return_dangling_pointer_method(s);
    let _ufcs_dp = return_dangling_pointer_ufcs(s);
}
