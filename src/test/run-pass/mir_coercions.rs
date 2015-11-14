// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs, coerce_unsized, unsize)]

use std::ops::CoerceUnsized;
use std::marker::Unsize;

#[rustc_mir]
fn identity_coercion(x: &(Fn(u32)->u32 + Send)) -> &Fn(u32)->u32 {
    x
}
#[rustc_mir]
fn fn_coercions(f: &fn(u32) -> u32) ->
    (unsafe fn(u32) -> u32,
     &(Fn(u32) -> u32+Send))
{
    (*f, f)
}

#[rustc_mir]
fn simple_array_coercion(x: &[u8; 3]) -> &[u8] { x }

fn square(a: u32) -> u32 { a * a }

#[derive(PartialEq,Eq)]
struct PtrWrapper<'a, T: 'a+?Sized>(u32, u32, (), &'a T);
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized>
    CoerceUnsized<PtrWrapper<'a, U>> for PtrWrapper<'a, T> {}

struct TrivPtrWrapper<'a, T: 'a+?Sized>(&'a T);
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized>
    CoerceUnsized<TrivPtrWrapper<'a, U>> for TrivPtrWrapper<'a, T> {}

#[rustc_mir]
fn coerce_ptr_wrapper(p: PtrWrapper<[u8; 3]>) -> PtrWrapper<[u8]> {
    p
}

#[rustc_mir]
fn coerce_triv_ptr_wrapper(p: TrivPtrWrapper<[u8; 3]>) -> TrivPtrWrapper<[u8]> {
    p
}

#[rustc_mir]
fn coerce_fat_ptr_wrapper(p: PtrWrapper<Fn(u32) -> u32+Send>)
                          -> PtrWrapper<Fn(u32) -> u32> {
    p
}


fn main() {
    let a = [0,1,2];
    let square_local : fn(u32) -> u32 = square;
    let (f,g) = fn_coercions(&square_local);
    assert_eq!(f as usize, square as usize);
    assert_eq!(g(4), 16);
    assert_eq!(identity_coercion(g)(5), 25);

    assert_eq!(simple_array_coercion(&a), &a);
    let w = coerce_ptr_wrapper(PtrWrapper(2,3,(),&a));
    assert!(w == PtrWrapper(2,3,(),&a) as PtrWrapper<[u8]>);

    let w = coerce_triv_ptr_wrapper(TrivPtrWrapper(&a));
    assert_eq!(&w.0, &a);

    let z = coerce_fat_ptr_wrapper(PtrWrapper(2,3,(),&square_local));
    assert_eq!((z.3)(6), 36);
}
