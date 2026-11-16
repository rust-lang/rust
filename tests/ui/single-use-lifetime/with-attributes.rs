//! Regression test for https://github.com/rust-lang/rust/issues/105031

//@ run-rustfix
//@ check-pass

#![feature(dropck_eyepatch)]
#![warn(single_use_lifetimes)]
#![allow(unused)]

struct S1(&'static str);

struct D1<'a, 'b>(&'a S1, &'b S1);

unsafe impl<'a, #[may_dangle] 'b> Drop for D1<'a, 'b> {
    //~^ WARN lifetime parameter `'a` only used once
    fn drop(&mut self) {
        let _: Option<&'b S1> = None;
    }
}

struct D2<'a, 'b>(&'a S1, &'b S1);

unsafe impl<#[may_dangle] 'a, 'b> Drop for D2<'a, 'b> {
    //~^ WARN lifetime parameter `'b` only used once
    fn drop(&mut self) {
        let _: Option<&'a S1> = None;
    }
}

fn main() {}
