//@ known-bug: #137187
#![feature(const_trait_impl, const_ops)]

use std::ops::Add;
const trait A where
    *const Self: const Add,
{
    fn b(c: *const Self) -> <*const Self as Add>::Output {
        c + c
    }
}

fn main() {}
