//@ check-pass
#![allow(dead_code)]
// Regression test for issue #21245. Check that we are able to infer
// the types in these examples correctly. It used to be that
// insufficient type propagation caused the type of the iterator to be
// incorrectly unified with the `*const` type to which it is coerced.


use std::ptr;

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter2(self) -> Self::Iter;
}

impl<I> IntoIterator for I where I: Iterator {
    type Iter = I;

    fn into_iter2(self) -> I {
        self
    }
}

fn desugared_for_loop_bad<T>(v: Vec<T>) {
    match IntoIterator::into_iter2(v.iter()) {
        mut iter => {
            loop {
                match ::std::iter::Iterator::next(&mut iter) {
                    ::std::option::Option::Some(x) => {
                        unsafe { ptr::read(x); }
                    },
                    ::std::option::Option::None => break
                }
            }
        }
    }
}

fn desugared_for_loop_good<T>(v: Vec<T>) {
    match v.iter().into_iter() {
        mut iter => {
            loop {
                match ::std::iter::Iterator::next(&mut iter) {
                    ::std::option::Option::Some(x) => {
                        unsafe { ptr::read(x); }
                    },
                    ::std::option::Option::None => break
                }
            }
        }
    }
}

fn main() {}
