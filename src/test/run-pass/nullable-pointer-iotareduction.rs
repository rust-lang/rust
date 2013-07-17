// xfail-test

// xfail'd due to a bug in move detection for macros.

// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{option, cast};

// Iota-reduction is a rule in the Calculus of (Co-)Inductive Constructions,
// which "says that a destructor applied to an object built from a constructor
// behaves as expected".  -- http://coq.inria.fr/doc/Reference-Manual006.html
//
// It's a little more complicated here, because of pointers and regions and
// trying to get assert failure messages that at least identify which case
// failed.

enum E<T> { Thing(int, T), Nothing((), ((), ()), [i8, ..0]) }
impl<T> E<T> {
    fn is_none(&self) -> bool {
        match *self {
            Thing(*) => false,
            Nothing(*) => true
        }
    }
    fn get_ref<'r>(&'r self) -> (int, &'r T) {
        match *self {
            Nothing(*) => fail!("E::get_ref(Nothing::<%s>)",  stringify!($T)),
            Thing(x, ref y) => (x, y)
        }
    }
}

macro_rules! check_option {
    ($e:expr: $T:ty) => {{
        check_option!($e: $T, |ptr| assert!(*ptr == $e));
    }};
    ($e:expr: $T:ty, |$v:ident| $chk:expr) => {{
        assert!(option::None::<$T>.is_none());
        let e = $e;
        let s_ = option::Some::<$T>(e);
        let $v = s_.get_ref();
        $chk
    }}
}

macro_rules! check_fancy {
    ($e:expr: $T:ty) => {{
        check_fancy!($e: $T, |ptr| assert!(*ptr == $e));
    }};
    ($e:expr: $T:ty, |$v:ident| $chk:expr) => {{
        assert!(Nothing::<$T>((), ((), ()), [23i8, ..0]).is_none());
        let e = $e;
        let t_ = Thing::<$T>(23, e);
        match t_.get_ref() {
            (23, $v) => { $chk }
            _ => fail!("Thing::<%s>(23, %s).get_ref() != (23, _)",
                       stringify!($T), stringify!($e))
        }
    }}
}

macro_rules! check_type {
    ($($a:tt)*) => {{
        check_option!($($a)*);
        check_fancy!($($a)*);
    }}
}

pub fn main() {
    check_type!(&17: &int);
    check_type!(~18: ~int);
    check_type!(@19: @int);
    check_type!(~"foo": ~str);
    check_type!(@"bar": @str);
    check_type!(~[20, 22]: ~[int]);
    check_type!(@[]: @[int]);
    check_type!(@[24, 26]: @[int]);
    let mint: uint = unsafe { cast::transmute(main) };
    check_type!(main: extern fn(), |pthing| {
        assert!(mint == unsafe { cast::transmute(*pthing) })
    });
}
