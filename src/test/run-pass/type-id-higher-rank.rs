// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that type IDs correctly account for higher-rank lifetimes
// Also acts as a regression test for an ICE (issue #19791)


#![feature(core)]

use std::any::{Any, TypeId};

struct Struct<'a>(&'a ());
trait Trait<'a> {}

fn main() {
    // Bare fns
    {
        let a = TypeId::of::<fn(&'static isize, &'static isize)>();
        let b = TypeId::of::<for<'a> fn(&'static isize, &'a isize)>();
        let c = TypeId::of::<for<'a, 'b> fn(&'a isize, &'b isize)>();
        let d = TypeId::of::<for<'a, 'b> fn(&'b isize, &'a isize)>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<for<'a> fn(fn(&'a isize) -> &'a isize)>();
        let f = TypeId::of::<fn(for<'a> fn(&'a isize) -> &'a isize)>();
        assert!(e != f);

        // Make sure lifetime parameters of items are not ignored.
        let g = TypeId::of::<for<'a> fn(&'a Trait<'a>) -> Struct<'a>>();
        let h = TypeId::of::<for<'a> fn(&'a Trait<'a>) -> Struct<'static>>();
        let i = TypeId::of::<for<'a, 'b> fn(&'a Trait<'b>) -> Struct<'b>>();
        assert!(g != h);
        assert!(g != i);
        assert!(h != i);
    }
    // Boxed unboxed closures
    {
        let a = TypeId::of::<Box<Fn(&'static isize, &'static isize)>>();
        let b = TypeId::of::<Box<for<'a> Fn(&'static isize, &'a isize)>>();
        let c = TypeId::of::<Box<for<'a, 'b> Fn(&'a isize, &'b isize)>>();
        let d = TypeId::of::<Box<for<'a, 'b> Fn(&'b isize, &'a isize)>>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<Box<for<'a> Fn(Box<Fn(&'a isize) -> &'a isize>)>>();
        let f = TypeId::of::<Box<Fn(Box<for<'a> Fn(&'a isize) -> &'a isize>)>>();
        assert!(e != f);
    }
    // Raw unboxed closures
    // Note that every unboxed closure has its own anonymous type,
    // so no two IDs should equal each other, even when compatible
    {
        let a = id(|_: &isize, _: &isize| {});
        let b = id(|_: &isize, _: &isize| {});
        assert!(a != b);
    }

    fn id<T:Any>(_: T) -> TypeId {
        TypeId::of::<T>()
    }
}
