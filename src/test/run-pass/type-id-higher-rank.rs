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

#![feature(unboxed_closures)]

use std::intrinsics::TypeId;

fn main() {
    // Bare fns
    {
        let a = TypeId::of::<fn(&'static int, &'static int)>();
        let b = TypeId::of::<for<'a> fn(&'static int, &'a int)>();
        let c = TypeId::of::<for<'a, 'b> fn(&'a int, &'b int)>();
        let d = TypeId::of::<for<'a, 'b> fn(&'b int, &'a int)>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<for<'a> fn(fn(&'a int) -> &'a int)>();
        let f = TypeId::of::<fn(for<'a> fn(&'a int) -> &'a int)>();
        assert!(e != f);
    }
    // Boxed unboxed closures
    {
        let a = TypeId::of::<Box<Fn(&'static int, &'static int)>>();
        let b = TypeId::of::<Box<for<'a> Fn(&'static int, &'a int)>>();
        let c = TypeId::of::<Box<for<'a, 'b> Fn(&'a int, &'b int)>>();
        let d = TypeId::of::<Box<for<'a, 'b> Fn(&'b int, &'a int)>>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<Box<for<'a> Fn(Box<Fn(&'a int) -> &'a int>)>>();
        let f = TypeId::of::<Box<Fn(Box<for<'a> Fn(&'a int) -> &'a int>)>>();
        assert!(e != f);
    }
    // Raw unboxed closures
    // Note that every unboxed closure has its own anonymous type,
    // so no two IDs should equal each other, even when compatible
    {
        let a = id(|&: _: &int, _: &int| {});
        let b = id(|&: _: &int, _: &int| {});
        assert!(a != b);
    }

    fn id<T:'static>(_: T) -> TypeId {
        TypeId::of::<T>()
    }
}
