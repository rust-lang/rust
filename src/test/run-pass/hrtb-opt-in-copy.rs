// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we handle binder levels correctly when checking whether a
// type can implement `Copy`. In particular, we had a bug where we failed to
// liberate the late-bound regions from the impl, and thus wound up
// searching for an impl of `for<'tcx> Foo<&'tcx T>`. The impl that
// exists however is `impl<T> Copy for Foo<T>` and the current rules
// did not consider that a match (something I would like to revise in
// a later PR).

#![allow(dead_code)]

use std::marker::PhantomData;

#[derive(Copy)]
struct Foo<T> { x: T }

type Ty<'tcx> = &'tcx TyS<'tcx>;

enum TyS<'tcx> {
    Boop(PhantomData<*mut &'tcx ()>)
}

#[derive(Copy)]
enum Bar<'tcx> {
    Baz(Foo<Ty<'tcx>>)
}

fn main() { }
