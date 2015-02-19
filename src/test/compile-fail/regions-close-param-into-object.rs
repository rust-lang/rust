// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

trait X { fn foo(&self) {} }

fn p1<T>(v: T) -> Box<X+'static>
    where T : X
{
    box v //~ ERROR parameter type `T` may not live long enough
}

fn p2<T>(v: Box<T>) -> Box<X+'static>
    where Box<T> : X
{
    box v //~ ERROR parameter type `T` may not live long enough
}

fn p3<'a,T>(v: T) -> Box<X+'a>
    where T : X
{
    box v //~ ERROR parameter type `T` may not live long enough
}

fn p4<'a,T>(v: Box<T>) -> Box<X+'a>
    where Box<T> : X
{
    box v //~ ERROR parameter type `T` may not live long enough
}

fn main() {}

