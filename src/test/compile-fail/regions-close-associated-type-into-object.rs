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

use std::marker::MarkerTrait;

trait X : MarkerTrait {}

trait Iter {
    type Item: X;

    fn into_item(self) -> Self::Item;
    fn as_item(&self) -> &Self::Item;
}

fn bad1<T: Iter>(v: T) -> Box<X+'static>
{
    let item = v.into_item();
    box item //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad2<T: Iter>(v: T) -> Box<X+'static>
    where Box<T::Item> : X
{
    let item = box v.into_item();
    box item //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad3<'a, T: Iter>(v: T) -> Box<X+'a>
{
    let item = v.into_item();
    box item //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad4<'a, T: Iter>(v: T) -> Box<X+'a>
    where Box<T::Item> : X
{
    let item = box v.into_item();
    box item //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn ok1<'a, T: Iter>(v: T) -> Box<X+'a>
    where T::Item : 'a
{
    let item = v.into_item();
    box item // OK, T::Item : 'a is declared
}

fn ok2<'a, T: Iter>(v: &T, w: &'a T::Item) -> Box<X+'a>
    where T::Item : Clone
{
    let item = Clone::clone(w);
    box item // OK, T::Item : 'a is implied
}

fn ok3<'a, T: Iter>(v: &'a T) -> Box<X+'a>
    where T::Item : Clone + 'a
{
    let item = Clone::clone(v.as_item());
    box item // OK, T::Item : 'a was declared
}

fn meh1<'a, T: Iter>(v: &'a T) -> Box<X+'a>
    where T::Item : Clone
{
    // This case is kind of interesting. It's the same as `ok3` but
    // without the explicit declaration. In principle, it seems like
    // we ought to be able to infer that `T::Item : 'a` because we
    // invoked `v.as_self()` which yielded a value of type `&'a
    // T::Item`. But we're not that smart at present.

    let item = Clone::clone(v.as_item());
    box item //~ ERROR associated type `<T as Iter>::Item` may not live
}

fn main() {}

