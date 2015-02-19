// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo : ::std::marker::MarkerTrait {
    type X;
    type Y;
}

fn have_x_want_x<T:Foo<X=u32>>(t: &T)
{
    want_x(t);
}

fn have_x_want_y<T:Foo<X=u32>>(t: &T)
{
    want_y(t); //~ ERROR type mismatch
}

fn have_y_want_x<T:Foo<Y=i32>>(t: &T)
{
    want_x(t); //~ ERROR type mismatch
}

fn have_y_want_y<T:Foo<Y=i32>>(t: &T)
{
    want_y(t);
}

fn have_xy_want_x<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_x(t);
}

fn have_xy_want_y<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_y(t);
}

fn have_xy_want_xy<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_x(t);
    want_y(t);
}

fn want_x<T:Foo<X=u32>>(t: &T) { }

fn want_y<T:Foo<Y=i32>>(t: &T) { }

fn main() { }
