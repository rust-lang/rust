// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]

trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

fn method1<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _: <T as Trait<'a>>::Type = a;
}

fn method2<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _: <T as Trait<'b>>::Type = a; //~ ERROR E0623
}

fn method3<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _: <T as Trait<'a>>::Type = b; //~ ERROR E0623
}

fn method4<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _: <T as Trait<'b>>::Type = b;
}

fn main() { }
