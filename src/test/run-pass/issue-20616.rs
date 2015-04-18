// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type MyType<'a, T> = &'a T;

// combine lifetime bounds and type arguments in usual way
type TypeA<'a> = MyType<'a, ()>;

// ensure token `>>` works fine
type TypeB = Box<TypeA<'static>>;
type TypeB_ = Box<TypeA<'static,>>;

// trailing comma when combine lifetime bounds and type arguments
type TypeC<'a> = MyType<'a, (),>;

// normal lifetime bounds
type TypeD = TypeA<'static>;

// trailing comma on lifetime bounds
type TypeE = TypeA<'static,>;

// normal type arugment
type TypeF<T> = Box<T>;

// type argument with trailing comma
type TypeG<T> = Box<T,>;

// trailing comma on liftime defs
type TypeH<'a,> = &'a ();

// trailing comma on type argument
type TypeI<T,> = T;

static STATIC: () = ();

fn main() {

    // ensure token `>=` works fine
    let _: TypeA<'static>= &STATIC;
    let _: TypeA<'static,>= &STATIC;

    // ensure token `>>=` works fine
    let _: Box<TypeA<'static>>= Box::new(&STATIC);
    let _: Box<TypeA<'static,>>= Box::new(&STATIC);
}
