// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Exercises a bug in the shape code that was exposed
// on x86_64: when there is an enum embedded in an
// interior record which is then itself interior to
// something else, shape calculations were off.

#[deriving(Clone, Show)]
enum opt_span {
    //hack (as opposed to option), to make `span` compile
    os_none,
    os_some(Box<Span>),
}

#[deriving(Clone, Show)]
struct Span {
    lo: uint,
    hi: uint,
    expanded_from: opt_span,
}

#[deriving(Clone, Show)]
struct Spanned<T> {
    data: T,
    span: Span,
}

type ty_ = uint;

#[deriving(Clone, Show)]
struct Path_ {
    global: bool,
    idents: Vec<String> ,
    types: Vec<Box<ty>>,
}

type path = Spanned<Path_>;
type ty = Spanned<ty_>;

#[deriving(Clone, Show)]
struct X {
    sp: Span,
    path: path,
}

pub fn main() {
    let sp: Span = Span {lo: 57451u, hi: 57542u, expanded_from: os_none};
    let t: Box<ty> = box Spanned { data: 3u, span: sp.clone() };
    let p_: Path_ = Path_ {
        global: true,
        idents: vec!("hi".to_string()),
        types: vec!(t),
    };
    let p: path = Spanned { data: p_, span: sp.clone() };
    let x = X { sp: sp, path: p };
    println!("{}", x.path.clone());
    println!("{}", x.clone());
}
