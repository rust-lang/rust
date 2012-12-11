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
// on x86_64: when there is a enum embedded in an
// interior record which is then itself interior to
// something else, shape calculations were off.
extern mod std;
use std::list;
use std::list::list;

enum opt_span {

    //hack (as opposed to option), to make `span` compile
    os_none,
    os_some(@span),
}
type span = {lo: uint, hi: uint, expanded_from: opt_span};
type spanned<T> = { data: T, span: span };
type ty_ = uint;
type path_ = { global: bool, idents: ~[~str], types: ~[@ty] };
type path = spanned<path_>;
type ty = spanned<ty_>;

fn main() {
    let sp: span = {lo: 57451u, hi: 57542u, expanded_from: os_none};
    let t: @ty = @{ data: 3u, span: sp };
    let p_: path_ = { global: true, idents: ~[~"hi"], types: ~[t] };
    let p: path = { data: p_, span: sp };
    let x = { sp: sp, path: p };
    log(error, copy x.path);
    log(error, copy x);
}
