// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ll nll
//[nll] compile-flags: -Znll -Zborrowck=mir

fn static_id<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    t //[ll]~ ERROR E0312
        //[nll]~^ WARNING not reporting region error due to -Znll
        //[nll]~| ERROR free region `'a` does not outlive free region `'static`
}

fn error(u: &(), v: &()) {
    static_id(&u); //[ll]~ ERROR cannot infer an appropriate lifetime
    //[nll]~^ WARNING not reporting region error due to -Znll
    //[nll]~| ERROR free region `'_#1r` does not outlive free region `'static`
    static_id_indirect(&v); //[ll]~ ERROR cannot infer an appropriate lifetime
    //[nll]~^ WARNING not reporting region error due to -Znll
    //[nll]~| ERROR free region `'_#2r` does not outlive free region `'static`
}

fn main() {}
