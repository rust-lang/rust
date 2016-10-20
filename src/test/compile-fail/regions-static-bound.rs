// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn static_id<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    //~^ NOTE ...but the borrowed content is only valid for the lifetime 'a
    t //~ ERROR E0312
    //~^ NOTE ...the reference is valid for the static lifetime...
}

fn error(u: &(), v: &()) {
    //~^ NOTE the lifetime cannot outlive the anonymous lifetime #1 defined on the block
    //~| NOTE the lifetime cannot outlive the anonymous lifetime #2 defined on the block
    static_id(&u);
    //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter
    //~| NOTE cannot infer an appropriate lifetime
    //~| NOTE ...so that reference does not outlive borrowed content
    //~| NOTE ...so that the declared lifetime parameter bounds are satisfied
    //~| NOTE the lifetime must be valid for the static lifetime
    //~| NOTE the lifetime must be valid for the static lifetime
    static_id_indirect(&v);
    //~^ ERROR cannot infer an appropriate lifetime
    //~| ERROR cannot infer an appropriate lifetime
    //~| NOTE cannot infer an appropriate lifetime
    //~| NOTE ...so that reference does not outlive borrowed content
    //~| NOTE ...so that the declared lifetime parameter bounds are satisfied
    //~| NOTE the lifetime must be valid for the static lifetime
    //~| NOTE the lifetime must be valid for the static lifetime
}

fn main() {}
