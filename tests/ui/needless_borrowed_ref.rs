// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::needless_borrowed_reference)]
#[allow(unused_variables)]
fn main() {
    let mut v = Vec::<String>::new();
    let _ = v.iter_mut().filter(|&ref a| a.is_empty());
    //                            ^ should be linted

    let var = 3;
    let thingy = Some(&var);
    if let Some(&ref v) = thingy {
        //          ^ should be linted
    }

    let mut var2 = 5;
    let thingy2 = Some(&mut var2);
    if let Some(&mut ref mut v) = thingy2 {
        //          ^ should *not* be linted
        // v is borrowed as mutable.
        *v = 10;
    }
    if let Some(&mut ref v) = thingy2 {
        //          ^ should *not* be linted
        // here, v is borrowed as immutable.
        // can't do that:
        //*v = 15;
    }
}

#[allow(dead_code)]
enum Animal {
    Cat(u64),
    Dog(u64),
}

#[allow(unused_variables)]
#[allow(dead_code)]
fn foo(a: &Animal, b: &Animal) {
    match (a, b) {
        (&Animal::Cat(v), &ref k) | (&ref k, &Animal::Cat(v)) => (), // lifetime mismatch error if there is no '&ref'
        //                  ^    and   ^ should *not* be linted
        (&Animal::Dog(ref a), &Animal::Dog(_)) => (), //              ^ should *not* be linted
    }
}
