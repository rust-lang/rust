// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(irrefutable_match_without_binding)];
#[allow(unused_variable)];

fn main() {
    let a = 8;
    let _ = a; //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
    let _ = 99; //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
    let (_, _) = (1, 2); //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
    let ((_, _), ()) = (((), ()), ()); //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
    let () = (); //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
    match 23 { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        _ => { }
    }
    match (1, 2) { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        (_, _) => { }
    }
    match false { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        false|true => { }
    }
    match [1, 2] { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        [..] => { }
    }
    match [1, 2] { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        [_, _] => { }
    }
    match ~666 { //~ ERROR: irrefutable let or match without any bindings (equivalent to inner expression)
        ~_ => { }
    }

    match false { // okay: more than one arm
        false => { },
        _ => { }
    }
    match [1, 2] {
        [..v] => { }
    }
    let ((_, (_, a)), ()) = ((1, (2, 3)), ()); // okay: has a binding
    let ((_, (_, ref a)), ()) = ((1, (2, 3)), ()); // okay: has a binding
    let ~ref x = ~789;
}
