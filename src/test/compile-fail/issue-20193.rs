// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(t: &mut int){
    println!("{}", t);
}

fn main() {
    let test = 10;

    let h = move || {  //~error boxed closures can't capture by value
        let mut r = &mut test.clone();
        foo(r);
    };

    h();
}
