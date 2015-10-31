// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// macro f should not be able to inject a reference to 'n'.

macro_rules! f { () => (n) }

fn main() -> (){
    for n in 0..1 {
        println!("{}", f!()); //~ ERROR unresolved name `n`
    }

    if let Some(n) = None {
        println!("{}", f!()); //~ ERROR unresolved name `n`
    }

    if false {
    } else if let Some(n) = None {
        println!("{}", f!()); //~ ERROR unresolved name `n`
    }

    while let Some(n) = None {
        println!("{}", f!()); //~ ERROR unresolved name `n`
    }
}
