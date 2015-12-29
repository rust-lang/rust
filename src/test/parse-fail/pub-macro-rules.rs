// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use] mod bleh {
    pub macro_rules! foo { //~ ERROR can't qualify macro_rules invocation with `pub`
    //~^ HELP did you mean #[macro_export]?
        ($n:ident) => (
            fn $n () -> i32 {
                1
            }
        )
    }

}

foo!(meh);

fn main() {
    println!("{}", meh());
}
