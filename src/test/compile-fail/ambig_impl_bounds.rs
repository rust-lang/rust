// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A { fn foo(); }
trait B { fn foo(); }

fn foo<T: A B>(t: T) {
    t.foo(); //~ ERROR multiple applicable methods in scope
    //~^ NOTE candidate #1 derives from the bound `A`
    //~^^ NOTE candidate #2 derives from the bound `B`
}

fn main() {}