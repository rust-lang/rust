// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #14603: Check for references to type parameters from the
// outer scope (in this case, the trait) used on items in an inner
// scope (in this case, the enum).

trait TraitA<A> {
    fn outer(&self) {
        enum Foo<B> {
            Variance(A)
                //~^ ERROR can't use type parameters from outer function
        }
    }
}

trait TraitB<A> {
    fn outer(&self) {
        struct Foo<B>(A);
                //~^ ERROR can't use type parameters from outer function
    }
}

trait TraitC<A> {
    fn outer(&self) {
        struct Foo<B> { a: A }
                //~^ ERROR can't use type parameters from outer function
    }
}

trait TraitD<A> {
    fn outer(&self) {
        fn foo<B>(a: A) { }
                //~^ ERROR can't use type parameters from outer function
    }
}

fn main() { }
