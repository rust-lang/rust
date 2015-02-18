// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a where clause that uses a non-normalized projection type.

trait Int
{
    type T;

    fn dummy(&self) { }
}

trait NonZero
{
    fn non_zero(self) -> bool;
}

fn foo<I:Int<T=J>,J>(t: I) -> bool
    where <I as Int>::T : NonZero
    //    ^~~~~~~~~~~~~ canonical form is just J
{
    bar::<J>()
}

fn bar<NZ:NonZero>() -> bool { true }

fn main ()
{
}
