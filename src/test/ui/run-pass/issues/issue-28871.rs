// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #28871. The problem is that rustc encountered
// two ways to project, one from a where clause and one from the where
// clauses on the trait definition. (In fact, in this case, the where
// clauses originated from the trait definition as well.) The true
// cause of the error is that the trait definition where clauses are
// not being normalized, and hence the two sources are considered in
// conflict, and not a duplicate. Hacky solution is to prefer where
// clauses over the data found in the trait definition.

trait T {
    type T;
}

struct S;
impl T for S {
    type T = S;
}

trait T2 {
    type T: Iterator<Item=<S as T>::T>;
}

fn main() { }
