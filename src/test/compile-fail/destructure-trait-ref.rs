// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The regression test for #15031 to make sure destructuring trait
// reference work properly.

trait T {}
impl T for int {}

fn main() {
    // For an expression of the form:
    //
    //      let &...&x = &..&SomeTrait;
    //
    // Say we have n `&` at the left hand and m `&` right hand, then:
    // if n < m, we are golden;
    // if n == m, it's a derefing non-derefable type error;
    // if n > m, it's a type mismatch error.

    // n < m
    let &x = &(&1i as &T);
    let &x = &&(&1i as &T);
    let &&x = &&(&1i as &T);

    // n == m
    let &x = &1i as &T;      //~ ERROR cannot be dereferenced
    let &&x = &(&1i as &T);  //~ ERROR cannot be dereferenced
    let box x = box 1i as Box<T>; //~ ERROR cannot be dereferenced

    // n > m
    let &&x = &1i as &T;     //~ ERROR found an `&`-pointer pattern
    let &&&x = &(&1i as &T); //~ ERROR found an `&`-pointer pattern
    let box box x = box 1i as Box<T>;    //~ ERROR found a box pattern
}
