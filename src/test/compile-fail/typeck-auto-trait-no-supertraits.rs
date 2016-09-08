// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is for #29859, we need to ensure auto traits,
// (also known previously as default traits), do not have
// supertraits. Since the compiler synthesizes these
// instances on demand, we are essentially enabling
// users to write axioms if we view trait selection,
// as a proof system.
//
// For example the below test allows us to add the rule:
//  forall (T : Type), T : Copy
//
// Providing a copy instance for *any* type, which
// is most definitely unsound. Imagine copying a
// type that contains a mutable reference, enabling
// mutable aliasing.
//
// You can imagine an even more dangerous test,
// which currently compiles on nightly.
//
// fn main() {
//    let mut i = 10;
//    let (a, b) = copy(&mut i);
//    println!("{:?} {:?}", a, b);
// }

#![feature(optin_builtin_traits)]

trait Magic: Copy {} //~ ERROR E0568
impl Magic for .. {}
impl<T:Magic> Magic for T {}

fn copy<T: Magic>(x: T) -> (T, T) { (x, x) }

#[derive(Debug)]
struct NoClone;

fn main() {
    let (a, b) = copy(NoClone);
    println!("{:?} {:?}", a, b);
}
