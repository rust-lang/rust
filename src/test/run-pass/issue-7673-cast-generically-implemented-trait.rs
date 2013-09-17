// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

#7673 Polymorphically creating traits barely works

*/

fn main() {}

trait A {}
impl<T: 'static> A for T {}

fn owned1<T: 'static>(a: T) { ~a as ~A:; } /* note `:` */
fn owned2<T: 'static>(a: ~T) { a as ~A:; }
fn owned3<T: 'static>(a: ~T) { ~a as ~A:; }

fn managed1<T: 'static>(a: T) { @a as @A; }
fn managed2<T: 'static>(a: @T) { a as @A; }
fn managed3<T: 'static>(a: @T) { @a as @A; }
