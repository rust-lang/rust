// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we give suitable error messages when the user attempts to
// impl a trait `Trait` for its own object type.

trait Foo { fn dummy(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

// Supertraits of Baz are not legal:
impl Foo for Baz { }   //~ ERROR E0371
impl Bar for Baz { }   //~ ERROR E0371
impl Baz for Baz { }   //~ ERROR E0371

// But other random traits are:
trait Other { }
impl Other for Baz { } // OK, Other not a supertrait of Baz

fn main() { }
