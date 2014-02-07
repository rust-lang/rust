// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(macro_rules)];

macro_rules! test ( ($nm:ident,
                     $a:attr,
                     $i:item) => (mod $nm { $a; $i }); )

test!(a,
      #[cfg(qux)],
      pub fn bar() { })

test!(b,
      #[cfg(not(qux))],
      pub fn bar() { })

#[qux]
fn main() {
    a::bar();
    //~^ ERROR use of undeclared module `a`
    //~^^ ERROR unresolved name
    //~^^^ ERROR unresolved name `a::bar`
    b::bar();
}

