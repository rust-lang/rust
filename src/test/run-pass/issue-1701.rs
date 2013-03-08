// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum pattern { tabby, tortoiseshell, calico }
enum breed { beagle, rottweiler, pug }
type name = ~str;
enum ear_kind { lop, upright }
enum animal { cat(pattern), dog(breed), rabbit(name, ear_kind), tiger }

fn noise(a: animal) -> Option<~str> {
    match a {
      cat(*)    => { Some(~"meow") }
      dog(*)    => { Some(~"woof") }
      rabbit(*) => { None }
      tiger(*)  => { Some(~"roar") }
    }
}

pub fn main() {
    fail_unless!(noise(cat(tabby)) == Some(~"meow"));
    fail_unless!(noise(dog(pug)) == Some(~"woof"));
    fail_unless!(noise(rabbit(~"Hilbert", upright)) == None);
    fail_unless!(noise(tiger) == Some(~"roar"));
}
