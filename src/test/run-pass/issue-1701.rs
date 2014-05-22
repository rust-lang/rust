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
type name = String;
enum ear_kind { lop, upright }
enum animal { cat(pattern), dog(breed), rabbit(name, ear_kind), tiger }

fn noise(a: animal) -> Option<String> {
    match a {
      cat(..)    => { Some("meow".to_strbuf()) }
      dog(..)    => { Some("woof".to_strbuf()) }
      rabbit(..) => { None }
      tiger(..)  => { Some("roar".to_strbuf()) }
    }
}

pub fn main() {
    assert_eq!(noise(cat(tabby)), Some("meow".to_strbuf()));
    assert_eq!(noise(dog(pug)), Some("woof".to_strbuf()));
    assert_eq!(noise(rabbit("Hilbert".to_strbuf(), upright)), None);
    assert_eq!(noise(tiger), Some("roar".to_strbuf()));
}
