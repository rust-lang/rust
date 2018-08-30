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
      animal::cat(..)    => { Some("meow".to_string()) }
      animal::dog(..)    => { Some("woof".to_string()) }
      animal::rabbit(..) => { None }
      animal::tiger  => { Some("roar".to_string()) }
    }
}

pub fn main() {
    assert_eq!(noise(animal::cat(pattern::tabby)), Some("meow".to_string()));
    assert_eq!(noise(animal::dog(breed::pug)), Some("woof".to_string()));
    assert_eq!(noise(animal::rabbit("Hilbert".to_string(), ear_kind::upright)), None);
    assert_eq!(noise(animal::tiger), Some("roar".to_string()));
}
