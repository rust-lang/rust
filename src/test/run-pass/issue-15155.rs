// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait TraitWithSend: Send {}
trait IndirectTraitWithSend: TraitWithSend {}

// Check struct instantiation (Box<TraitWithSend> will only have Send if TraitWithSend has Send)
#[allow(dead_code)]
struct Blah { x: Box<TraitWithSend> }
impl TraitWithSend for Blah {}

// Struct instantiation 2-levels deep
#[allow(dead_code)]
struct IndirectBlah { x: Box<IndirectTraitWithSend> }
impl TraitWithSend for IndirectBlah {}
impl IndirectTraitWithSend for IndirectBlah {}

fn test_trait<Sized? T: Send>() { println!("got here!") }

fn main() {
    test_trait::<TraitWithSend>();
    test_trait::<IndirectTraitWithSend>();
}


