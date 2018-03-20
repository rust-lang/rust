// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_49482_macro_def.rs
// aux-build:issue_49482_reexport.rs
// revisions: rpass1

extern crate issue_49482_reexport;

pub trait KvStorage
{
    fn get(&self);
}

impl<K> KvStorage for Box<K>
where
    K: KvStorage + ?Sized,
{
    fn get(&self) {
        (**self).get()
    }
}

impl KvStorage for u32 {
    fn get(&self) {}
}

fn main() {
    /* force issue_49482_reexport to be loaded */
    issue_49482_reexport::foo();

    Box::new(2).get();
}
