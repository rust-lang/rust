// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

extern crate serialize;

use serialize::{Encodable, Decodable};
use std::fmt::Display;

pub trait Entity : Decodable + Encodable + Sized {
    type Key: Clone + Decodable + Encodable + ToString + Display + Eq + Ord + Sized;

    fn id(&self) -> Self::Key;

    fn find_by_id(id: Self::Key) -> Option<Self>;
}

pub struct DbRef<E: Entity> {
    pub id: E::Key,
}

impl<E> DbRef<E> where E: Entity {
    fn get(self) -> Option<E> {
        E::find_by_id(self.id)
    }
}

fn main() {}
