#![allow(dead_code)]
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
