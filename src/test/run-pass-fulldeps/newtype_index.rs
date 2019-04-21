#![feature(rustc_attrs, rustc_private, step_trait)]

#[macro_use] extern crate rustc_data_structures;
extern crate serialize as rustc_serialize;

use rustc_data_structures::indexed_vec::Idx;

newtype_index!(struct MyIdx { MAX = 0xFFFF_FFFA });

use std::mem::size_of;

fn main() {
    assert_eq!(size_of::<MyIdx>(), 4);
    assert_eq!(size_of::<Option<MyIdx>>(), 4);
    assert_eq!(size_of::<Option<Option<MyIdx>>>(), 4);
    assert_eq!(size_of::<Option<Option<Option<MyIdx>>>>(), 4);
    assert_eq!(size_of::<Option<Option<Option<Option<MyIdx>>>>>(), 4);
    assert_eq!(size_of::<Option<Option<Option<Option<Option<MyIdx>>>>>>(), 4);
    assert_eq!(size_of::<Option<Option<Option<Option<Option<Option<MyIdx>>>>>>>(), 8);
}
