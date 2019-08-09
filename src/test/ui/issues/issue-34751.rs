// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// #34751 ICE: 'rustc' panicked at 'assertion failed: !substs.has_regions_escaping_depth(0)'

#[allow(dead_code)]

use std::marker::PhantomData;

fn f<'a>(PhantomData::<&'a u8>: PhantomData<&'a u8>) {}

fn main() {}
