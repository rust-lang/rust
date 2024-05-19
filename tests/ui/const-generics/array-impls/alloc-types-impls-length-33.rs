//@ check-pass

use std::{convert::TryFrom, rc::Rc, sync::Arc};

pub fn yes_vec() {
    let v: Vec<_> = [0; 33].into();
}

pub fn yes_box() {
    let boxed_slice = Box::new([0; 33]) as Box<[i32]>;
    let boxed_array = <Box<[i32; 33]>>::try_from(boxed_slice);
    let boxed_slice = <Box<[i32]>>::from([0; 33]);
}

pub fn yes_rc() {
    let boxed_slice = Rc::new([0; 33]) as Rc<[i32]>;
    let boxed_array = <Rc<[i32; 33]>>::try_from(boxed_slice);
}

pub fn yes_arc() {
    let boxed_slice = Arc::new([0; 33]) as Arc<[i32]>;
    let boxed_array = <Arc<[i32; 33]>>::try_from(boxed_slice);
}

fn main() {}
