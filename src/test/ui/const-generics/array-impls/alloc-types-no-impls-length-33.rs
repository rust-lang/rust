// ignore-tidy-linelength

use std::{convert::TryFrom, rc::Rc, sync::Arc};

pub fn no_vec() {
    let v: Vec<_> = [0; 33].into();
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_box() {
    let boxed_slice = Box::new([0; 33]) as Box<[i32]>;
    let boxed_array = <Box<[i32; 33]>>::try_from(boxed_slice);
    //~^ ERROR the trait bound `std::boxed::Box<[i32; 33]>: std::convert::From<std::boxed::Box<[i32]>>` is not satisfied
    //~^^ ERROR the trait bound `std::boxed::Box<[i32; 33]>: std::convert::TryFrom<std::boxed::Box<[i32]>>` is not satisfied
    let boxed_slice = <Box<[i32]>>::from([0; 33]);
    //~^ 15:42: 15:49: arrays only have std trait implementations for lengths 0..=32 [E0277]
}

pub fn no_rc() {
    let boxed_slice = Rc::new([0; 33]) as Rc<[i32]>;
    let boxed_array = <Rc<[i32; 33]>>::try_from(boxed_slice);
    //~^ ERROR the trait bound `std::rc::Rc<[i32; 33]>: std::convert::From<std::rc::Rc<[i32]>>` is not satisfied
    //~^^ ERROR the trait bound `std::rc::Rc<[i32; 33]>: std::convert::TryFrom<std::rc::Rc<[i32]>>` is not satisfied
}

pub fn no_arc() {
    let boxed_slice = Arc::new([0; 33]) as Arc<[i32]>;
    let boxed_array = <Arc<[i32; 33]>>::try_from(boxed_slice);
    //~^ ERROR the trait bound `std::sync::Arc<[i32; 33]>: std::convert::From<std::sync::Arc<[i32]>>` is not satisfied
    //~^^ ERROR the trait bound `std::sync::Arc<[i32; 33]>: std::convert::TryFrom<std::sync::Arc<[i32]>>` is not satisfied
}

fn main() {}
