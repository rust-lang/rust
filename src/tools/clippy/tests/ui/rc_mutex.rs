#![warn(clippy::rc_mutex)]
#![allow(clippy::blacklisted_name)]

use std::rc::Rc;
use std::sync::Mutex;

pub struct MyStruct {
    foo: Rc<Mutex<i32>>,
}

pub struct SubT<T> {
    foo: T,
}

pub enum MyEnum {
    One,
    Two,
}

pub fn test1<T>(foo: Rc<Mutex<T>>) {}

pub fn test2(foo: Rc<Mutex<MyEnum>>) {}

pub fn test3(foo: Rc<Mutex<SubT<usize>>>) {}

fn main() {
    test1(Rc::new(Mutex::new(1)));
    test2(Rc::new(Mutex::new(MyEnum::One)));
    test3(Rc::new(Mutex::new(SubT { foo: 1 })));

    let _my_struct = MyStruct {
        foo: Rc::new(Mutex::new(1)),
    };
}
