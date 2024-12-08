#![warn(clippy::all)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::disallowed_names, unused_variables, dead_code)]
#![allow(unused_imports)]

pub struct MyStruct;

pub struct SubT<T> {
    foo: T,
}

pub enum MyEnum {
    One,
    Two,
}

mod outer_box {
    use crate::{MyEnum, MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn box_test1<T>(foo: Box<&T>) {}

    pub fn box_test2(foo: Box<&MyStruct>) {}

    pub fn box_test3(foo: Box<&MyEnum>) {}

    pub fn box_test4_neg(foo: Box<SubT<&usize>>) {}

    pub fn box_test5<T>(foo: Box<Box<T>>) {}
}

mod outer_rc {
    use crate::{MyEnum, MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn rc_test1<T>(foo: Rc<&T>) {}

    pub fn rc_test2(foo: Rc<&MyStruct>) {}

    pub fn rc_test3(foo: Rc<&MyEnum>) {}

    pub fn rc_test4_neg(foo: Rc<SubT<&usize>>) {}

    pub fn rc_test6(a: Rc<Rc<bool>>) {}
}

mod outer_arc {
    use crate::{MyEnum, MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn arc_test1<T>(foo: Arc<&T>) {}

    pub fn arc_test2(foo: Arc<&MyStruct>) {}

    pub fn arc_test3(foo: Arc<&MyEnum>) {}

    pub fn arc_test4_neg(foo: Arc<SubT<&usize>>) {}

    pub fn arc_test7(a: Arc<Arc<bool>>) {}
}

fn main() {}
