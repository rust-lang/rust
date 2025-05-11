#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::disallowed_names)]

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
    //~^ redundant_allocation

    pub fn box_test2(foo: Box<&MyStruct>) {}
    //~^ redundant_allocation

    pub fn box_test3(foo: Box<&MyEnum>) {}
    //~^ redundant_allocation

    pub fn box_test4_neg(foo: Box<SubT<&usize>>) {}

    pub fn box_test5<T>(foo: Box<Box<T>>) {}
    //~^ redundant_allocation
}

mod outer_rc {
    use crate::{MyEnum, MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn rc_test1<T>(foo: Rc<&T>) {}
    //~^ redundant_allocation

    pub fn rc_test2(foo: Rc<&MyStruct>) {}
    //~^ redundant_allocation

    pub fn rc_test3(foo: Rc<&MyEnum>) {}
    //~^ redundant_allocation

    pub fn rc_test4_neg(foo: Rc<SubT<&usize>>) {}

    pub fn rc_test6(a: Rc<Rc<bool>>) {}
    //~^ redundant_allocation
}

mod outer_arc {
    use crate::{MyEnum, MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn arc_test1<T>(foo: Arc<&T>) {}
    //~^ redundant_allocation

    pub fn arc_test2(foo: Arc<&MyStruct>) {}
    //~^ redundant_allocation

    pub fn arc_test3(foo: Arc<&MyEnum>) {}
    //~^ redundant_allocation

    pub fn arc_test4_neg(foo: Arc<SubT<&usize>>) {}

    pub fn arc_test7(a: Arc<Arc<bool>>) {}
    //~^ redundant_allocation
}

fn main() {}
