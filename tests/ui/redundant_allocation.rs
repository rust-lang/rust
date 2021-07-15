#![warn(clippy::all)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::blacklisted_name, unused_variables, dead_code)]
#![allow(unused_imports)]

pub struct MyStruct {}

pub struct SubT<T> {
    foo: T,
}

pub enum MyEnum {
    One,
    Two,
}

mod outer_box {
    use crate::MyEnum;
    use crate::MyStruct;
    use crate::SubT;
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn box_test6<T>(foo: Box<Rc<T>>) {}

    pub fn box_test7<T>(foo: Box<Arc<T>>) {}

    pub fn box_test8() -> Box<Rc<SubT<usize>>> {
        unimplemented!();
    }

    pub fn box_test9<T>(foo: Box<Arc<T>>) -> Box<Arc<SubT<T>>> {
        unimplemented!();
    }
}

mod outer_rc {
    use crate::MyEnum;
    use crate::MyStruct;
    use crate::SubT;
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn rc_test5(a: Rc<Box<bool>>) {}

    pub fn rc_test7(a: Rc<Arc<bool>>) {}

    pub fn rc_test8() -> Rc<Box<SubT<usize>>> {
        unimplemented!();
    }

    pub fn rc_test9<T>(foo: Rc<Arc<T>>) -> Rc<Arc<SubT<T>>> {
        unimplemented!();
    }
}

mod outer_arc {
    use crate::MyEnum;
    use crate::MyStruct;
    use crate::SubT;
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn arc_test5(a: Arc<Box<bool>>) {}

    pub fn arc_test6(a: Arc<Rc<bool>>) {}

    pub fn arc_test8() -> Arc<Box<SubT<usize>>> {
        unimplemented!();
    }

    pub fn arc_test9<T>(foo: Arc<Rc<T>>) -> Arc<Rc<SubT<T>>> {
        unimplemented!();
    }
}

fn main() {}
