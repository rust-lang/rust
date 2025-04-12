#![allow(clippy::boxed_local, clippy::disallowed_names)]

pub struct MyStruct;

pub struct SubT<T> {
    foo: T,
}

mod outer_box {
    use crate::{MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn box_test6<T>(foo: Box<Rc<T>>) {}
    //~^ redundant_allocation

    pub fn box_test7<T>(foo: Box<Arc<T>>) {}
    //~^ redundant_allocation

    pub fn box_test8() -> Box<Rc<SubT<usize>>> {
        //~^ redundant_allocation

        unimplemented!();
    }

    pub fn box_test9<T>(foo: Box<Arc<T>>) -> Box<Arc<SubT<T>>> {
        //~^ redundant_allocation
        //~| redundant_allocation

        unimplemented!();
    }
}

mod outer_rc {
    use crate::{MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn rc_test5(a: Rc<Box<bool>>) {}
    //~^ redundant_allocation

    pub fn rc_test7(a: Rc<Arc<bool>>) {}
    //~^ redundant_allocation

    pub fn rc_test8() -> Rc<Box<SubT<usize>>> {
        //~^ redundant_allocation

        unimplemented!();
    }

    pub fn rc_test9<T>(foo: Rc<Arc<T>>) -> Rc<Arc<SubT<T>>> {
        //~^ redundant_allocation
        //~| redundant_allocation

        unimplemented!();
    }
}

mod outer_arc {
    use crate::{MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn arc_test5(a: Arc<Box<bool>>) {}
    //~^ redundant_allocation

    pub fn arc_test6(a: Arc<Rc<bool>>) {}
    //~^ redundant_allocation

    pub fn arc_test8() -> Arc<Box<SubT<usize>>> {
        //~^ redundant_allocation

        unimplemented!();
    }

    pub fn arc_test9<T>(foo: Arc<Rc<T>>) -> Arc<Rc<SubT<T>>> {
        //~^ redundant_allocation
        //~| redundant_allocation

        unimplemented!();
    }
}

// https://github.com/rust-lang/rust-clippy/issues/7487
mod box_dyn {
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub trait T {}

    struct S {
        a: Box<Box<dyn T>>,
        b: Rc<Box<dyn T>>,
        c: Arc<Box<dyn T>>,
    }

    pub fn test_box(_: Box<Box<dyn T>>) {}
    pub fn test_rc(_: Rc<Box<dyn T>>) {}
    pub fn test_arc(_: Arc<Box<dyn T>>) {}
    pub fn test_rc_box(_: Rc<Box<Box<dyn T>>>) {}
    //~^ redundant_allocation
}

// https://github.com/rust-lang/rust-clippy/issues/8604
mod box_fat_ptr {
    use std::boxed::Box;
    use std::path::Path;
    use std::rc::Rc;
    use std::sync::Arc;

    pub struct DynSized {
        foo: [usize],
    }

    struct S {
        a: Box<Box<str>>,
        b: Rc<Box<str>>,
        c: Arc<Box<str>>,

        e: Box<Box<[usize]>>,
        f: Box<Box<Path>>,
        g: Box<Box<DynSized>>,
    }

    pub fn test_box_str(_: Box<Box<str>>) {}
    pub fn test_rc_str(_: Rc<Box<str>>) {}
    pub fn test_arc_str(_: Arc<Box<str>>) {}

    pub fn test_box_slice(_: Box<Box<[usize]>>) {}
    pub fn test_box_path(_: Box<Box<Path>>) {}
    pub fn test_box_custom(_: Box<Box<DynSized>>) {}

    pub fn test_rc_box_str(_: Rc<Box<Box<str>>>) {}
    //~^ redundant_allocation

    pub fn test_rc_box_slice(_: Rc<Box<Box<[usize]>>>) {}
    //~^ redundant_allocation

    pub fn test_rc_box_path(_: Rc<Box<Box<Path>>>) {}
    //~^ redundant_allocation

    pub fn test_rc_box_custom(_: Rc<Box<Box<DynSized>>>) {}
    //~^ redundant_allocation
}

// https://github.com/rust-lang/rust-clippy/issues/11417
fn type_in_closure() {
    let _ = |_: &mut Box<Box<dyn ToString>>| {};
}

fn main() {}
