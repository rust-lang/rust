#![warn(clippy::all)]
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
    //~^ ERROR: usage of `Box<Rc<T>>`
    //~| NOTE: `Rc<T>` is already on the heap, `Box<Rc<T>>` makes an extra allocation

    pub fn box_test7<T>(foo: Box<Arc<T>>) {}
    //~^ ERROR: usage of `Box<Arc<T>>`
    //~| NOTE: `Arc<T>` is already on the heap, `Box<Arc<T>>` makes an extra allocation

    pub fn box_test8() -> Box<Rc<SubT<usize>>> {
        //~^ ERROR: usage of `Box<Rc<SubT<usize>>>`
        //~| NOTE: `Rc<SubT<usize>>` is already on the heap, `Box<Rc<SubT<usize>>>` makes an e
        unimplemented!();
    }

    pub fn box_test9<T>(foo: Box<Arc<T>>) -> Box<Arc<SubT<T>>> {
        //~^ ERROR: usage of `Box<Arc<T>>`
        //~| NOTE: `Arc<T>` is already on the heap, `Box<Arc<T>>` makes an extra allocation
        //~| ERROR: usage of `Box<Arc<SubT<T>>>`
        //~| NOTE: `Arc<SubT<T>>` is already on the heap, `Box<Arc<SubT<T>>>` makes an extra a
        unimplemented!();
    }
}

mod outer_rc {
    use crate::{MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn rc_test5(a: Rc<Box<bool>>) {}
    //~^ ERROR: usage of `Rc<Box<bool>>`
    //~| NOTE: `Box<bool>` is already on the heap, `Rc<Box<bool>>` makes an extra allocati

    pub fn rc_test7(a: Rc<Arc<bool>>) {}
    //~^ ERROR: usage of `Rc<Arc<bool>>`
    //~| NOTE: `Arc<bool>` is already on the heap, `Rc<Arc<bool>>` makes an extra allocati

    pub fn rc_test8() -> Rc<Box<SubT<usize>>> {
        //~^ ERROR: usage of `Rc<Box<SubT<usize>>>`
        //~| NOTE: `Box<SubT<usize>>` is already on the heap, `Rc<Box<SubT<usize>>>` makes an
        unimplemented!();
    }

    pub fn rc_test9<T>(foo: Rc<Arc<T>>) -> Rc<Arc<SubT<T>>> {
        //~^ ERROR: usage of `Rc<Arc<T>>`
        //~| NOTE: `Arc<T>` is already on the heap, `Rc<Arc<T>>` makes an extra allocation
        //~| ERROR: usage of `Rc<Arc<SubT<T>>>`
        //~| NOTE: `Arc<SubT<T>>` is already on the heap, `Rc<Arc<SubT<T>>>` makes an extra al
        unimplemented!();
    }
}

mod outer_arc {
    use crate::{MyStruct, SubT};
    use std::boxed::Box;
    use std::rc::Rc;
    use std::sync::Arc;

    pub fn arc_test5(a: Arc<Box<bool>>) {}
    //~^ ERROR: usage of `Arc<Box<bool>>`
    //~| NOTE: `Box<bool>` is already on the heap, `Arc<Box<bool>>` makes an extra allocat

    pub fn arc_test6(a: Arc<Rc<bool>>) {}
    //~^ ERROR: usage of `Arc<Rc<bool>>`
    //~| NOTE: `Rc<bool>` is already on the heap, `Arc<Rc<bool>>` makes an extra allocatio

    pub fn arc_test8() -> Arc<Box<SubT<usize>>> {
        //~^ ERROR: usage of `Arc<Box<SubT<usize>>>`
        //~| NOTE: `Box<SubT<usize>>` is already on the heap, `Arc<Box<SubT<usize>>>` makes an
        unimplemented!();
    }

    pub fn arc_test9<T>(foo: Arc<Rc<T>>) -> Arc<Rc<SubT<T>>> {
        //~^ ERROR: usage of `Arc<Rc<T>>`
        //~| NOTE: `Rc<T>` is already on the heap, `Arc<Rc<T>>` makes an extra allocation
        //~| ERROR: usage of `Arc<Rc<SubT<T>>>`
        //~| NOTE: `Rc<SubT<T>>` is already on the heap, `Arc<Rc<SubT<T>>>` makes an extra all
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
    //~^ ERROR: usage of `Rc<Box<Box<dyn T>>>`
    //~| NOTE: `Box<Box<dyn T>>` is already on the heap, `Rc<Box<Box<dyn T>>>` makes an ex
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
    //~^ ERROR: usage of `Rc<Box<Box<str>>>`
    //~| NOTE: `Box<Box<str>>` is already on the heap, `Rc<Box<Box<str>>>` makes an extra
    pub fn test_rc_box_slice(_: Rc<Box<Box<[usize]>>>) {}
    //~^ ERROR: usage of `Rc<Box<Box<[usize]>>>`
    //~| NOTE: `Box<Box<[usize]>>` is already on the heap, `Rc<Box<Box<[usize]>>>` makes a
    pub fn test_rc_box_path(_: Rc<Box<Box<Path>>>) {}
    //~^ ERROR: usage of `Rc<Box<Box<Path>>>`
    //~| NOTE: `Box<Box<Path>>` is already on the heap, `Rc<Box<Box<Path>>>` makes an extr
    pub fn test_rc_box_custom(_: Rc<Box<Box<DynSized>>>) {}
    //~^ ERROR: usage of `Rc<Box<Box<DynSized>>>`
    //~| NOTE: `Box<Box<DynSized>>` is already on the heap, `Rc<Box<Box<DynSized>>>` makes
}

// https://github.com/rust-lang/rust-clippy/issues/11417
fn type_in_closure() {
    let _ = |_: &mut Box<Box<dyn ToString>>| {};
}

fn main() {}
