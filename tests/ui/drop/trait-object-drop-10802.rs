//! Regression test for https://github.com/rust-lang/rust/issues/10802

//@ run-pass
#![allow(dead_code)]

struct DroppableStruct;
enum DroppableEnum {
    DroppableVariant1, DroppableVariant2
}

static mut DROPPED: bool = false;

impl Drop for DroppableStruct {
    fn drop(&mut self) {
        unsafe { DROPPED = true; }
    }
}
impl Drop for DroppableEnum {
    fn drop(&mut self) {
        unsafe { DROPPED = true; }
    }
}

trait MyTrait { fn dummy(&self) { } }
impl MyTrait for Box<DroppableStruct> {}
impl MyTrait for Box<DroppableEnum> {}

struct Whatever { w: Box<dyn MyTrait+'static> }
impl  Whatever {
    fn new(w: Box<dyn MyTrait+'static>) -> Whatever {
        Whatever { w: w }
    }
}

fn main() {
    {
        let f: Box<_> = Box::new(DroppableStruct);
        let _a = Whatever::new(Box::new(f) as Box<dyn MyTrait>);
    }
    assert!(unsafe { DROPPED });
    unsafe { DROPPED = false; }
    {
        let f: Box<_> = Box::new(DroppableEnum::DroppableVariant1);
        let _a = Whatever::new(Box::new(f) as Box<dyn MyTrait>);
    }
    assert!(unsafe { DROPPED });
}
