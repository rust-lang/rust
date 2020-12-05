// run-pass
#![allow(dead_code)]
#![feature(box_syntax)]

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
        let f: Box<_> = box DroppableStruct;
        let _a = Whatever::new(box f as Box<dyn MyTrait>);
    }
    assert!(unsafe { DROPPED });
    unsafe { DROPPED = false; }
    {
        let f: Box<_> = box DroppableEnum::DroppableVariant1;
        let _a = Whatever::new(box f as Box<dyn MyTrait>);
    }
    assert!(unsafe { DROPPED });
}
