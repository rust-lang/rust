#![feature(box_syntax)]

struct DroppableStruct;

static mut DROPPED: bool = false;

impl Drop for DroppableStruct {
    fn drop(&mut self) {
        unsafe { DROPPED = true; }
    }
}

trait MyTrait { fn dummy(&self) { } }
impl MyTrait for Box<DroppableStruct> {}

#[allow(dead_code)]
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
}
