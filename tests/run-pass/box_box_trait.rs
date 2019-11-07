struct DroppableStruct;

static mut DROPPED: bool = false;

impl Drop for DroppableStruct {
    fn drop(&mut self) {
        unsafe { DROPPED = true; }
    }
}

trait MyTrait { fn dummy(&self) { } }
impl MyTrait for Box<DroppableStruct> {}

struct Whatever { w: Box<dyn MyTrait+'static> }

impl  Whatever {
    fn new(w: Box<dyn MyTrait+'static>) -> Whatever {
        Whatever { w: w }
    }
}

fn main() {
    {
        let f = Box::new(DroppableStruct);
        let a = Whatever::new(Box::new(f) as Box<dyn MyTrait>);
        a.w.dummy();
    }
    assert!(unsafe { DROPPED });
}
