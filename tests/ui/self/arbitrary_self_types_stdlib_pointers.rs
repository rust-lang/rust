//@ run-pass
#![feature(arbitrary_self_types)]
#![feature(rustc_attrs)]

use std::{
    rc::Rc,
    sync::Arc,
    pin::Pin,
};

trait Trait {
    fn by_rc(self: Rc<Self>) -> i64;
    fn by_arc(self: Arc<Self>) -> i64;
    fn by_pin_mut(self: Pin<&mut Self>) -> i64;
    fn by_pin_box(self: Pin<Box<Self>>) -> i64;
    fn by_pin_pin_pin_ref(self: Pin<Pin<Pin<&Self>>>) -> i64;
}

impl Trait for i64 {
    fn by_rc(self: Rc<Self>) -> i64 {
        *self
    }
    fn by_arc(self: Arc<Self>) -> i64 {
        *self
    }
    fn by_pin_mut(self: Pin<&mut Self>) -> i64 {
        *self
    }
    fn by_pin_box(self: Pin<Box<Self>>) -> i64 {
        *self
    }
    fn by_pin_pin_pin_ref(self: Pin<Pin<Pin<&Self>>>) -> i64 {
        *self
    }
}

fn main() {
    let rc = Rc::new(1i64) as Rc<dyn Trait>;
    assert_eq!(1, rc.by_rc());

    let arc = Arc::new(2i64) as Arc<dyn Trait>;
    assert_eq!(2, arc.by_arc());

    let mut value = 3i64;
    let pin_mut = Pin::new(&mut value) as Pin<&mut dyn Trait>;
    assert_eq!(3, pin_mut.by_pin_mut());

    let pin_box = Into::<Pin<Box<i64>>>::into(Box::new(4i64)) as Pin<Box<dyn Trait>>;
    assert_eq!(4, pin_box.by_pin_box());

    let value = 5i64;
    let pin_pin_pin_ref = Pin::new(Pin::new(Pin::new(&value))) as Pin<Pin<Pin<&dyn Trait>>>;
    assert_eq!(5, pin_pin_pin_ref.by_pin_pin_pin_ref());
}
