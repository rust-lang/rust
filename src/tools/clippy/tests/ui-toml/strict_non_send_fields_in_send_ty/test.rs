#![warn(clippy::non_send_fields_in_send_ty)]
#![feature(extern_types)]

use std::rc::Rc;

// Basic tests should not be affected
pub struct NoGeneric {
    rc_is_not_send: Rc<String>,
}

unsafe impl Send for NoGeneric {}

pub struct MultiField<T> {
    field1: T,
    field2: T,
    field3: T,
}

unsafe impl<T> Send for MultiField<T> {}

pub enum MyOption<T> {
    MySome(T),
    MyNone,
}

unsafe impl<T> Send for MyOption<T> {}

// All fields are disallowed when raw pointer heuristic is off
extern "C" {
    type NonSend;
}

pub struct HeuristicTest {
    field1: Vec<*const NonSend>,
    field2: [*const NonSend; 3],
    field3: (*const NonSend, *const NonSend, *const NonSend),
    field4: (*const NonSend, Rc<u8>),
    field5: Vec<Vec<*const NonSend>>,
}

unsafe impl Send for HeuristicTest {}

fn main() {}
