//@ check-pass
use std::cell::Cell;

pub enum JsValue {
    Undefined,
    Object(Cell<bool>),
}

impl ::std::ops::Drop for JsValue {
    fn drop(&mut self) {}
}

const UNDEFINED: &JsValue = &JsValue::Undefined;
    //~^ WARN encountered mutable pointer in final value of constant
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main() {
}
