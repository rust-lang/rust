// Verifies that MIR building for a call expression respects
// privacy when checking if a call return type is uninhabited.
//@ run-rustfix
#![allow(unreachable_code, unused_variables)]

pub mod widget {
    enum Unimplemented {}
    pub struct Widget(Unimplemented);

    impl Widget {
        pub fn new() -> Widget {
            todo!();
        }
    }

    pub fn f() {
        let x: &mut u32;
        Widget::new();
        // Ok. Widget type returned from new is known to be uninhabited
        // and the following code is considered unreachable.
        *x = 1;
    }
}

fn main() {
    let y: &mut u32;
    widget::Widget::new();
    // Error. Widget type is not known to be uninhabited here,
    // so the following code is considered reachable.
    *y = 2; //~ ERROR E0381
}
