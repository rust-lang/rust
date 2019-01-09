// compile-flags: -Z no-landing-pads -C codegen-units=1
// error-pattern:converging_fn called
// ignore-cloudabi no std::process

use std::io::{self, Write};

struct Droppable;
impl Drop for Droppable {
    fn drop(&mut self) {
        ::std::process::exit(1)
    }
}

fn converging_fn() {
    panic!("converging_fn called")
}

fn mir(d: Droppable) {
    let x = Droppable;
    converging_fn();
    drop(x);
    drop(d);
}

fn main() {
    mir(Droppable);
}
