// https://github.com/rust-lang/rust/issues/71406
use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel::new(1);
    //~^ ERROR: cannot find `channel`
    //~| NOTE: expected type, found function `channel` in `mpsc`
}
