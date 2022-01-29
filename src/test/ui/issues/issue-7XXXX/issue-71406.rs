use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel::new(1);
    //~^ ERROR expected type, found function `channel` in `mpsc`
}
