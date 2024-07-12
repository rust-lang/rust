use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel::new(1);
    //~^ ERROR cannot find item `channel`
    //~| NOTE expected type, found function `channel` in `mpsc`
}
