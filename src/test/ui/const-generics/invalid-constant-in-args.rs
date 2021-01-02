use std::cell::Cell;

fn main() {
    let _: Cell<&str, "a"> = Cell::new("");
    //~^ ERROR this struct takes 1 generic argument but 2 generic arguments were supplied
}
