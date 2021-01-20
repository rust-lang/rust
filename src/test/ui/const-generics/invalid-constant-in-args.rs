use std::cell::Cell;

fn main() {
    let _: Cell<&str, "a"> = Cell::new("");
    //~^ ERROR wrong number of generic arguments
}
