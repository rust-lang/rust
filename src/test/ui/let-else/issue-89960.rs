#![feature(let_else)]

fn main() {
    // FIXME: more precise diagnostics
    let Some(ref mut meow) = Some(()) else { return };
    //~^ ERROR: cannot borrow value as mutable, as `val` is not declared as mutable
}
