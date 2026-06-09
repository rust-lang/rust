//@ edition: 2015..2021
#![warn(rust_2021_incompatible_closure_captures)]
trait Owner { type Ty<T: FnMut()>; }
impl Owner for () { type Ty<T: FnMut()> = T; }
pub struct Warns<T> {
    _significant_drop: <() as Owner>::Ty<T>,
    //~^ ERROR expected a `FnMut()` closure, found `T`
    field: String,
}
pub fn test<T>(w: Warns<T>) {
    //~^ ERROR expected a `FnMut()` closure, found `T`
    _ = || w.field
    //~^ ERROR expected a `FnMut()` closure, found `T`
    //~| ERROR expected a `FnMut()` closure, found `T`
    //~| WARN: changes to closure capture in Rust 2021 will affect drop order
}
fn main() {}
