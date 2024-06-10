//@revisions: edition2018 edition2021
//@[edition2018] edition:2018
//@[edition2021] edition:2021
fn main() {
    let x: u32 = <Default>::default();
    //[edition2021]~^ ERROR trait objects must include the `dyn` keyword
    //[edition2018]~^^ WARNING trait objects without an explicit `dyn` are deprecated
    //[edition2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2021!
    //[edition2018]~| ERROR trait `Default` cannot be made into an object
    //[edition2018]~| ERROR trait `Default` cannot be made into an object
    //[edition2018]~| ERROR the size for values of type `dyn Default` cannot be known at compilation time
    //[edition2018]~| ERROR mismatched types
    //[edition2018]~| ERROR the size for values of type `dyn Default` cannot be known at compilation time
}
