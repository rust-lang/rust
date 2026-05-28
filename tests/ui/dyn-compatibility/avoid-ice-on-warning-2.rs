//@ revisions: old new
//@[old] edition:2015
//@[new] edition:2021
fn id<F>(f: Copy) -> usize {
//[new]~^ ERROR expected a type, found a trait
//[old]~^^ ERROR the trait `Copy` is not dyn compatible
//[old]~| ERROR the size for values of type `(dyn Copy + 'static)`
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    f()
    //[old]~^ ERROR: expected function, found `(dyn Copy + 'static)`
}
fn main() {}
