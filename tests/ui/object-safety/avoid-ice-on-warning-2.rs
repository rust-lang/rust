fn id<F>(f: Copy) -> usize {
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//~| ERROR the trait `Copy` cannot be made into an object
    f()
}
fn main() {}
