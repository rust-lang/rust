//! Checks that an incorrect number of arguments to splat doesn't panic.

#![feature(splat)]
struct Foo {}
trait BarTrait {
    fn trait_assoc<W>(w: W, #[splat] _s: (u32, u8));
}
impl BarTrait for Foo {
    fn trait_assoc<W>(_w: W, #[splat] _s: (u32, u8)) {}
}
fn main() {
    Foo::trait_assoc()
    //~^ ERROR: this splatted function takes 3 arguments, but 0 were provided
}
