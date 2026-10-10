//@ needs-rustc-debug-assertions

#![feature(return_type_notation)]

// This unrelated item forces error recovery. Keep more lifetime parameters
// here than the anonymous lifetimes introduced by the function so that using
// this item's bound variables cannot accidentally hide an out-of-bounds access.
struct Wrapper<'a, 'b, 'c, 'd>();
//~^ ERROR parameter `'a` is never used
//~| ERROR parameter `'b` is never used
//~| ERROR parameter `'c` is never used
//~| ERROR parameter `'d` is never used

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send>>;
}

fn main() {}
