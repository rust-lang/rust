// Test that manual impls of the `Fn` traits are not possible without
// a feature gate. In fact, the specialized check for these cases
// never triggers (yet), because they encounter other problems around
// angle bracket vs parentheses notation.

#![allow(dead_code)]

struct Foo;
impl Fn<()> for Foo {
    extern "rust-call" fn call(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Foo1;
impl FnOnce() for Foo1 {
    extern "rust-call" fn call_once(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Bar;
impl FnMut<()> for Bar {
    extern "rust-call" fn call_mut(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Baz;
impl FnOnce<()> for Baz {
    extern "rust-call" fn call_once(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}

fn main() {}
