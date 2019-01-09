// Test that manual impls of the `Fn` traits are not possible without
// a feature gate. In fact, the specialized check for these cases
// never triggers (yet), because they encounter other problems around
// angle bracket vs parentheses notation.

#![feature(fn_traits)]

struct Foo;
impl Fn<()> for Foo {
//~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    extern "rust-call" fn call(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Foo1;
impl FnOnce() for Foo1 {
//~^ ERROR associated type bindings are not allowed here
    extern "rust-call" fn call_once(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Bar;
impl FnMut<()> for Bar {
//~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    extern "rust-call" fn call_mut(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Baz;
impl FnOnce<()> for Baz {
//~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    extern "rust-call" fn call_once(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}

fn main() {}
