// Test that manual impls of the `Fn` traits are not possible without
// a feature gate. In fact, the specialized check for these cases
// never triggers (yet), because they encounter other problems around
// angle bracket vs parentheses notation.

#![feature(fn_traits)]

struct Foo;
impl Fn<()> for Foo {
    //~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    //~| ERROR manual implementations of `Fn` are experimental
    //~| ERROR expected a `FnMut()` closure, found `Foo`
    extern "rust-call" fn call(self, args: ()) -> () {}
    //~^ ERROR "rust-call" ABI is experimental and subject to change
    //~| ERROR `call` has an incompatible type for trait
}
struct Foo1;
impl FnOnce() for Foo1 {
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR manual implementations of `FnOnce` are experimental
    //~| ERROR not all trait items implemented
    extern "rust-call" fn call_once(self, args: ()) -> () {}
    //~^ ERROR "rust-call" ABI is experimental and subject to change
}
struct Bar;
impl FnMut<()> for Bar {
    //~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    //~| ERROR manual implementations of `FnMut` are experimental
    //~| ERROR expected a `FnOnce()` closure, found `Bar`
    extern "rust-call" fn call_mut(&self, args: ()) -> () {}
    //~^ ERROR "rust-call" ABI is experimental and subject to change
    //~| ERROR incompatible type for trait
}
struct Baz;
impl FnOnce<()> for Baz {
    //~^ ERROR the precise format of `Fn`-family traits' type parameters is subject to change
    //~| ERROR manual implementations of `FnOnce` are experimental
    //~| ERROR not all trait items implemented
    extern "rust-call" fn call_once(&self, args: ()) -> () {}
    //~^ ERROR "rust-call" ABI is experimental and subject to change
}

fn main() {}
