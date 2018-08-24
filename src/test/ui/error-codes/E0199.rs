#![feature(optin_builtin_traits)]

struct Foo;

trait Bar { }
unsafe impl Bar for Foo { } //~ ERROR implementing the trait `Bar` is not unsafe [E0199]

fn main() {
}
