#![feature(optin_builtin_traits)]

struct Foo;

unsafe impl !Send for Foo { } //~ ERROR E0198

fn main() {
}
