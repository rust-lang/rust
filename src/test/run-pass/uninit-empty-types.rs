// Test the uninit() construct returning various empty types.

// pretty-expanded FIXME #23616

use std::mem;

#[derive(Clone)]
struct Foo;

#[allow(deprecated)]
pub fn main() {
    unsafe {
        let _x: Foo = mem::uninitialized();
        let _x: [Foo; 2] = mem::uninitialized();
    }
}
