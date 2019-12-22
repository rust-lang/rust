// run-pass
// Test the uninit() construct returning various empty types.

// pretty-expanded FIXME #23616

use std::mem::MaybeUninit;

#[derive(Clone)]
struct Foo;

#[allow(deprecated)]
pub fn main() {
    unsafe {
        // `Foo` and `[Foo; 2]` are both zero sized so this is safe.
        let _x: Foo = MaybeUninit::uninit().assume_init();
        let _x: [Foo; 2] = MaybeUninit::uninit().assume_init();
        let _x: Foo = std::mem::uninitialized();
        let _x: [Foo; 2] = std::mem::uninitialized();
    }
}
