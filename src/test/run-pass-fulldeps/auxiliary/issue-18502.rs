#![crate_type="lib"]

struct Foo;
// This is the ICE trigger
struct Formatter;

trait Show {
    fn fmt(&self);
}

impl Show for Foo {
    fn fmt(&self) {}
}

fn bar<T>(f: extern "Rust" fn(&T), t: &T) { }

// ICE requirement: this has to be marked as inline
#[inline]
pub fn baz() {
    bar(Show::fmt, &Foo);
}
