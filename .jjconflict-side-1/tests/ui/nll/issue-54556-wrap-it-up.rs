// This is testing how the diagnostic from issue #54556 behaves when
// the destructor code is attached to a place held in a field of the
// temporary being dropped.
//
// Eventually it would be nice if the diagnostic would actually report
// that specific place and its type that implements the `Drop` trait.
// But for the short term, it is acceptable to just print out the
// whole type of the temporary.

#![allow(warnings)]

struct Wrap<'p> { p: &'p mut i32 }

impl<'p> Drop for Wrap<'p> {
    fn drop(&mut self) {
        *self.p += 1;
    }
}

struct Foo<'p> { a: String, b: Wrap<'p> }

fn main() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    let s = String::from("str");
    let foo = Foo { a: s, b: wrap };
    x = 1; //~ ERROR cannot assign to `x` because it is borrowed [E0506]
}
