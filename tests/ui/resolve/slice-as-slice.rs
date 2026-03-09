//@ check-pass
// This is a regression test for github.com/rust-lang/rust/issues/152961
// This broke when a method `as_slice` was added on slices
// This pattern is used in the `rgb` crate

struct Meow;

trait ComponentSlice<T> {
    fn as_slice(&self) -> &[T];
}

impl ComponentSlice<u8> for [Meow] {
    fn as_slice(&self) -> &[u8] {
        todo!()
    }
}

fn a(data: &[Meow]) {
    b(data.as_slice());
    //~^ WARN a method with this name may be added to the standard library in the future
    //~| WARN once this associated item is added to the standard library, the ambiguity may cause an error or change in behavior!
}

fn b(_b: &[u8]) { }

fn main() {}
