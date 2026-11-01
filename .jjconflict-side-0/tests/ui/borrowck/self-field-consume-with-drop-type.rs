//! Regression test for https://github.com/rust-lang/rust/issues/47703
//@ check-pass

struct MyStruct<'a> {
    field: &'a mut (),
    field2: WithDrop
}

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

impl<'a> MyStruct<'a> {
    fn consume(self) -> &'a mut () { self.field }
}

fn main() {}
