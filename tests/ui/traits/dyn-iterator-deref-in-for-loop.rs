//! Tests that dereferencing a Box<dyn Iterator> in a for loop correctly yields an error,
//! as the unsized trait object does not implement IntoIterator.
//! regression test for <https://github.com/rust-lang/rust/issues/20605>
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn changer<'a>(mut things: Box<dyn Iterator<Item = &'a mut u8>>) {
    for item in *things {
        //~^ ERROR `dyn Iterator<Item = &'a mut u8>` is not an iterator
        *item = 0
    }
}

fn main() {}
