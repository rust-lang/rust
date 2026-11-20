//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/157800>.
// Used to ICE

trait Dbg {}

struct Foo<I, E> {
    input: I,
    errors: E,
}

trait Bar: Offset<<Self as Bar>::Checkpoint> {
    type Checkpoint;
}

impl<I: Bar, E: Dbg> Bar for Foo<I, E> {
    type Checkpoint = I::Checkpoint;
}

trait Offset<Start = Self> {}

impl<I: Bar, E: Dbg> Offset<<Foo<I, E> as Bar>::Checkpoint> for Foo<I, E> {}

impl<I: Bar, E: Dbg> Foo<I, E> {
    fn record_err(self, _: <Self as Bar>::Checkpoint) {}
}

fn main() {}
