//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #120254

//[next]~^^^^^^^ ERROR: the trait bound `E: Dbg` is not satisfied

// Regression test for #120254. This no longer reproduces with the new solver
// even though the underlying issue has not been fixed.

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

impl<I: Bar, E> Foo<I, E> {
    fn record_err(self, _: <Self as Bar>::Checkpoint) -> () {}
    //[next]~^ ERROR: the trait bound `E: Dbg` is not satisfied
    //[next]~| ERROR: the type `<Foo<I, E> as Bar>::Checkpoint` is not well-formed
    //[next]~| ERROR: the trait bound `E: Dbg` is not satisfied
}

fn main() {}
