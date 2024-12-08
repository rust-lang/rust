//@ known-bug: #120254

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
}
