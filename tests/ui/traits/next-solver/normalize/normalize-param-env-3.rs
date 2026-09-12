//@ compile-flags: -Znext-solver

// Regression test for #100177.
// This was fixed by lazy norm of param env with the next solver.
// But it regressed again as we switched back to be consistent with
// the old solver. See #158643.

trait GenericTrait<T> {}

trait Channel<I>: GenericTrait<Self::T> {
    type T;
}

trait Sender {
    type Msg;

    fn send<C>()
    where
        C: Channel<Self::Msg>;
}

impl<T> Sender for T {
    type Msg = ();

    fn send<C>()
    //~^ ERROR: the trait bound `C: Channel<()>` is not satisfied
    //~| ERROR: the trait bound `C: Channel<()>` is not satisfied
    where
        C: Channel<Self::Msg>,
    {
    }
}

// This works
fn foo<I, C>(ch: C) where C: Channel<I> {}

fn main() {}
