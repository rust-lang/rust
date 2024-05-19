//@ known-bug: rust-lang/rust#124891

type Tait = impl FnOnce() -> ();

fn reify_as_tait() -> Thunk<Tait> {
    Thunk::new(|cont| cont)
}

struct Thunk<F>(F);

impl<F> Thunk<F> {
    fn new(f: F)
    where
        F: ContFn,
    {
        todo!();
    }
}

trait ContFn {}

impl<F: FnOnce(Tait) -> ()> ContFn for F {}
