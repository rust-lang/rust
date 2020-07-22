// build-pass
// compile-flags:-Zpolymorphize=on

pub trait ParallelIterator: Sized {
    fn drive<C: Consumer<()>>(_: C) {
        C::into_folder();
    }
}

pub trait Consumer<T>: Sized {
    type Result;
    fn into_folder() -> Self::Result;
}

impl ParallelIterator for () {}

impl<F: Fn(), T> Consumer<T> for F {
    type Result = ();
    fn into_folder() -> Self::Result {
        unimplemented!()
    }
}

fn main() {
    <()>::drive(|| ());
}
