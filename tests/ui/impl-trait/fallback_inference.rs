use std::marker::PhantomData;

fn weird() -> PhantomData<impl Sized> {
    PhantomData //~ ERROR type annotations needed
}

fn main() {}
