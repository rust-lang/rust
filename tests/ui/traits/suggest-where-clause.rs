use std::mem;

struct Misc<T:?Sized>(T);

fn check<T: Iterator, U: ?Sized>() {
    // suggest a where-clause, if needed
    mem::size_of::<U>();
    //~^ ERROR the size for values of type

    mem::size_of::<Misc<U>>();
    //~^ ERROR the size for values of type

    // ... even if T occurs as a type parameter

    <u64 as From<T>>::from;
    //~^ ERROR trait `From<T>` is not implemented for `u64`

    <u64 as From<<T as Iterator>::Item>>::from;
    //~^ ERROR trait `From<<T as Iterator>::Item>` is not implemented for `u64`

    // ... but not if there are inference variables

    <Misc<_> as From<T>>::from;
    //~^ ERROR trait `From<T>` is not implemented for `Misc<_>`

    // ... and also not if the error is not related to the type

    mem::size_of::<[T]>();
    //~^ ERROR the size for values of type

    mem::size_of::<[&U]>();
    //~^ ERROR the size for values of type
}

fn main() {
}
