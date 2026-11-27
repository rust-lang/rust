//@ build-pass
//@ compile-flags: -Zinline-mir

trait Storage {
    type Buffer: ?Sized;
}

struct Array<const N: usize>;
impl<const N: usize> Storage for Array<N> {
    type Buffer = [(); N];
}

struct Slice;
impl Storage for Slice {
    type Buffer = [()];
}

struct Wrap<S: Storage> {
    _b: S::Buffer,
}

fn coerce<const N: usize>(this: &Wrap<Array<N>>) -> &Wrap<Slice>
where
    Array<N>: Storage,
{
    coerce_again(this)
}

fn coerce_again<const N: usize>(this: &Wrap<Array<N>>) -> &Wrap<Slice> {
    this
}

fn main() {
    let inner: Wrap<Array<1>> = Wrap { _b: [(); 1] };
    let _: &Wrap<Slice> = coerce(&inner);
}
