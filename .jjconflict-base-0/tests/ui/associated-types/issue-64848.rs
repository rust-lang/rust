//@ build-pass

trait AssociatedConstant {
    const DATA: ();
}

impl<F, T> AssociatedConstant for F
where
    F: FnOnce() -> T,
    T: AssociatedConstant,
{
    const DATA: () = T::DATA;
}

impl AssociatedConstant for () {
    const DATA: () = ();
}

fn foo() -> impl AssociatedConstant {
    ()
}

fn get_data<T: AssociatedConstant>(_: T) -> &'static () {
    &T::DATA
}

fn main() {
    get_data(foo);
}
