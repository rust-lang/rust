//@ check-pass
//@ compile-flags: -Znormalize-docs

trait Woo<T> {
    type Assoc;
}

impl<T> Woo<T> for () {
    type Assoc = ();
}

type Alias<P> = <() as Woo<P>>::Assoc;

pub fn hello<S>() -> Alias<S> {}
