trait Monad {
    type Unwrapped;
    type Wrapped<B>;

    fn bind<B, F>(self, f: F) -> Self::Wrapped<B> {
        //~^ ERROR: the size for values of type `Self` cannot be known
        todo!()
    }
}

fn join<MOuter, MInner, A>(outer: MOuter) -> MOuter::Wrapped<A>
where
    MOuter: Monad<Unwrapped = MInner>,
    MInner: Monad<Unwrapped = A, Wrapped = MOuter::Wrapped<A>>,
    //~^ ERROR: missing generics for associated type `Monad::Wrapped`
{
    outer.bind(|inner| inner)
}

fn main() {
    assert_eq!(join(Some(Some(true))), Some(true));
    //~^ ERROR: `Option<Option<bool>>: Monad` is not satisfied
}
