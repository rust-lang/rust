trait Monad {
    type Unwrapped;
    type Wrapped<B>;

    fn bind<B, F>(self, f: F) -> Self::Wrapped<B> {
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
}
