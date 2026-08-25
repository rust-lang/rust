mod a {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);

    impl Debug for Inner<()> {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>); //~ ERROR `a::Inner<T>` doesn't implement `Debug`
}

mod b {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);

    impl<T: Debug> Debug for Inner<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>);
}

mod c {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);
    trait Trait {}

    impl<T: Debug + Trait> Debug for Inner<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>); //~ ERROR the trait bound `T: c::Trait` is not satisfied
}

mod d {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);
    trait Trait {}

    impl<T> Debug for Inner<T> where T: Debug, T: Trait {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>); //~ ERROR the trait bound `T: d::Trait` is not satisfied
}

mod e {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);
    trait Trait {}

    impl<T> Debug for Inner<T> where T: Debug + Trait {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>); //~ ERROR the trait bound `T: e::Trait` is not satisfied
}

mod f {
    use std::fmt::{Debug, Formatter, Result};
    struct Inner<T>(T);
    trait Trait {}

    impl<T: Debug> Debug for Inner<T> where T: Trait {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Outer<T>(Inner<T>); //~ ERROR the trait bound `T: f::Trait` is not satisfied
}

fn main() {}
