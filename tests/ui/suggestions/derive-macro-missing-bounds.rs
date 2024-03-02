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
    struct Outer<T>(Inner<T>); //~ ERROR trait `c::Trait` is not implemented for `T`
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
    struct Outer<T>(Inner<T>); //~ ERROR trait `d::Trait` is not implemented for `T`
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
    struct Outer<T>(Inner<T>); //~ ERROR trait `e::Trait` is not implemented for `T`
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
    struct Outer<T>(Inner<T>); //~ ERROR trait `f::Trait` is not implemented for `T`
}

fn main() {}
