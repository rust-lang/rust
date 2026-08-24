#![feature(fn_delegation)]

mod test_1 {
    struct X<T>(T);

    impl X<()> {
        fn foo() {}
        fn foo_self(self) {}
    }

    impl X<usize> {
        fn foo() {}
        fn foo_self(self) {}
    }

    reuse X::foo;
    //~^ ERROR: cannot find function `foo` in `X`

    reuse X::foo_self;
    //~^ ERROR: cannot find function `foo_self` in `X`

    reuse X::<()>::foo as foo1;
    //~^ ERROR: cannot find function `foo` in `X`

    reuse X::<usize>::foo_self as foo_self1;
    //~^ ERROR: cannot find function `foo_self` in `X`
}

mod test_2 {
    struct X<T, U>(T, U);
    trait Marker1 {}
    trait Marker2 {}

    impl<T: Marker1> X<T, usize> {
        fn foo() {}
        fn foo_self(self) {}
    }

    impl<T: Marker2> X<T, ()> {
        fn foo() {}
        fn foo_self(self) {}
    }

    struct M1;
    impl Marker1 for M1 {}
    struct M2;
    impl Marker2 for M2 {}

    reuse X::foo;
    //~^ ERROR: cannot find function `foo` in `X`

    reuse X::foo_self;
    //~^ ERROR: cannot find function `foo_self` in `X`

    reuse X::<M1, usize>::foo as foo1;
    //~^ ERROR: cannot find function `foo` in `X`

    reuse X::<M2, String>::foo_self as foo_self1;
    //~^ ERROR: cannot find function `foo_self` in `X`
}

fn main() {}
