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
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::foo_self;
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::<()>::foo as foo1;
    //~^ ERROR: ambiguous delegation to inherent impl function

    reuse X::<usize>::foo_self as foo_self1;
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: this function takes 1 argument but 0 arguments were supplied
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
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::foo_self;
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::<M1, usize>::foo as foo1;
    //~^ ERROR: ambiguous delegation to inherent impl function

    reuse X::<M2, String>::foo_self as foo_self1;
    //~^ ERROR: ambiguous delegation to inherent impl function
    //~| ERROR: no associated function or constant named `foo_self` found for struct `test_2::X<M2, String>` in the current scope
}

fn main() {}
