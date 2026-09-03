//@compile-flags: -Z deduplicate-diagnostics=yes

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
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::foo_self;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::<()>::foo as foo1;

    reuse X::<usize>::foo_self as foo_self1;
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
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::foo_self;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    //~| ERROR: multiple applicable items in scope [E0034]

    reuse X::<M1, usize>::foo as foo1;
    reuse X::<M1, usize>::foo_self as foo_self1;
    reuse X::<M2, ()>::foo as foo2;
    reuse X::<M2, ()>::foo_self as foo_self2;

    reuse X::<M2, String>::foo as foo3;
    //~^ ERROR: no associated function or constant named `foo` found for struct `test_2::X<M2, String>` in the current scope
    reuse X::<M2, String>::foo_self as foo_self3;
    //~^ ERROR: no associated function or constant named `foo_self` found for struct `test_2::X<M2, String>` in the current scope
}

fn main() {}
