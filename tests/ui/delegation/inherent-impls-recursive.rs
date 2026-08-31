#![feature(fn_delegation)]

mod test_1 {
    struct S1;
    impl S1 {
        fn foo() {}
    }

    struct S2;
    impl S2 {
        reuse S1::foo;
        //~^ ERROR: cannot find function `foo` in `S1`
    }

    struct S3;
    impl S3 {
        reuse S2::foo;
        //~^ ERROR: cannot find function `foo` in `S2`
    }
}

mod test_2 {
    trait Trait1 {
        fn foo(&self) {}
    }

    impl Trait1 for () {}

    struct S1<T>(T);
    impl<T: Trait1> S1<T> {
        reuse Trait1::foo { self.0 }
    }

    struct S2(S1<()>);
    impl S2 {
        reuse S1::<()>::foo { self.0 }
        //~^ ERROR: cannot find function `foo` in `S1`
    }

    reuse S2::foo;
    //~^ ERROR: cannot find function `foo` in `S2`

    struct S3;
    impl S3 {
        reuse foo;
    }

    impl Trait1 for S3 {
        reuse S2::foo { &S2(S1(())) }
        //~^ ERROR: method `foo` has a `&self` declaration in the trait, but not in the impl
        //~| ERROR: cannot find function `foo` in `S2`
    }

    trait Trait2 {
        reuse <S3 as Trait1>::foo { S3 }
    }

    reuse Trait2::foo as trait_foo;

    struct S4;
    impl S4 {
        reuse trait_foo;
    }

    reuse S4::trait_foo as trait_foo_reused;
    //~^ ERROR: cannot find function `trait_foo` in `S4`
}

fn main() {}
