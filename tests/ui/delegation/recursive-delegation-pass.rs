//@ run-pass
//@ edition:2018
//@ aux-crate:recursive_delegation_aux=recursive-delegation-aux.rs

#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod first_mod {
    pub mod to_reuse {
        pub fn foo(x: usize) -> usize {
            x + 1
        }
    }

    pub mod single_reuse {
        reuse crate::first_mod::to_reuse::foo { self + 1 }
        reuse foo as bar { self + 1 }
        reuse foo as bar1 { self + 1 }
        reuse bar as goo { self + 1 }
        reuse goo as koo { self + 1 }
        pub reuse koo as too { self + 1 }

        pub fn check() {
            assert_eq!(foo(1), 3);
            assert_eq!(bar(1), 4);
            assert_eq!(bar1(1), 4);
            assert_eq!(goo(1), 5);
            assert_eq!(koo(1), 6);
            assert_eq!(too(1), 7);
        }
    }

    mod glob_reuse {
        reuse super::to_reuse::{foo as bar, foo as bar1} { self }
        reuse super::glob_reuse::{bar as goo, goo as koo, koo as too} { self }

        fn check() {
            bar(1);
            bar1(1);
            goo(1);
            koo(1);
            too(1);
        }
    }
}

mod second_mod {
    trait T {
        fn foo(&self);
        reuse T::foo as bar;
        reuse T::bar as goo;
        reuse T::goo as poo;
    }

    trait TGlob {
        fn xd(&self) -> &Self;
        fn foo1(&self);
        fn foo2(&self);
        fn foo3(&self);
        fn foo4(&self);

        reuse TGlob::{foo1 as bar1, foo3 as bar3, bar1 as bar11, bar11 as bar111} { self.xd() }
    }

    fn check() {
        struct X;
        impl T for X {
            fn foo(&self) {}
            fn bar(&self) {}
            fn goo(&self) {}
            fn poo(&self) {}
        }

        impl TGlob for X {
            fn xd(&self) -> &Self { &self }
            fn foo1(&self) {}
            fn foo2(&self) {}
            fn foo3(&self) {}
            fn foo4(&self) {}

            fn bar1(&self) {}
            fn bar3(&self) {}
            fn bar11(&self) {}
            fn bar111(&self) {}
        }
    }
}

mod third_mod {
    reuse crate::first_mod::to_reuse::foo {
        reuse foo as bar {
            reuse bar as goo {
                bar(123)
            }

            goo(123)
        }

        bar(123)
    }
}

mod fourth_mod {
    reuse recursive_delegation_aux::goo as bar;
    reuse bar as foo;

    fn check() {
        bar();
        foo();
    }
}

mod fifth_mod {
    mod m {
        fn foo() { }
        pub reuse foo as bar;
    }

    trait T {
        reuse m::bar as foo;
    }

    struct X;
    impl T for X {}

    trait T1 {
        reuse <X as T>::foo as baz;
    }

    impl T1 for X {}

    struct Y;
    impl T1 for Y {
        reuse <X as T>::foo as baz;
    }

    fn check() {
        m::bar();
        <X as T>::foo();
        <X as T1>::baz();
        <Y as T1>::baz();
    }
}

fn main() {
    first_mod::single_reuse::check();
}
