//@ check-pass
//@ edition:2018
//@ aux-crate:recursive_delegation_aux=recursive-delegation-aux.rs

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod first_mod {
    pub mod to_reuse {
        pub fn foo(x: usize) -> usize {
            x + 1
        }
    }

    mod single_reuse {
        reuse crate::first_mod::to_reuse::foo;
        reuse foo as bar;
        reuse foo as bar1;
        reuse bar as goo;
        reuse goo as koo;
        reuse koo as too;
    }

    mod glob_reuse {
        reuse super::to_reuse::{foo as bar, foo as bar1} { self }
        reuse super::glob_reuse::{bar as goo, goo as koo, koo as too} { self }
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
}

fn main() {}
