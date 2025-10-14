//@ check-fail
//@ edition:2018
#![feature(transparent_modules)]

mod limit_macro_lexical_scope {
    #[macro_export] // add path-based scope name for macro to crate root
    macro_rules! s {
        () => {};
    }
}

struct S;

mod non_transparent {
    fn foo() {
        #[transparent]
        mod transparent {
            // early resolution
            s!(); //~ ERROR cannot find macro `s` in this scope
            // late resolution
            struct Y(S); //~ ERROR cannot find type `S` in this scope
        }
    }
}

fn main() {}
