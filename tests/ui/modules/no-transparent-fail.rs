//@ check-fail
//@ edition:2018
// gate-test-transparent_modules
mod y {
    macro_rules! s {
        () => {};
    }

    pub(crate) use s;
}

trait IWantAMethod {
    fn method(&self) {}
}

impl IWantAMethod for () {}

fn foo() {
    struct S;
    impl S {
        const NAME: &'static str = "S";
    }
    enum C {
        B,
    }

    use y::s;
    mod x {
        // early resolution
        s!(); //~ ERROR cannot find macro `s` in this scope
        // late resolution
        struct Y(S); //~ ERROR cannot find type `S` in this scope
        impl Y {
            // hir_typeck type dependent name resolution of associated const
            const SNAME: &'static str = S::NAME; //~ ERROR failed to resolve: use of undeclared type `S`
        }
        fn bar() -> C {
            //~^ ERROR cannot find type `C` in this scope
            // method lookup, resolving appropriate trait in scope
            ().method(); //~ ERROR no method named `method` found for unit type `()` in the current scope
            // hir ty lowering type dependent name resolution of associated enum variant
            C::B //~ ERROR failed to resolve: use of undeclared type `C`
        }
    }
}

fn main() {}
