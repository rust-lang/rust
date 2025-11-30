//@ check-pass
//@ edition:2018
#![feature(transparent_modules)]

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
    #[transparent]
    mod x {
        // early resolution
        s!();
        // late resolution
        struct Y(S);
        impl Y {
            // hir_typeck type dependent name resolution of associated const
            const SNAME: &'static str = S::NAME;
        }
        fn bar() -> C {
            // method lookup, resolving appropriate trait in scope
            ().method();
            // hir ty lowering type dependent name resolution of associated enum variant
            C::B
        }
    }
}

pub fn main() {}
