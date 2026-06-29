#![feature(decl_macro)]

mod mod1 {
    mod mod2 {
        pub fn foo() {}
    }

    pub(crate) macro m1() {
        mod2::foo()
    }
}

pub macro m() {
    mod1::m1!()
}
