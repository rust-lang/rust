// build-pass (FIXME(62277): could be check-pass?)
#![allow(warnings)]

mod foo {
    pub use foo::bar::S;
    mod bar {
        #[derive(Default)]
        pub struct S {
            pub(in foo) x: i32,
        }
        impl S {
            pub(in foo) fn f(&self) -> i32 { 0 }
        }

        pub struct S2 {
            pub(crate) x: bool,
        }
        impl S2 {
            pub(crate) fn f(&self) -> bool { false }
        }

        impl ::std::ops::Deref for S {
            type Target = S2;
            fn deref(&self) -> &S2 { unimplemented!() }
        }
    }
}


fn main() {
    let s = foo::S::default();
    let _: bool = s.x;
    let _: bool = s.f();
}
