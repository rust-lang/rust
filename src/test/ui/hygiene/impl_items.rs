// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]

mod foo {
    struct S;
    impl S {
        fn f(&self) {}
    }

    pub macro m() {
        let _: () = S.f(); //~ ERROR type `for<'r> fn(&'r foo::S) {<foo::S>::f}` is private
    }
}

struct S;

macro m($f:ident) {
    impl S {
        fn f(&self) -> u32 { 0 }
        fn $f(&self) -> i32 { 0 }
    }
    fn f() {
        let _: u32 = S.f();
        let _: i32 = S.$f();
    }
}

m!(f);

fn main() {
    let _: i32 = S.f();
    foo::m!();
}
