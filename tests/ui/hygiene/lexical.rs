//@ check-pass

#![feature(decl_macro)]

mod bar {
    mod baz {
        pub fn f() {}
    }

    pub macro m($f:ident) {
        baz::f();
        let _: i32 = $f();
        {
            fn $f() -> u32 { 0 }
            let _: u32 = $f();
        }
    }
}

fn main() {
    fn f() -> i32 { 0 }
    bar::m!(f);
}
