#![feature(decl_macro)]

mod foo {
    pub macro m() { Vec::new(); ().clone() }
    fn f() { ::bar::m!(); }
}

#[no_implicit_prelude]
mod bar {
    pub macro m() {
        Vec::new(); //~ ERROR failed to resolve
        ().clone() //~ ERROR no method named `clone` found
    }
    fn f() {
        ::foo::m!();
        assert!(true); // OK
        assert_eq!(0, 0); // OK
    }
}

fn main() {}
