#![crate_type = "lib"]

pub trait X {
    fn x() {
        fn f() { }
        f();
    }
    fn dummy(&self) { }
}
