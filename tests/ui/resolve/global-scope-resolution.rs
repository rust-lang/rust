//! Test global scope resolution with :: operator

//@ run-pass

pub fn f() -> isize {
    return 1;
}

pub mod foo {
    pub fn f() -> isize {
        return 2;
    }
    pub fn g() {
        assert_eq!(f(), 2);
        assert_eq!(::f(), 1);
    }
}

pub fn main() {
    return foo::g();
}
