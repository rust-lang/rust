// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

mod a {
    pub enum Enum<T> {
        A(T),
    }

    pub trait X {
        fn dummy(&self) { }
    }
    impl X for isize {}

    pub struct Z<'a>(Enum<&'a (dyn X + 'a)>);
    fn foo() { let x: isize = 42; let z = Z(Enum::A(&x as &dyn X)); let _ = z; }
}

mod b {
    trait X {
        fn dummy(&self) { }
    }
    impl X for isize {}
    struct Y<'a>{
        x:Option<&'a (dyn X + 'a)>,
    }

    fn bar() {
        let x: isize = 42;
        let _y = Y { x: Some(&x as &dyn X) };
    }
}

mod c {
    pub trait X { fn f(&self); }
    impl X for isize { fn f(&self) {} }
    pub struct Z<'a>(Option<&'a (dyn X + 'a)>);
    fn main() { let x: isize = 42; let z = Z(Some(&x as &dyn X)); let _ = z; }
}

pub fn main() {}
