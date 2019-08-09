// check-pass
// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro, associated_type_defaults)]

trait Base {
    type AssocTy;
    fn f();
}
trait Derived: Base {
    fn g();
}

macro mac() {
    type A = Base<AssocTy = u8>;
    type B = Derived<AssocTy = u8>;

    impl Base for u8 {
        type AssocTy = u8;
        fn f() {
            let _: Self::AssocTy;
        }
    }
    impl Derived for u8 {
        fn g() {
            let _: Self::AssocTy;
        }
    }

    fn h<T: Base, U: Derived>() {
        let _: T::AssocTy;
        let _: U::AssocTy;
    }
}

mac!();

fn main() {}
