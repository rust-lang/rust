// check-pass
// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro, associated_type_defaults)]

trait Base {
    type AssocTy;
    fn f(&self);
}
trait Derived: Base {
    fn g(&self);
}

macro mac() {
    type A = dyn Base<AssocTy = u8>;
    type B = dyn Derived<AssocTy = u8>;

    impl Base for u8 {
        type AssocTy = u8;
        fn f(&self) {
            let _: Self::AssocTy;
        }
    }
    impl Derived for u8 {
        fn g(&self) {
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
