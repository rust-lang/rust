//@ check-pass

mod m1 {
    pub trait Trait {
        fn method1(&self) {}
    }
    impl Trait for u8 {}
}
mod m2 {
    pub trait Trait {
        fn method2(&self) {}
    }
    impl Trait for u8 {}
}

pub fn test1() {
    // Create an ambiguous import for `Trait` in one order
    use m1::*;
    use m2::*;
    0u8.method1(); //~ WARNING Usage of ambiguously imported trait `Trait` [ambiguous_trait_glob_imports]
    0u8.method2(); //~ WARNING Usage of ambiguously imported trait `Trait` [ambiguous_trait_glob_imports]
}

fn test2() {
    // Create an ambiguous import for `Trait` in another order
    use m2::*;
    use m1::*;
    0u8.method1(); //~ WARNING Usage of ambiguously imported trait `Trait` [ambiguous_trait_glob_imports]
    0u8.method2(); //~ WARNING Usage of ambiguously imported trait `Trait` [ambiguous_trait_glob_imports]
}

fn main() {}
