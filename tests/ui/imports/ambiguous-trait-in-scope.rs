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
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_trait_glob_imports]
                   //~| WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test2() {
    // Create an ambiguous import for `Trait` in another order
    use m2::*;
    use m1::*;
    0u8.method1(); //~ ERROR: no method named `method1` found for type `u8` in the current scope
    0u8.method2(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_trait_glob_imports]
                   //~| WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
