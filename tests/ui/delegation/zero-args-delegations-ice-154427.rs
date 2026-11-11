#![feature(fn_delegation)]

mod ice_154427 {
    trait Trait {
        fn foo();
    }
    struct F;
    struct S;
    mod to_reuse {
        use super::F;
        pub fn foo(_: F) {}
    }
    impl Trait for S {
        reuse to_reuse::foo { self }
        //~^ ERROR: delegation's target expression is specified for function with no params
    }

    fn main() {}
}

fn main() {}
