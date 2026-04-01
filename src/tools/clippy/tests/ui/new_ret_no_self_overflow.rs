#![feature(type_alias_impl_trait)]
#![warn(clippy::new_ret_no_self)]

mod issue10041 {
    struct Bomb;

    impl Bomb {
        // Hidden <Rhs = Self> default generic parameter.
        pub fn new() -> impl PartialOrd {
            0i32
        }
    }

    // TAIT with self-referencing bounds
    type X = impl std::ops::Add<Output = X>;

    struct Bomb2;

    impl Bomb2 {
        #[define_opaque(X)]
        pub fn new() -> X {
            //~^ ERROR: overflow evaluating the requirement
            0i32
        }
    }
}

fn main() {}
