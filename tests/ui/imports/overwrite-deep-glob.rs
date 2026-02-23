//@ check-pass

mod openssl {
    pub use self::handwritten::*;

    mod handwritten {
        mod m1 {
            pub struct S {}
        }
        mod m2 {
            #[derive(Default)]
            pub struct S {}
        }

        pub use self::m1::*; //~ WARN ambiguous glob re-exports
        pub use self::m2::*;
    }
}

pub use openssl::*;

fn main() {}
