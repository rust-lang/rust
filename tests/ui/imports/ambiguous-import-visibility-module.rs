//@ check-pass
//@ edition:2018..

mod reexport {
    mod m {
        pub struct S {}
    }

    macro_rules! mac {
        () => {
            use m::S;
        };
    }

    pub use m::*;
    mac!();

    pub use S as Z;
}

fn main() {
    reexport::Z {};
}
