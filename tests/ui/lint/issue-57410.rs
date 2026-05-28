//@ check-pass

// Tests that the `unreachable_pub` lint doesn't fire for `pub self::imp::f`.

#![deny(unreachable_pub)]

mod m {
    mod imp {
        pub fn f() {}
    }

    pub use self::imp::f;
}

pub use self::m::f;

fn main() {}
