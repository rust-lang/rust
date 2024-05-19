//@ check-pass

// Originally from #53925.
// Tests that the `unreachable_pub` lint doesn't fire for `pub self::bar::Bar`.

#![deny(unreachable_pub)]

mod foo {
    mod bar {
        pub struct Bar;
    }

    pub use self::bar::Bar;
}

pub use foo::Bar;

fn main() {}
