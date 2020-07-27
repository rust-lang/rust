// Check what token streams proc macros see when interpolated tokens are passed to them as input.

// check-pass
// normalize-stdout-test "#\d+" -> "#CTXT"
// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

macro_rules! pass_ident {
    ($i:ident) => {
        fn f() {
            print_bang!($i);
        }

        #[print_attr]
        const $i: u8 = 0;

        #[derive(Print)]
        struct $i {}
    };
}

pass_ident!(A);

fn main() {}
