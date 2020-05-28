// aux-build: test-macros.rs

extern crate test_macros;
use test_macros::recollect_attr;

macro_rules! reemit {
    ($name:ident => $($token:expr)*) => {

        #[recollect_attr]
        pub fn $name() {
            $($token)*;
        }
    }
}

reemit! { foo => 45u32.into() } //~ ERROR type annotations

fn main() {}
