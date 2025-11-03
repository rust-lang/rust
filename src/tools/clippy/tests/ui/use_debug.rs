#![warn(clippy::use_debug)]

use std::fmt::{Debug, Display, Formatter, Result};

struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:?}", 43.1415)
        //~^ use_debug
    }
}

impl Debug for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        // ok, we can use `Debug` formatting in `Debug` implementations
        write!(f, "{:?}", 42.718)
    }
}

fn main() {
    print!("Hello {:?}", "World");
    //~^ use_debug

    print!("Hello {:#?}", "#orld");
    //~^ use_debug

    assert_eq!(42, 1337);

    vec![1, 2];
}

// don't get confused by nested impls
fn issue15942() {
    struct Bar;
    impl Debug for Bar {
        fn fmt(&self, f: &mut Formatter) -> Result {
            struct Baz;
            impl Debug for Baz {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    // ok, we can use `Debug` formatting in `Debug` implementations
                    write!(f, "{:?}", 42.718)
                }
            }

            // ok, we can use `Debug` formatting in `Debug` implementations
            write!(f, "{:?}", 42.718)
        }
    }
}
