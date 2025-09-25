macro_rules! foo {
    () => {
        use $crate::{self}; //~ ERROR `$crate` may not be imported
    };
}

foo!();

fn main() {}
