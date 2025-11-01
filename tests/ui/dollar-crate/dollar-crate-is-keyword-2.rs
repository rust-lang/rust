mod a {}

macro_rules! m {
    () => {
        use a::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use a::$crate::b; //~ ERROR `$crate` in paths can only be used in start position
        type A = a::$crate; //~ ERROR `$crate` in paths can only be used in start position
    }
}

m!();

fn main() {}
