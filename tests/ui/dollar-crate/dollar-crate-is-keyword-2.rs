mod a {}

macro_rules! m {
    () => {
        use a::$crate; //~ ERROR unresolved import `a::$crate`
        use a::$crate::b; //~ ERROR cannot find module `$crate`
        type A = a::$crate; //~ ERROR cannot find module `$crate`
    }
}

m!();

fn main() {}
