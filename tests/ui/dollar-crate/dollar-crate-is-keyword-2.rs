mod a {}

macro_rules! m {
    () => {
        use a::$crate; //~ ERROR unresolved import `a::$crate`
        //~^ NOTE no `$crate` in `a`
        use a::$crate::b; //~ ERROR cannot find module `$crate`
        //~^ NOTE in paths can only be used in start position
        type A = a::$crate; //~ ERROR cannot find module `$crate`
        //~^ NOTE in paths can only be used in start position
    }
}

m!();
//~^ NOTE in this expansion
//~| NOTE in this expansion
//~| NOTE in this expansion

fn main() {}
