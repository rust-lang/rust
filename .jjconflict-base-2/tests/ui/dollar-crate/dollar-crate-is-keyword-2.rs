mod a {}

macro_rules! m {
    () => {
        use a::$crate; //~ ERROR: `$crate` in paths can only be used in start position
        use a::$crate::b; //~ ERROR: `$crate` in paths can only be used in start position
        //~^ NOTE: can only be used in path start position
        type A = a::$crate; //~ ERROR: `$crate` in paths can only be used in start position
        //~^ NOTE: can only be used in path start position
    }
}

m!();
//~^ NOTE: in this expansion
//~| NOTE: in this expansion
//~| NOTE: in this expansion

fn main() {}
