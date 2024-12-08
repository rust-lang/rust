pub mod inner {
    #[cfg(FALSE)]
    mod gone {
        pub fn uwu() {}
    }

    #[cfg(FALSE)] //~ NOTE the item is gated here
    pub use super::uwu;
    //~^ NOTE found an item that was configured out
}

pub use a::x;
//~^ ERROR unresolved import `a::x`
//~| NOTE no `x` in `a`

mod a {
    #[cfg(FALSE)] //~ NOTE the item is gated here
    pub fn x() {}
    //~^ NOTE found an item that was configured out
}

pub use b::{x, y};
//~^ ERROR unresolved imports `b::x`, `b::y`
//~| NOTE no `x` in `b`
//~| NOTE no `y` in `b`

mod b {
    #[cfg(FALSE)] //~ NOTE the item is gated here
    pub fn x() {}
    //~^ NOTE found an item that was configured out
    #[cfg(FALSE)] //~ NOTE the item is gated here
    pub fn y() {}
    //~^ NOTE found an item that was configured out
}

fn main() {
    // There is no uwu at this path - no diagnostic.
    inner::uwu(); //~ ERROR cannot find function
    //~^ NOTE not found in `inner`
}
