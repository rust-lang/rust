pub mod inner {
    #[cfg(false)]
    mod gone {
        pub fn uwu() {}
    }

    #[cfg(false)] //~ NOTE the item is gated here
    pub use super::uwu;
    //~^ NOTE found an item that was configured out
}

pub use a::x;
//~^ ERROR unresolved import `a::x`
//~| NOTE no `x` in `a`

mod a {
    #[cfg(false)] //~ NOTE the item is gated here
    pub fn x() {}
    //~^ NOTE found an item that was configured out
}

pub use b::{x, y};
//~^ ERROR unresolved imports `b::x`, `b::y`
//~| NOTE no `x` in `b`
//~| NOTE no `y` in `b`

mod b {
    #[cfg(false)] //~ NOTE the item is gated here
    pub fn x() {}
    //~^ NOTE found an item that was configured out
    #[cfg(false)] //~ NOTE the item is gated here
    pub fn y() {}
    //~^ NOTE found an item that was configured out
}

fn main() {
    // There is no uwu at this path, but there's one in a cgfd out sub-module, so we mention it.
    inner::uwu(); //~ ERROR cannot find function
    //~^ NOTE not found in `inner`
}
