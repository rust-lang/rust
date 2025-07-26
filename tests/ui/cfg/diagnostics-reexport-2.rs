// issue#141256

mod original {
    #[cfg(false)]
    //~^ NOTE the item is gated here
    //~| NOTE the item is gated here
    //~| NOTE the item is gated here
    //~| NOTE the item is gated here
    //~| NOTE the item is gated here
    pub mod gated {
    //~^ NOTE found an item that was configured out
    //~| NOTE found an item that was configured out
    //~| NOTE found an item that was configured out
    //~| NOTE found an item that was configured out
    //~| NOTE found an item that was configured out
        pub fn foo() {}
    }
}

mod reexport {
    pub use super::original::*;
}

mod reexport2 {
    pub use super::reexport::*;
}

mod reexport30 {
    pub use super::original::*;
    pub use super::reexport31::*;
}

mod reexport31 {
    pub use super::reexport30::*;
}

mod reexport32 {
    pub use super::reexport30::*;
}

fn main() {
    reexport::gated::foo();
    //~^ ERROR failed to resolve: could not find `gated` in `reexport`
    //~| NOTE  could not find `gated` in `reexport`

    reexport2::gated::foo();
    //~^ ERROR failed to resolve: could not find `gated` in `reexport2`
    //~| NOTE  could not find `gated` in `reexport2`

    reexport30::gated::foo();
    //~^ ERROR failed to resolve: could not find `gated` in `reexport30`
    //~| NOTE  could not find `gated` in `reexport30`

    reexport31::gated::foo();
    //~^ ERROR failed to resolve: could not find `gated` in `reexport31`
    //~| NOTE  could not find `gated` in `reexport31`

    reexport32::gated::foo();
    //~^ ERROR failed to resolve: could not find `gated` in `reexport32`
    //~| NOTE  could not find `gated` in `reexport32`
}
