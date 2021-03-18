#![feature(pub_macro_rules)]

#[macro_use]
mod m {
    pub macro_rules! mac { () => {} }

    // `pub` `macro_rules` cannot be redefined in the same module.
    pub macro_rules! mac { () => {} } //~ ERROR the name `mac` is defined multiple times

    pub(self) macro_rules! private_mac { () => {} }
}

const _: () = {
    pub macro_rules! block_mac { () => {} }
};

mod n {
    // Scope of `pub` `macro_rules` is not extended by `#[macro_use]`.
    mac!(); //~ ERROR cannot find macro `mac` in this scope

    // `pub` `macro_rules` doesn't put the macro into the root module, unlike `#[macro_export]`.
    crate::mac!(); //~ ERROR failed to resolve: maybe a missing crate `mac`
    crate::block_mac!(); //~ ERROR failed to resolve: maybe a missing crate `block_mac`

    crate::m::private_mac!(); //~ ERROR macro `private_mac` is private
}

fn main() {}
