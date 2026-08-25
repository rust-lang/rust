// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.

#![deny(unused_attributes)]

mod a {
    #![crate_type = "bin"] //~ ERROR the `#![crate_type]` attribute can only be used at the crate root
}

#[crate_type = "bin"] fn main() {} //~ ERROR should be an inner
