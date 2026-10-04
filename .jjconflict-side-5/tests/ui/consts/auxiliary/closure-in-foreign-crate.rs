//@ compile-flags: -Znext-solver
#![crate_type = "lib"]
#![feature(const_closures, const_trait_impl)]

pub const fn test() {
    let cl = const || {};
    cl();
}
