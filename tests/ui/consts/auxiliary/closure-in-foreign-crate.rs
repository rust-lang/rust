#![crate_type = "lib"]
#![feature(const_closures, const_trait_impl, effects)]

pub const fn test() {
    let cl = const || {};
    cl();
}
