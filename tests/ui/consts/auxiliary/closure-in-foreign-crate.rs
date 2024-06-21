#![crate_type = "lib"]
#![feature(const_closures, const_trait_impl, effects)] //~ WARN the feature `effects` is incomplete

pub const fn test() {
    let cl = const || {};
    cl();
}
