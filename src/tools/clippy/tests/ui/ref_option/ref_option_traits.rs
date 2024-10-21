//@no-rustfix: fixes are only done to traits, not the impls
//@revisions: private all
//@[private] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/private
//@[all] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/all

#![allow(unused, clippy::all)]
#![warn(clippy::ref_option)]

pub trait PubTrait {
    fn pub_trait_opt(&self, a: &Option<Vec<u8>>);
    fn pub_trait_ret(&self) -> &Option<Vec<u8>>;
}

trait PrivateTrait {
    fn trait_opt(&self, a: &Option<String>);
    fn trait_ret(&self) -> &Option<String>;
}

pub struct PubStruct;

impl PubTrait for PubStruct {
    fn pub_trait_opt(&self, a: &Option<Vec<u8>>) {}
    fn pub_trait_ret(&self) -> &Option<Vec<u8>> {
        panic!()
    }
}

struct PrivateStruct;

impl PrivateTrait for PrivateStruct {
    fn trait_opt(&self, a: &Option<String>) {}
    fn trait_ret(&self) -> &Option<String> {
        panic!()
    }
}

fn main() {}
