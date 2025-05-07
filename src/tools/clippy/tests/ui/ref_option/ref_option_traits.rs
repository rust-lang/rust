//@no-rustfix: fixes are only done to traits, not the impls
//@revisions: private all
//@[private] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/private
//@[all] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/all

#![warn(clippy::ref_option)]

pub trait PubTrait {
    fn pub_trait_opt(&self, a: &Option<Vec<u8>>);
    //~[all]^ ref_option
    fn pub_trait_ret(&self) -> &Option<Vec<u8>>;
    //~[all]^ ref_option
}

trait PrivateTrait {
    fn trait_opt(&self, a: &Option<String>);
    //~^ ref_option
    fn trait_ret(&self) -> &Option<String>;
    //~^ ref_option
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
