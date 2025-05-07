//@revisions: private all
//@[private] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/private
//@[all] rustc-env:CLIPPY_CONF_DIR=tests/ui/ref_option/all

#![allow(unused, clippy::needless_lifetimes, clippy::borrowed_box)]
#![warn(clippy::ref_option)]

fn opt_u8(a: &Option<u8>) {}
//~^ ref_option
fn opt_gen<T>(a: &Option<T>) {}
//~^ ref_option
fn opt_string(a: &std::option::Option<String>) {}
//~^ ref_option
fn ret_string<'a>(p: &'a str) -> &'a Option<u8> {
    //~^ ref_option
    panic!()
}
fn ret_string_static() -> &'static Option<u8> {
    //~^ ref_option
    panic!()
}
fn mult_string(a: &Option<String>, b: &Option<Vec<u8>>) {}
//~^ ref_option
fn ret_box<'a>() -> &'a Option<Box<u8>> {
    //~^ ref_option
    panic!()
}

pub fn pub_opt_string(a: &Option<String>) {}
//~[all]^ ref_option
pub fn pub_mult_string(a: &Option<String>, b: &Option<Vec<u8>>) {}
//~[all]^ ref_option

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

impl PubStruct {
    pub fn pub_opt_params(&self, a: &Option<()>) {}
    //~[all]^ ref_option
    pub fn pub_opt_ret(&self) -> &Option<String> {
        //~[all]^ ref_option
        panic!()
    }

    fn private_opt_params(&self, a: &Option<()>) {}
    //~^ ref_option
    fn private_opt_ret(&self) -> &Option<String> {
        //~^ ref_option
        panic!()
    }
}

// valid, don't change
fn mut_u8(a: &mut Option<u8>) {}
pub fn pub_mut_u8(a: &mut Option<String>) {}

// might be good to catch in the future
fn mut_u8_ref(a: &mut &Option<u8>) {}
pub fn pub_mut_u8_ref(a: &mut &Option<String>) {}
fn lambdas() {
    // Not handled for now, not sure if we should
    let x = |a: &Option<String>| {};
    let x = |a: &Option<String>| -> &Option<String> { panic!() };
}

fn main() {}
