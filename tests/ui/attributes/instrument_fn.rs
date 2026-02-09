#![feature(instrument_fn)]

#[instrument_fn = "on"] //~ ERROR attribute cannot be used on
struct F;

#[instrument_fn = "on"] //~ ERROR attribute cannot be used on
mod module {}

#[instrument_fn = "on"] //~ ERROR attribute cannot be used on
impl F {
    #[instrument_fn = "off"]
    fn no_instrument_fn(self, x: u32) -> u32 {
        #[instrument_fn = "off"] //~ ERROR attribute cannot be used on
        //~^ ERROR attributes on expressions are experimental
        x * 2
    }
}

#[instrument_fn = "off"] //~ ERROR attribute cannot be used on
trait Foo {
    #[instrument_fn = "off"] //~ ERROR attribute cannot be used on
    fn bar();

    #[instrument_fn = "off"]
    fn baz() {}
}

impl Foo for F {
    #[instrument_fn = "off"]
    fn bar() {}
}

#[instrument_fn = "on"]
fn main() {}

#[instrument_fn(entry = "on")] //~ ERROR malformed
fn instrument_fn_list() {}

#[instrument_fn] //~ ERROR malformed
fn instrument_fn_noarg() {}

#[instrument_fn = 1] //~ ERROR malformed
fn instrument_fn_invalid_opt() {}
