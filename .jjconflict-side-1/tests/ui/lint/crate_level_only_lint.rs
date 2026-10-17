#![deny(uncommon_codepoints, unused_attributes)]

mod foo {
#![allow(uncommon_codepoints)]
//~^ ERROR allow(uncommon_codepoints) is ignored unless specified at crate level [unused_attributes]

#[allow(uncommon_codepoints)]
//~^ ERROR allow(uncommon_codepoints) is ignored unless specified at crate level [unused_attributes]
const BAR: f64 = 0.000001;

}

#[allow(uncommon_codepoints)]
//~^ ERROR allow(uncommon_codepoints) is ignored unless specified at crate level [unused_attributes]
fn main() {
}
