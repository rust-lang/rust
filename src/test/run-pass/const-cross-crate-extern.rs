// aux-build:cci_const.rs


extern crate cci_const;
use cci_const::bar;
static foo: extern "C" fn() = bar;

pub fn main() {
    assert!(foo == bar);
}
