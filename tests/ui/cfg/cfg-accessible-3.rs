//@ edition:2015
#![feature(cfg_accessible)]

fn main() {
    assert!(cfg!(accessible(::std::boxed::Box))); //~ ERROR: `cfg(accessible(..))` cannot be used in edition 2015
}
