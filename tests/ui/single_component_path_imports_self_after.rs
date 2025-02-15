//@ check-pass

#![warn(clippy::single_component_path_imports)]
#![allow(unused_imports)]

use self::regex::{Regex as xeger, RegexSet as tesxeger};
#[rustfmt::skip]
pub use self::{
    regex::{Regex, RegexSet},
    some_mod::SomeType,
};
use regex;

mod some_mod {
    pub struct SomeType;
}

fn main() {}
