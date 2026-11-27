use foo::{bar, baz;

use std::fmt::Display;

mod bar { }

mod baz { }

//~v ERROR this file contains an unclosed delimiter
fn main() {}
