// error-pattern: this file contains an unclosed delimiter
use foo::{bar, baz;

use std::fmt::Display;

mod bar { }

mod baz { }

fn main() {}
