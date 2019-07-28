// no-prefer-dynamic

#![crate_type = "rlib"]

use std::fmt;

pub fn work_with(p: &fmt::Debug) {
    drop(p);
}
