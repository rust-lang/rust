//@compile-flags: --test
//@revisions: default enabled
//@[enabled] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/check_incompatible_msrv_in_tests/enabled
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/check_incompatible_msrv_in_tests/default

#![warn(clippy::incompatible_msrv)]
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.3.0"]

use std::thread::sleep;
use std::time::Duration;

fn main() {
    sleep(Duration::new(1, 0))
    //~^ incompatible_msrv
}

#[test]
fn test() {
    sleep(Duration::new(1, 0));
    //~[enabled]^ incompatible_msrv
}

#[cfg(test)]
mod tests {
    use super::*;
    fn helper() {
        sleep(Duration::new(1, 0));
        //~[enabled]^ incompatible_msrv
    }
}
