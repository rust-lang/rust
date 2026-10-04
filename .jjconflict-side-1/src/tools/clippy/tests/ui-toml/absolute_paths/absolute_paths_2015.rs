//@revisions: default allow_crates
//@[default]rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/default
//@[allow_crates]rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/allow_crates
//@[allow_crates]check-pass
//@edition:2015

#![deny(clippy::absolute_paths)]

mod m1 {
    pub mod m2 {
        pub struct X;
    }
}

fn main() {
    let _ = ::m1::m2::X; //~[default] absolute_paths
}
