extern crate build_helper;

use build_helper::{Config, build_static_lib};

fn main() {
    let cfg = Config::new();
    build_static_lib(&cfg).files(&["rt/miniz.c"]).compile("miniz");
}
