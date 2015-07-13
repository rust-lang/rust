extern crate build_helper;

use build_helper::{Config, build_static_lib};

fn main() {
    build_rust_test_helpers();
}

fn build_rust_test_helpers() {
    let cfg = Config::new();
    let src_dir = cfg.src_dir().join("rt");
    let src_files = vec!["rust_test_helpers.c"];
    build_static_lib(&cfg)
        .set_src_dir(&src_dir)
        .set_build_dir(&cfg.out_dir())
        .files(&src_files)
        .compile("rust_test_helpers");
}
