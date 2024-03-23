extern crate run_make_support;

use run_make_support::{out_dir, rustdoc};
use std::{fs, iter};
use std::path::Path;

fn generate_a_lot_of_cfgs(path: &Path) {
    let content = iter::repeat("--cfg=a\n").take(100_000).collect::<String>();
    fs::write(path, content.as_bytes()).expect("failed to create args file");
}

fn main() {
    let arg_file = out_dir().join("args");
    generate_a_lot_of_cfgs(&arg_file);

    let arg_file = format!("@{}", arg_file.display());
    rustdoc().arg("--test").arg(&arg_file).arg("foo.rs").run();
}
