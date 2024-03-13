extern crate run_make_support;

use run_make_support::{rustdoc, tmp_dir};
use std::path::Path;
use std::{fs, iter};

fn generate_a_lot_of_cfgs(path: &Path) {
    let content = iter::repeat("--cfg=a\n").take(100_000).collect::<String>();
    fs::write(path, content.as_bytes()).expect("failed to create args file");
}

fn main() {
    let arg_file = tmp_dir().join("args");
    generate_a_lot_of_cfgs(&arg_file);

    rustdoc().out_dir(tmp_dir()).input("foo.rs").arg_file(&arg_file).arg("--test").run();
}
