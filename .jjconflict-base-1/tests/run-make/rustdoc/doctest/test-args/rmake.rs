//@ ignore-cross-compile (needs to run doctest binary)

use std::iter;
use std::path::Path;

use run_make_support::{rfs, rustdoc};

fn generate_a_lot_of_cfgs(path: &Path) {
    let content = iter::repeat("--cfg=a\n").take(100_000).collect::<String>();
    rfs::write(path, content.as_bytes());
}

fn main() {
    let arg_file = Path::new("args");
    generate_a_lot_of_cfgs(&arg_file);

    rustdoc().input("foo.rs").arg_file(&arg_file).arg("--test").run();
}
