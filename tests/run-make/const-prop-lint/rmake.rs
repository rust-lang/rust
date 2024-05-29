// Tests that const prop lints interrupting codegen don't leave `.o` files around.

use std::fs;

use run_make_support::{rustc, tmp_dir};

fn main() {
    rustc().input("input.rs").run_fail_assert_exit_code(1);

    for entry in fs::read_dir(tmp_dir()).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() && path.extension().is_some_and(|ext| ext == "o") {
            panic!("there should not be `.o` files!");
        }
    }
}
