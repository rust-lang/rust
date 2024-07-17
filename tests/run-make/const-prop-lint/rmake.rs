// Tests that const prop lints interrupting codegen don't leave `.o` files around.

use run_make_support::{cwd, rfs, rustc};

fn main() {
    rustc().input("input.rs").run_fail().assert_exit_code(1);

    for entry in rfs::read_dir(cwd()) {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() && path.extension().is_some_and(|ext| ext == "o") {
            panic!("there should not be `.o` files!");
        }
    }
}
