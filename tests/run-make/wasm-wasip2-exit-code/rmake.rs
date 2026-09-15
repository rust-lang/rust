use run_make_support::{run_fail, rustc};

fn main() {
    rustc().arg("exit-code.rs").run();
    run_fail("exit-code.wasm").assert_exit_code(42);
}
