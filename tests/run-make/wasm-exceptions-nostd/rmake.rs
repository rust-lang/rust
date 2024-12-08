//@ only-wasm32-bare

use std::path::Path;

use run_make_support::{cmd, env_var, rustc};

fn main() {
    // Add a few command line args to make exceptions work
    rustc()
        .input(Path::new("src").join("lib.rs"))
        .target("wasm32-unknown-unknown")
        .panic("unwind")
        .arg("-Cllvm-args=-wasm-enable-eh")
        .arg("-Ctarget-feature=+exception-handling")
        .run();

    cmd(&env_var("NODE")).arg("verify.mjs").arg("lib.wasm").run();
}
