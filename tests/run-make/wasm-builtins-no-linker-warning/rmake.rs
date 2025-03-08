//@ only-wasm32-bare

use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").target("wasm32-unknown-unknown").arg("-D").arg("linker-messages").run();
}
