//@ ignore-cross-compile

use run_make_support::rustc;

fn main() {
    rustc().input("windows.rs").run();
    rustc().input("console.rs").run();
}
