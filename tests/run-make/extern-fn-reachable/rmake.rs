//@ ignore-cross-compile
use run_make_support::{rustc, symbols::contains_exact_symbols};

fn main() {
    rustc().input("dylib.rs").output("dylib.so").prefer_dynamic().run();
    assert!(contains_exact_symbols("dylib.so", &["fun1", "fun2", "fun3", "fun4", "fun5"]));
}
