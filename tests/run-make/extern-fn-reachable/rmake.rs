//@ ignore-cross-compile
use run_make_support::object::ObjectSymbol;
use run_make_support::rustc;
use run_make_support::symbols::assert_contains_exact_symbols;
fn main() {
    rustc().input("dylib.rs").output("dylib.so").prefer_dynamic().run();
    assert_contains_exact_symbols("dylib.so", &["fun1", "fun2", "fun3", "fun4", "fun5"], |sym| {
        dbg!(dbg!(sym).is_global()) && !dbg!(sym.is_undefined())
    });
}
