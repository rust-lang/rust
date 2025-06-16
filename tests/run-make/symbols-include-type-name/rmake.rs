//@ needs-target-std
//
// Method names used to be obfuscated when exported into symbols,
// leaving only an obscure `<impl>`. After the fix in #30328,
// this test checks that method names are successfully saved in the symbol list.
// See https://github.com/rust-lang/rust/issues/30260

use run_make_support::{invalid_utf8_contains, rustc};

fn main() {
    rustc().crate_type("staticlib").emit("asm").input("lib.rs").run();
    // Check that symbol names for methods include type names, instead of <impl>.
    invalid_utf8_contains("lib.s", "Def");
}
