//@ needs-target-std
//@ needs-backends: llvm
//
// Regression test for https://github.com/rust-lang/rust/issues/152910.
//
// With v0 mangling, crate disambiguators appear before the function name.
// Sorting monomorphized items by their mangled names can therefore change
// codegen order when only -Cmetadata changes.
//
// Compile two crates with the same crate name but different disambiguators.
// The functions' order should follow their hash-free names, not the hashes.
//
// Also check distinct monomorphizations of the same function, where the
// crate disambiguators occur within the mangled generic arguments.

use run_make_support::{rfs, rustc};

fn main() {
    rustc()
        .input("right.rs")
        .arg("--crate-name=dupe")
        .crate_type("rlib")
        .metadata("fixed_right")
        .arg("-Csymbol-mangling-version=v0")
        .output("libright.rlib")
        .run();

    for metadata in ["alpha", "beta", "test_1", "test_2", "test_3"] {
        rustc()
            .input("left.rs")
            .arg("--crate-name=dupe")
            .crate_type("rlib")
            .metadata(metadata)
            .arg("-Csymbol-mangling-version=v0")
            .output("libleft.rlib")
            .run();

        let output = format!("{metadata}.ll");

        rustc()
            .input("main.rs")
            .arg("--crate-name=dupe_order_test")
            .extern_("left", "libleft.rlib")
            .extern_("right", "libright.rlib")
            .metadata("fixed_main")
            .arg("-Csymbol-mangling-version=v0")
            .codegen_units(1)
            .opt_level("0")
            .emit("llvm-ir")
            .output(&output)
            .run();

        let ir = rfs::read_to_string(&output);
        for (prefix, needles) in
            [("", ["left_function", "right_function"]), ("7generic", ["4Left", "5Right"])]
        {
            let order: Vec<_> = ir
                .lines()
                .filter(|line| line.starts_with("define ") && line.contains(prefix))
                .filter_map(|line| needles.iter().copied().find(|needle| line.contains(needle)))
                .collect();

            assert_eq!(order, needles, "unexpected symbol order with metadata {metadata}");
        }
    }
}
