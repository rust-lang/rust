//! Before #78122, writing any `fmt::Arguments` would trigger the inclusion of `usize` formatting
//! and padding code in the resulting binary, because indexing used in `fmt::write` would generate
//! code using `panic_bounds_check`, which prints the index and length.
//!
//! These bounds checks are not necessary, as `fmt::Arguments` never contains any out-of-bounds
//! indexes. The test is a `run-make` test, because it needs to check the result after linking. A
//! codegen or assembly test doesn't check the parts that will be pulled in from `core` by the
//! linker.
//!
//! In this test, we try to check that the `usize` formatting and padding code are not present in
//! the final binary by checking that panic symbols such as `panic_bounds_check` are **not**
//! present.
//!
//! Some CI jobs try to run faster by disabling debug assertions (through setting
//! `NO_DEBUG_ASSERTIONS=1`). If std debug assertions are disabled, then we can check for the
//! absence of additional `usize` formatting and padding related symbols.

// ignore-tidy-linelength

//@ ignore-cross-compile

use std::path::Path;

use run_make_support::env::std_debug_assertions_enabled;
use run_make_support::llvm::{llvm_filecheck, llvm_pdbutil};
use run_make_support::symbols::object_contains_any_symbol_substring;
use run_make_support::{bin_name, is_windows_msvc, rfs, rustc};

fn main() {
    // panic machinery identifiers, these should not appear in the final binary
    let mut panic_syms = vec!["panic_bounds_check", "Debug"];
    if std_debug_assertions_enabled() {
        // if debug assertions are allowed, we need to allow these,
        // otherwise, add them to the list of symbols to deny.
        panic_syms.extend_from_slice(&["panicking", "panic_fmt", "pad_integral", "Display"]);
    }

    let expect_no_panic_symbols = bin_name("expect_no_panic_symbols");
    let expect_no_panic_symbols = Path::new(&expect_no_panic_symbols);
    rustc().input("main.rs").output(&expect_no_panic_symbols).opt().run();

    let expect_panic_symbols = bin_name("expect_panic_symbols");
    let expect_panic_symbols = Path::new(&expect_panic_symbols);
    rustc().input("main.rs").output(&expect_panic_symbols).run();

    if is_windows_msvc() {
        // FIXME(#143737): use actual DIA wrappers instead of parsing `llvm-pdbutil` textual output.

        let expect_no_filecheck_pattern = r#"
            CHECK: main
            CHECK-NOT: {{.*}}Display{{.*}}
            CHECK-NOT: {{.*}}panic_bounds_check{{.*}}
        "#;
        let expect_no_filecheck_path = Path::new("expect_no_panic_symbols_filecheck.txt");
        rfs::write(&expect_no_filecheck_path, expect_no_filecheck_pattern);

        let expect_no_panic_symbols_pdbutil_dump = llvm_pdbutil()
            .arg("dump")
            .arg("-publics")
            .input(expect_no_panic_symbols.with_extension("pdb"))
            .run();
        llvm_filecheck()
            .patterns(expect_no_filecheck_path)
            .stdin_buf(expect_no_panic_symbols_pdbutil_dump.stdout_utf8())
            .run();

        // NOTE: on different platforms, they may not go through the same code path. E.g. on
        // `i686-msvc`, we do not go through `panic_fmt` (only `panic`). So for the check here we
        // only try to match for mangled `core::panicking` module path.
        let expect_filecheck_pattern = r#"
            CHECK: main
            CHECK: {{.*}}core{{.*}}panicking{{.*}}
            // _ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize...
            CHECK: {{.*}}Display{{.*}}
            CHECK-NOT: {{.*}}panic_bounds_check{{.*}}
        "#;
        let expect_filecheck_path = Path::new("expect_panic_symbols_filecheck.txt");
        rfs::write(&expect_filecheck_path, expect_filecheck_pattern);

        let expect_panic_symbols_pdbutil_dump = llvm_pdbutil()
            .arg("dump")
            .arg("-publics")
            .input(expect_panic_symbols.with_extension("pdb"))
            .run();
        llvm_filecheck()
            .patterns(expect_filecheck_path)
            .stdin_buf(expect_panic_symbols_pdbutil_dump.stdout_utf8())
            .run();
    } else {
        // At least the `main` symbol (or `_main`) should be present.
        assert!(object_contains_any_symbol_substring(&expect_no_panic_symbols, &["main"]));
        assert!(object_contains_any_symbol_substring(&expect_panic_symbols, &["main"]));

        assert!(!object_contains_any_symbol_substring(&expect_no_panic_symbols, &panic_syms));
        assert!(object_contains_any_symbol_substring(&expect_panic_symbols, &panic_syms));
    }
}
