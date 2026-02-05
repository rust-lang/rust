// This test builds `foo.rs` into a `cdylib` and verifies that
// `#[export_visibility = ...]` affects visibility of symbols.
//
// This test is loosely based on manual test steps described when
// discussing the related RFC at:
// https://github.com/rust-lang/rfcs/pull/3834#issuecomment-3403039933

use std::collections::HashSet;

use run_make_support::symbols::exported_dynamic_symbol_names;
use run_make_support::{dynamic_lib_name, object, rustc};

struct TestCase {
    name: &'static str,
    extra_rustc_arg: Option<&'static str>,
    expected_exported_symbols: &'static [&'static str],
    expected_private_symbols: &'static [&'static str],
}

impl TestCase {
    fn run(&self) {
        let test_name = self.name;

        let mut rustc = rustc();
        rustc.input("foo.rs");
        if let Some(extra_arg) = self.extra_rustc_arg {
            rustc.arg(extra_arg);
        }
        rustc.run();

        let lib_path = dynamic_lib_name("foo");
        let object_file_bytes = std::fs::read(&lib_path)
            .unwrap_or_else(|e| panic!("{test_name}: failed to read `{lib_path}`: {e}"));
        let object_file = object::File::parse(object_file_bytes.as_slice())
            .unwrap_or_else(|e| panic!("{test_name}: failed to parse `{lib_path}`: {e}"));
        let actual_exported_symbols =
            exported_dynamic_symbol_names(&object_file).into_iter().collect::<HashSet<_>>();

        for s in self.expected_exported_symbols {
            assert!(
                actual_exported_symbols.contains(s),
                "{test_name}: Expecting `{s}` to be an actually exported symbol in `{lib_path}`",
            );
        }
        for s in self.expected_private_symbols {
            assert!(
                !actual_exported_symbols.contains(s),
                "{test_name}: Expecting `{s}` to *not* be exported from `{lib_path}`",
            );
        }
    }
}

fn main() {
    TestCase {
        name: "Test #1",
        extra_rustc_arg: Some("-Zdefault-visibility=hidden"),
        expected_exported_symbols: &["test_fn_no_attr"],
        expected_private_symbols: &["test_fn_export_visibility_asks_for_target_default"],
    }
    .run();

    TestCase {
        name: "Test #2",
        extra_rustc_arg: None,
        expected_exported_symbols: &[
            "test_fn_no_attr",
            "test_fn_export_visibility_asks_for_target_default",
        ],
        expected_private_symbols: &[],
    }
    .run();
}
