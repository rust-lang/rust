//@ run-pass
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2024
//@ ignore-stage1
//! Uses a rustc driver to check that test entrypoints get a `#[rustc_test_entrypoint_marker]`
//! and can be found using that attribute in rustc drivers (the main use for this attribute).

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_hir;

use interface::Compiler;
use rustc_driver::Compilation;
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use std::io::Write;

const CRATE_NAME: &str = "input";

struct TestAttr {
    expected_tests: usize,
}

impl rustc_driver::Callbacks for TestAttr {
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let mut tests = Vec::new();
        for did in tcx.hir_crate_items(()).definitions() {
            if find_attr!(tcx, did, RustcTestEntrypointMarker) {
                tests.push(did);
            }
        }

        // the file contains one test, so we should find one entrypoint marker.
        assert_eq!(tests.len(), self.expected_tests);

        Compilation::Stop
    }
}

fn count_tests(src: &str, expected_tests: usize) {
    let path = "test_input.rs";
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(src.as_bytes()).unwrap();

    let args = [
        "rustc".to_string(),
        "--test".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    rustc_driver::catch_fatal_errors(|| -> interface::Result<()> {
        rustc_driver::run_compiler(&args, &mut TestAttr { expected_tests });
        Ok(())
    })
    .unwrap()
    .unwrap();
}

fn main() {
    count_tests(
        r#"
        #[test]
        fn meow() {{ }}
        "#,
        1,
    );
    count_tests(
        r#"
        #[test]
        fn one() {{ }}

        #[test]
        fn two() {{ }}
        "#,
        2,
    );
}
