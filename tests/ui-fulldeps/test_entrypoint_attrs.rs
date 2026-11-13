//@ run-pass
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2024
//@ ignore-stage1
//! Uses a rustc driver to check that test entrypoints get a test trace attribute
//! and can be found using that attribute in rustc drivers (the main use for this attribute).

#![feature(rustc_private)]

extern crate rustc_attr_ir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;

use std::io::Write;

use rustc_attr_ir::find_attr;
use rustc_driver::Compilation;
use rustc_interface::interface::{self, Compiler};
use rustc_middle::ty::TyCtxt;

struct TestAttr {
    expected_tests: &'static [&'static str],
}

impl rustc_driver::Callbacks for TestAttr {
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let mut tests = Vec::new();
        for did in tcx.hir_crate_items(()).definitions() {
            if find_attr!(tcx, did, RustcTestEntrypointMarker) {
                tests.push(tcx.item_name(did.to_def_id()).to_string());
            }
        }
        tests.sort();

        assert_eq!(tests, self.expected_tests);

        Compilation::Stop
    }
}

fn find_tests(src: &str, expected_tests: &'static [&'static str]) {
    let path = "test_input.rs";
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(src.as_bytes()).unwrap();

    let args = [
        "rustc".to_string(),
        "--test".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        "input".to_string(),
        path.to_string(),
    ];
    rustc_driver::catch_fatal_errors(|| -> interface::Result<()> {
        rustc_driver::compiler_entrypoint(&args, &mut TestAttr { expected_tests });
        Ok(())
    })
    .unwrap()
    .unwrap();
}

fn main() {
    find_tests(
        r#"
        #[test]
        fn meow() {}
        "#,
        &["meow"],
    );

    find_tests(
        r#"
        #[test]
        fn one() {}

        #[test]
        #[should_panic]
        #[ignore = "reason"]
        fn two() { panic!() }

        fn not_a_test() {}
        "#,
        &["one", "two"],
    );

    find_tests(
        r#"
        #![feature(test)]
        extern crate test;

        #[bench]
        fn bench(_: &mut test::Bencher) {}
        "#,
        &["bench"],
    );

    // the marker is synthetic, so macros don't see it (...and don't re-emit it)
    find_tests(
        r#"
        #![feature(macro_attr)]

        macro_rules! re_emit {
            attr() { $(#[$a:meta])* fn $name:ident () $body: block } => {
                $(#[$a])*
                fn $name () $body
            }
        }

        #[re_emit]
        #[test]
        fn before() {}

        #[test]
        #[re_emit]
        fn after() {}
        "#,
        &["before"],
    );
}
