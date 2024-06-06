//! This checks the output of some `--print` options when
//! output to a file (instead of stdout)

use std::ffi::OsString;
use std::path::PathBuf;

use run_make_support::{fs_wrapper, rustc, target};

struct Option<'a> {
    target: &'a str,
    option: &'static str,
    includes: &'static [&'static str],
}

fn main() {
    // Printed from CodegenBackend trait impl in rustc_codegen_llvm/src/lib.rs
    check(Option { target: &target(), option: "relocation-models", includes: &["dynamic-no-pic"] });

    // Printed by compiler/rustc_codegen_llvm/src/llvm_util.rs
    check(Option {
        target: "wasm32-unknown-unknown",
        option: "target-features",
        includes: &["reference-types"],
    });

    // Printed by C++ code in rustc_llvm/llvm-wrapper/PassWrapper.cpp
    check(Option {
        target: "wasm32-unknown-unknown",
        option: "target-cpus",
        includes: &["generic"],
    });
}

fn check(args: Option) {
    fn check_(output: &str, includes: &[&str]) {
        for i in includes {
            assert!(output.contains(i), "output doesn't contains: {}", i);
        }
    }

    // --print={option}
    let stdout = rustc().target(args.target).print(args.option).run().stdout_utf8();

    // --print={option}=PATH
    let output = {
        let tmp_path = PathBuf::from(format!("{}.txt", args.option));
        let mut print_arg = OsString::from(format!("--print={}=", args.option));
        print_arg.push(tmp_path.as_os_str());

        rustc().target(args.target).arg(print_arg).run();

        fs_wrapper::read_to_string(&tmp_path)
    };

    check_(&stdout, args.includes);
    check_(&output, args.includes);

    assert_eq!(&stdout, &output);
}
