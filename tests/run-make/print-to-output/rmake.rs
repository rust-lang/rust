//! This checks the output of some `--print` options when
//! output to a file (instead of stdout)

extern crate run_make_support;

use std::ffi::OsString;

use run_make_support::{rustc, target, tmp_dir};

fn main() {
    // Printed from CodegenBackend trait impl in rustc_codegen_llvm/src/lib.rs
    check(
        /*target*/ &target(),
        /*option*/ "relocation-models",
        /*includes*/ &["dynamic-no-pic"],
    );

    // Printed by compiler/rustc_codegen_llvm/src/llvm_util.rs
    check(
        /*target*/ "wasm32-unknown-unknown",
        /*option*/ "target-features",
        /*includes*/ &["reference-types"],
    );

    // Printed by C++ code in rustc_llvm/llvm-wrapper/PassWrapper.cpp
    check(
        /*target*/ "wasm32-unknown-unknown",
        /*option*/ "target-cpus",
        /*includes*/ &["generic"],
    );
}

fn check(target: &str, option: &str, includes: &[&str]) {
    fn _inner(output: &str, includes: &[&str]) {
        for i in includes {
            assert!(output.contains(i), "output doesn't contains: {}", i);
        }
    }

    // --print={option}
    let stdout = {
        let output = rustc().target(target).print(option).run();

        let stdout = String::from_utf8(output.stdout).unwrap();

        _inner(&stdout, includes);

        stdout
    };

    // --print={option}=PATH
    let output = {
        let tmp_path = tmp_dir().join(format!("{option}.txt"));
        let mut print_arg = OsString::from(format!("--print={option}="));
        print_arg.push(tmp_path.as_os_str());

        let _output = rustc().target(target).arg(print_arg).run();

        let output = std::fs::read_to_string(&tmp_path).unwrap();

        _inner(&output, includes);

        output
    };

    assert_eq!(&stdout, &output);
}
