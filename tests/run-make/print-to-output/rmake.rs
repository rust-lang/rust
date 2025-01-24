//! This checks the output of some `--print` options when output to a file (instead of stdout)

// ignore-tidy-linelength
//@ needs-llvm-components: aarch64 arm avr bpf csky hexagon loongarch m68k mips msp430 nvptx powerpc riscv sparc systemz webassembly x86
// FIXME(jieyouxu): there has to be a better way to do this, without the needs-llvm-components it
// will fail on LLVM built without all of the components listed above. If adding a new target that
// relies on a llvm component not listed above, it will need to be added to the required llvm
// components above.

use std::path::PathBuf;

use run_make_support::{rfs, rustc, target};

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

        rustc().target(args.target).print(&format!("{}={}", args.option, tmp_path.display())).run();

        rfs::read_to_string(&tmp_path)
    };

    check_(&stdout, args.includes);
    check_(&output, args.includes);

    assert_eq!(&stdout, &output);
}
