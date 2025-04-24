// Using SIMD types in a program with foreign-function interfaces used to result in an ICE
// (internal compiler error). Since this was fixed in #21233, it should be checked that
// compilation of SIMD and FFI together should be successful on all the most common
// architectures.
// Note that this test does not check linking or binary execution.
// See https://github.com/rust-lang/rust/pull/21233

use run_make_support::{llvm_components_contain, rustc};

fn main() {
    let mut targets = Vec::new();
    // arm-specific targets.
    if llvm_components_contain("arm") {
        targets.append(&mut vec![
            "arm-linux-androideabi".to_owned(),
            "arm-unknown-linux-gnueabihf".to_owned(),
            "arm-unknown-linux-gnueabi".to_owned(),
        ]);
    }
    let mut x86_archs = Vec::new();
    if llvm_components_contain("x86") {
        x86_archs.append(&mut vec!["i686", "x86_64"]);
    }
    // Linux has all x86 targets, plus aarch64 and mips.
    let mut extra_targets = x86_archs.clone();
    if llvm_components_contain("aarch64") {
        extra_targets.push("aarch64");
    }
    if llvm_components_contain("mips") {
        extra_targets.append(&mut vec!["mips", "mipsel"]);
    }

    for target in extra_targets {
        let linux = format!("{target}-unknown-linux-gnu");
        targets.push(linux);
    }

    // Windows and Darwin (OSX) only receive x86 targets.
    let extra_targets = x86_archs.clone();
    for target in extra_targets {
        let windows = format!("{target}-pc-windows-gnu");
        let darwin = format!("{target}-apple-darwin");
        targets.push(windows);
        targets.push(darwin);
    }

    for target in targets {
        // compile the rust file to the given target, but only to asm and IR
        // form, to avoid having to have an appropriate linker.
        //
        // we need some features because the integer SIMD instructions are not
        // enabled by-default for i686 and ARM; these features will be invalid
        // on some platforms, but LLVM just prints a warning so that's fine for
        // now.
        let target_feature = if target.starts_with("i686") || target.starts_with("x86") {
            "+sse2"
        } else if target.starts_with("arm") || target.starts_with("aarch64") {
            "-soft-float,+neon"
        } else if target.starts_with("mips") {
            "+msa,+fp64"
        } else {
            panic!("missing target_feature case for {target}");
        };
        rustc()
            .target(&target)
            .emit("llvm-ir,asm")
            .input("simd.rs")
            .arg(format!("-Ctarget-feature={target_feature}"))
            .arg(&format!("-Cextra-filename=-{target}"))
            .run();
    }
}
