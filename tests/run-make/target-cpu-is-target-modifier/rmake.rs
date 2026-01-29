//! This checks that the target-cpu set when compiling foo is correctly set as a target-modifier
//! and also correctly set as the target-cpu.
//! It does this for two scenarios:
//! `-Ctarget-cpu is only specified one time when compiling `foo`
//! `-Ctarget-cpu` is specified twice when compiling `foo`
//! In both cases the last `-Ctarget-cpu` argument must be set as target-cpu and target-modifier.
//! The `-Ctarget-cpu` target-modifier-value is checked by compiling `bar` which has `foo` as
//! a dependency.
//! The error message tells us the `-Ctarget-cpu` target-modifier-value.
//! The target-cpu value is checked by filechecking foo.ll.

//@ ignore-backends: gcc

use std::collections::BTreeMap;

use run_make_support::*;

fn main() {
    let targets = BTreeMap::from([
        ("avr-none", ["at43usb320", "atmega328p"]),
        ("amdgcn-amd-amdhsa", ["gfx900", "gfx1100"]),
        ("nvptx64-nvidia-cuda", ["sm_60", "sm_70"]),
    ]);

    for (target, target_cpus) in targets {
        // compile foo with target_cpu[1]
        let mut foo_rustc = rustc();
        foo_rustc
            .target(target)
            .target_cpu(target_cpus[1])
            .input("foo.rs")
            .panic("abort")
            .emit("llvm-ir,link")
            .inspect(|cmd| {
                assert_eq!(
                    cmd.get_args()
                        .filter(|arg| arg.to_str().is_some_and(|arg| arg.contains("-Ctarget-cpu")))
                        .count(),
                    1
                );
            })
            .run();
        verify(target, &target_cpus);
        // compile foo with target_cpu[0] and target_cpu[1]
        let mut foo_rustc = rustc();
        let cmd = foo_rustc
            .target(target)
            .target_cpu(target_cpus[0])
            .target_cpu(target_cpus[1])
            .input("foo.rs")
            .panic("abort")
            .emit("llvm-ir,link")
            .inspect(|cmd| {
                assert_eq!(
                    cmd.get_args()
                        .filter(|arg| arg.to_str().is_some_and(|arg| arg.contains("-Ctarget-cpu")))
                        .count(),
                    2
                );
            })
            .run();
        verify(target, &target_cpus);
    }
}

fn verify(target: &str, target_cpus: &[&str]) {
    // verify that foo has target_cpu[1] set as modifier by compiling bar with target_cpu[0]
    // FIXME: This is a little hacky, to query the target modifier by an error
    // message. A way to read the value in the metadata would be preferrable.
    // Also this test would then be usable with other targets than the ones above.
    let mut bar_rustc = rustc();
    bar_rustc
        .target(target)
        .target_cpu(target_cpus[0])
        .input("bar.rs")
        .panic("abort")
        .run_fail()
        .assert_stderr_contains(
            "error: mixing `-Ctarget-cpu` will cause \
            an ABI mismatch in crate `bar`",
        )
        .assert_stderr_contains(format!(
            "`-Ctarget-cpu={}` in this crate is incompatible with \
            `-Ctarget-cpu={}` in dependency `foo`",
            target_cpus[0], target_cpus[1]
        ));
    // verify that foo has target_cpu[1] set as target cpu by filechecking foo with llvm filecheck
    let mut filecheck = llvm_filecheck();
    filecheck.patterns(target).input_file("foo.ll").run();
}
