//@ ignore-cross-compile
//@ needs-llvm-components: aarch64 x86
// FIXME(#132514): Is needs-llvm-components actually necessary for this test?

use run_make_support::{assert_contains_regex, rfs, rustc, target};

// Test that when querying `--print=target-cpus` for a target with the same
// architecture as the host, the first CPU is "native" with a suitable remark.

fn main() {
    let expected = r"^Available CPUs for this target:
    native +- Select the CPU of the current host \(currently [^ )]+\)\.
";

    // Without an explicit target.
    rustc().print("target-cpus").run().assert_stdout_contains_regex(expected);

    // With an explicit target that happens to be the host.
    let host = target(); // Because of ignore-cross-compile, assume host == target.
    rustc().print("target-cpus").target(host).run().assert_stdout_contains_regex(expected);

    // With an explicit output path.
    rustc().print("target-cpus=./xyzzy.txt").run().assert_stdout_equals("");
    assert_contains_regex(rfs::read_to_string("./xyzzy.txt"), expected);

    // Now try some cross-target queries with the same arch as the host.
    // (Specify multiple targets so that at least one of them is not the host.)
    let cross_targets: &[&str] = if cfg!(target_arch = "aarch64") {
        &["aarch64-unknown-linux-gnu", "aarch64-apple-darwin"]
    } else if cfg!(target_arch = "x86_64") {
        &["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"]
    } else {
        &[]
    };
    for target in cross_targets {
        println!("Trying target: {target}");
        rustc().print("target-cpus").target(target).run().assert_stdout_contains_regex(expected);
    }
}
