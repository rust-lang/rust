//@ needs-target-std
//@ ignore-cross-compile
// FIXME: remove the ignore gcc once fixed.
//@ ignore-backends: gcc
//@ ignore-windows-gnu
// GNU Linker for Windows is non-deterministic. (from `reproducible-build-2` test in this suite)

use run_make_support::parallel::check_reproducibility;

fn main() {
    check_reproducibility(
        "main",
        &[
            "--crate-type=lib",
            "-Zthreads=3",
            "-Clink-dead-code=true",
            "-Copt-level=0",
            "-Cembed-bitcode=true",
        ],
    );
}
