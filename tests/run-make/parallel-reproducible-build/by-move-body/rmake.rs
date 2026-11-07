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
            "--edition=2021",
            "-Zremap-cwd-prefix=reproducible_dir",
            "-Ccodegen-units=1",
            "-Zthreads=2",
            "--crate-type=lib",
        ],
    );
}
