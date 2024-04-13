// Check that rustc accepts various version info flags.
//@ dont-check-compiler-stdout
//@ revisions: version verbose-version long-verbose-version
//@ check-pass
//@[version] compile-flags: -V
//@[verbose-version] compile-flags: -vV
//@[long-verbose-verison] compile-flags: --version --verbose

fn main() {}
