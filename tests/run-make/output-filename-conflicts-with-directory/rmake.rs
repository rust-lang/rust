// ignore-tidy-linelength
// When the compiled executable would conflict with a directory, a
// rustc error should be displayed instead of a verbose and
// potentially-confusing linker error.
// See https://github.com/rust-lang/rust/pull/47203

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("foo");
    rustc().input("foo.rs").output("foo").run_fail().assert_stderr_contains(
        r#"the generated executable for the input file "foo.rs" conflicts with the existing directory "foo""#,
    );
}
