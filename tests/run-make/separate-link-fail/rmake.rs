// rustc usually wants Rust code as its input. The flag `link-only` is one
// exception, where a .rlink file is instead requested. The compiler should
// fail when the user is wrongly passing the original Rust code
// instead of the generated .rlink file when this flag is on.
// https://github.com/rust-lang/rust/issues/95297

use run_make_support::rustc;

fn main() {
    rustc()
        .arg("-Zlink-only")
        .input("foo.rs")
        .run_fail()
        .assert_stderr_contains("the input does not look like a .rlink file");
}
