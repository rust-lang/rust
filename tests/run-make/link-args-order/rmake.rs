//@ needs-target-std
//
// Passing linker arguments to the compiler used to be lost or reordered in a messy way
// as they were passed further to the linker. This was fixed in #70665, and this test
// checks that linker arguments remain intact and in the order they were originally passed in.
// See https://github.com/rust-lang/rust/pull/70665

use run_make_support::{is_msvc, rustc};

fn main() {
    let linker = if is_msvc() { "msvc" } else { "ld" };

    rustc()
        .input("empty.rs")
        .linker_flavor(linker)
        .link_arg("a")
        .link_args("b c")
        .link_args("d e")
        .link_arg("f")
        .arg("--print=link-args")
        .run_fail()
        .assert_stdout_contains(r#""a" "b" "c" "d" "e" "f""#);
    rustc()
        .input("empty.rs")
        .linker_flavor(linker)
        .arg("-Zpre-link-arg=a")
        .arg("-Zpre-link-args=b c")
        .arg("-Zpre-link-args=d e")
        .arg("-Zpre-link-arg=f")
        .arg("--print=link-args")
        .run_fail()
        .assert_stdout_contains(r#""a" "b" "c" "d" "e" "f""#);
}
