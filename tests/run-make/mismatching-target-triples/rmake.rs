// In this test, foo links against 32-bit architecture, and then, bar, which depends
// on foo, links against 64-bit architecture, causing a metadata mismatch due to the
// differences in target architectures. This used to cause an internal compiler error,
// now replaced by a clearer normal error message. This test checks that this aforementioned
// error message is used.
// See https://github.com/rust-lang/rust/issues/10814
//@ needs-llvm-components: x86

use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").target("i686-unknown-linux-gnu").run();
    rustc().input("bar.rs").target("x86_64-unknown-linux-gnu").run_fail().assert_stderr_contains(
        r#"couldn't find crate `foo` with expected target triple x86_64-unknown-linux-gnu"#,
    );
}
