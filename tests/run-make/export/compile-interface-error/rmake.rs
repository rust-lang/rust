use run_make_support::rustc;

fn main() {
    // Do not produce the interface, use the broken one.
    rustc()
        .input("app.rs")
        .run_fail()
        .assert_stderr_contains("couldn't compile interface");
}
