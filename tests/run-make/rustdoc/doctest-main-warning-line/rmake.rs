use run_make_support::rustdoc;

fn main() {
    rustdoc().input("foo.rs").arg("--test").run().assert_stderr_contains("foo.rs:5:5");
}
