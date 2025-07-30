// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

//@ needs-target-std

use run_make_support::rustdoc;

fn main() {
    let output = rustdoc().input("input.rs").arg("--test").run_fail().stdout_utf8();

    let should_contain = &[
        "input.rs:5 - foo",
        "input.rs:8:15",
        "input.rs:13 - bar",
        "input.rs:16:15",
        "input.rs:22 - bar",
        "input.rs:25:15",
    ];
    for text in should_contain {
        assert!(output.contains(text), "output doesn't contains {:?}", text);
    }
}
