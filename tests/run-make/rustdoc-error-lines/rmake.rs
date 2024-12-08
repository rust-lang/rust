// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

use run_make_support::rustdoc;

fn main() {
    let output = rustdoc().input("input.rs").arg("--test").run_fail().stdout_utf8();

    let should_contain = &[
        "input.rs - foo (line 5)",
        "input.rs:7:15",
        "input.rs - bar (line 13)",
        "input.rs:15:15",
        "input.rs - bar (line 22)",
        "input.rs:24:15",
    ];
    for text in should_contain {
        assert!(output.contains(text), "output doesn't contains {:?}", text);
    }
}
