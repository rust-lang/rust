// `--version --verbose` should display the git-commit hashes of rustc and rustdoc, but this
// functionality was lost due to #104184. After this feature was returned by #109981, this
// test ensures it will not be broken again.
// See https://github.com/rust-lang/rust/issues/107094

//@ needs-git-hash

use run_make_support::{bare_rustc, regex, rustdoc};

fn main() {
    let out_rustc =
        bare_rustc().arg("--version").arg("--verbose").run().stdout_utf8().to_lowercase();
    let out_rustdoc =
        rustdoc().arg("--version").arg("--verbose").run().stdout_utf8().to_lowercase();
    let re =
        regex::Regex::new(r#"commit-hash: [0-9a-f]{40}\ncommit-date: [0-9]{4}-[0-9]{2}-[0-9]{2}"#)
            .unwrap();
    assert!(re.is_match(&out_rustc));
    assert!(re.is_match(&out_rustdoc));
}
