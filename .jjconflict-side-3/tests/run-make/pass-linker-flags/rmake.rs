//@ needs-target-std
//
// This test checks the proper function of `-l link-arg=NAME`, which, unlike
// -C link-arg, is supposed to guarantee that the order relative to other -l
// options will be respected. In this test, compilation fails (because none of the
// link arguments like `a1` exist), but it is still checked if the output contains the
// link arguments in the exact order they were passed in. `attribute.rs` is a variant
// of the test where the flags are defined in the rust file itself.
// See https://github.com/rust-lang/rust/issues/99427

use run_make_support::{regex, rustc};

fn main() {
    let out = rustc()
        .input("empty.rs")
        .arg("-Zunstable-options")
        .arg("-lstatic=l1")
        .arg("-llink-arg=a1")
        .arg("-lstatic=l2")
        .arg("-llink-arg=a2")
        .arg("-ldylib=d1")
        .arg("-llink-arg=a3")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();
    let out2 = rustc().input("attribute.rs").print("link-args").run_unchecked().stdout_utf8();
    let re = regex::Regex::new("l1.*a1.*l2.*a2.*d1.*a3").unwrap();
    assert!(re.is_match(&out));
    assert!(re.is_match(&out2));
}
