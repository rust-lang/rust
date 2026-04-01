// Setting the linker flavor as a C compiler should cause the output of the -l flags to be
// prefixed by -Wl, except when a flag is requested to be verbatim. A bare linker (ld) should
// never cause prefixes to appear in the output. This test checks this ruleset twice, once with
// explicit flags and then with those flags passed inside the rust source code.
// See https://github.com/rust-lang/rust/pull/118202

//@ only-linux
// Reason: the `gnu-cc` linker is only available on linux

use run_make_support::{regex, rustc};

fn main() {
    let out_gnu = rustc()
        .input("empty.rs")
        .linker_flavor("gnu-cc")
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
    let out_gnu_verbatim = rustc()
        .input("empty.rs")
        .linker_flavor("gnu-cc")
        .arg("-Zunstable-options")
        .arg("-lstatic=l1")
        .arg("-llink-arg:+verbatim=a1")
        .arg("-lstatic=l2")
        .arg("-llink-arg=a2")
        .arg("-ldylib=d1")
        .arg("-llink-arg=a3")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();
    let out_ld = rustc()
        .input("empty.rs")
        .linker_flavor("ld")
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
    let out_att_gnu = rustc()
        .arg("-Zunstable-options")
        .linker_flavor("gnu-cc")
        .input("attribute.rs")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();
    let out_att_gnu_verbatim = rustc()
        .cfg(r#"feature="verbatim""#)
        .arg("-Zunstable-options")
        .linker_flavor("gnu-cc")
        .input("attribute.rs")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();
    let out_att_ld = rustc()
        .linker_flavor("ld")
        .input("attribute.rs")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();

    let no_verbatim = regex::Regex::new("l1.*-Wl,a1.*l2.*-Wl,a2.*d1.*-Wl,a3").unwrap();
    let one_verbatim = regex::Regex::new(r#"l1.*"a1".*l2.*-Wl,a2.*d1.*-Wl,a3"#).unwrap();
    let ld = regex::Regex::new(r#"l1.*"a1".*l2.*"a2".*d1.*"a3""#).unwrap();

    assert!(no_verbatim.is_match(&out_gnu));
    assert!(no_verbatim.is_match(&out_att_gnu));
    assert!(one_verbatim.is_match(&out_gnu_verbatim));
    assert!(one_verbatim.is_match(&out_att_gnu_verbatim));
    assert!(ld.is_match(&out_ld));
    assert!(ld.is_match(&out_att_ld));
}
