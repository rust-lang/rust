use super::*;

fn t(builder: UrlPartsBuilder, expect: &str) {
    assert_eq!(builder.finish(), expect);
}

#[test]
fn empty() {
    t(UrlPartsBuilder::new(), "");
}

#[test]
fn push_several() {
    let mut builder = UrlPartsBuilder::new();
    builder.push("core");
    builder.push("str");
    builder.push("struct.Bytes.html");
    t(builder, "core/str/struct.Bytes.html");
}

#[test]
fn push_fmt() {
    let mut builder = UrlPartsBuilder::new();
    builder.push("nightly");
    builder.push_fmt(format_args!("{}", "core"));
    builder.push("str");
    builder.push_fmt(format_args!("{}.{}.html", "struct", "Bytes"));
    t(builder, "nightly/core/str/struct.Bytes.html");
}

#[test]
fn collect() {
    t(["core", "str"].into_iter().collect(), "core/str");
    t(["core", "str", "struct.Bytes.html"].into_iter().collect(), "core/str/struct.Bytes.html");
}

#[test]
fn extend() {
    let mut builder = UrlPartsBuilder::new();
    builder.push("core");
    builder.extend(["str", "struct.Bytes.html"]);
    t(builder, "core/str/struct.Bytes.html");
}
