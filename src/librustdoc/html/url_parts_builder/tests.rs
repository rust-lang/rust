use super::*;

fn t(builder: UrlPartsBuilder, expect: &str) {
    assert_eq!(builder.finish(), expect);
}

#[test]
fn empty() {
    t(UrlPartsBuilder::new(), "");
}

#[test]
fn singleton() {
    t(UrlPartsBuilder::singleton("index.html"), "index.html");
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
fn push_front_empty() {
    let mut builder = UrlPartsBuilder::new();
    builder.push_front("page.html");
    t(builder, "page.html");
}

#[test]
fn push_front_non_empty() {
    let mut builder = UrlPartsBuilder::new();
    builder.push("core");
    builder.push("str");
    builder.push("struct.Bytes.html");
    builder.push_front("nightly");
    t(builder, "nightly/core/str/struct.Bytes.html");
}

#[test]
fn collect() {
    t(["core", "str"].into_iter().collect(), "core/str");
    t(["core", "str", "struct.Bytes.html"].into_iter().collect(), "core/str/struct.Bytes.html");
}

#[test]
fn extend() {
    let mut builder = UrlPartsBuilder::singleton("core");
    builder.extend(["str", "struct.Bytes.html"]);
    t(builder, "core/str/struct.Bytes.html");
}
