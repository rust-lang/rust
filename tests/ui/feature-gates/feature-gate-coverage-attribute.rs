//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![crate_type = "lib"]
#![feature(no_coverage)] //~ ERROR feature has been removed [E0557]

#[derive(PartialEq, Eq)] // ensure deriving `Eq` does not enable `feature(coverage)`
struct Foo {
    a: u8,
    b: u32,
}

#[coverage(off)] //~ ERROR the `#[coverage]` attribute is an experimental feature
fn requires_feature_coverage() -> bool {
    let bar = Foo { a: 0, b: 0 };
    bar == Foo { a: 0, b: 0 }
}
