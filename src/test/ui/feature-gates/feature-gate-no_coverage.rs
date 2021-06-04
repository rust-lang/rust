#![crate_type = "lib"]

#[derive(PartialEq, Eq)] // ensure deriving `Eq` does not enable `feature(no_coverage)`
struct Foo {
    a: u8,
    b: u32,
}

#[no_coverage] //~ ERROR the `#[no_coverage]` attribute is an experimental feature
fn requires_feature_no_coverage() -> bool {
    let bar = Foo { a: 0, b: 0 };
    bar == Foo { a: 0, b: 0 }
}
