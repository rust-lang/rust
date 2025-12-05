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
