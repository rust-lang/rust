// Make sure that existing root attributes are still respected even when `-Zcrate-attr` is present.
//@ run-pass
//@ compile-flags: -Zcrate-attr=feature(rustc_attrs)
#![crate_name = "override"]

#[rustc_dummy]
fn main() {
    assert_eq!(module_path!(), "r#override");
}
