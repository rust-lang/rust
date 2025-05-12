// Ensure that `crate_name` and `crate_type` can be set through `-Z crate-attr`.
//@ check-pass
//@ compile-flags: -Zcrate-attr=crate_name="override"
fn main() {
    assert_eq!(module_path!(), "r#override");
}
