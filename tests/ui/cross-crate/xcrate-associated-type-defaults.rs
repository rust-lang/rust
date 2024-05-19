//@ run-pass
//@ aux-build:xcrate_associated_type_defaults.rs

extern crate xcrate_associated_type_defaults;
use xcrate_associated_type_defaults::Foo;

struct LocalDefault;
impl Foo<u32> for LocalDefault {}

struct LocalOverride;
impl Foo<u64> for LocalOverride {
    type Out = bool;
}

fn main() {
    assert_eq!(
        <() as Foo<u32>>::Out::default().to_string(),
        "0");
    assert_eq!(
        <() as Foo<u64>>::Out::default().to_string(),
        "false");

    assert_eq!(
        <LocalDefault as Foo<u32>>::Out::default().to_string(),
        "0");
    assert_eq!(
        <LocalOverride as Foo<u64>>::Out::default().to_string(),
        "false");
}
