//@ revisions: default feature
//@ only-x86_64
#![cfg_attr(feature, feature(effective_target_features))]
//[feature]~^ WARN the feature `effective_target_features` is incomplete and may not be safe to use and/or cause compiler crashes

trait Foo {
    fn foo(&self);
}

struct Bar;

impl Foo for Bar {
    #[unsafe(force_target_feature(enable = "avx2"))]
    //[default]~^ ERROR the `#[force_target_feature]` attribute is an experimental feature
    fn foo(&self) {}
}

struct Bar2;

impl Foo for Bar2 {
    #[target_feature(enable = "avx2")]
    //~^ ERROR `#[target_feature(..)]` cannot be applied to safe trait method
    fn foo(&self) {}
    //~^ ERROR method `foo` has an incompatible type for trait
}

fn main() {}
