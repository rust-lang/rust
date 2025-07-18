/// Unstable feature bound can only be used only when
/// #[feature(staged_api)] is enabled.

pub trait Foo {
}
pub struct Bar;

#[unstable_feature_bound(feat_bar)]
//~^ ERROR: stability attributes may not be used outside of the standard library
impl Foo for Bar {
}

fn main(){}
