//@ check-pass
pub trait ValidTrait {}
/// This returns impl trait
pub fn g() -> impl ValidTrait {
    error::_in::impl_trait()
}
