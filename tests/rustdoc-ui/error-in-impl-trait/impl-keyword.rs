pub trait ValidTrait {}
pub fn g() -> impl ValidTrait {
    error::_in::impl_trait()
    //~^ ERROR use of undeclared crate or module `error`
}
