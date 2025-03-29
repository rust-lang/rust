#[derive(Default)]
pub struct Derived;

// @is '$.index[*][?(@.inner.for.inner.name == "Derived")].attrs' '["#[automatically_derived]"]'

pub struct ManualImpl;

// @is '$.index[*][?(@.docs == "The manual impl of default")].attrs' []
/// The manual impl of default
impl Default for ManualImpl {
    fn default() -> Self {
        Self
    }
}
