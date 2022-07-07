#![crate_type = "lib"]

pub struct Private { _priv: () }

#[non_exhaustive]
pub struct NonExhaustive {}

pub struct ExternalIndirection<T> {
    pub x: T,
}
