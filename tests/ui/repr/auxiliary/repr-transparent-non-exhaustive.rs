#![crate_type = "lib"]

pub struct Private { _priv: () }

#[non_exhaustive]
pub struct NonExhaustive {}

#[non_exhaustive]
pub enum NonExhaustiveEnum {}

pub enum NonExhaustiveVariant {
    #[non_exhaustive]
    A,
}

pub struct ExternalIndirection<T> {
    pub x: T,
}
