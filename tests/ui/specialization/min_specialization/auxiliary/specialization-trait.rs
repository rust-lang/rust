#![feature(rustc_attrs)]

#[rustc_specialization_trait]
pub trait SpecTrait {
    fn method(&self);
}
