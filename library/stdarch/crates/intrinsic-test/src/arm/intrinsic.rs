use crate::common::intrinsic_helpers::IntrinsicType;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct ArmType(pub IntrinsicType);

impl Deref for ArmType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ArmType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
