use crate::common::intrinsic_helpers::IntrinsicType;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct ArmIntrinsicType {
    pub data: IntrinsicType,
    pub target: String,
}

impl Deref for ArmIntrinsicType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for ArmIntrinsicType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
