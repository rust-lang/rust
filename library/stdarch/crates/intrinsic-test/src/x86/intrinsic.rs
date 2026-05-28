use crate::common::intrinsic_helpers::IntrinsicType;
use crate::x86::xml_parser::Parameter;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct X86IntrinsicType {
    pub data: IntrinsicType,
    pub param: Parameter,
}

impl Deref for X86IntrinsicType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for X86IntrinsicType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
