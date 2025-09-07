use crate::common::argument::ArgumentList;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
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

impl IntrinsicDefinition<ArmIntrinsicType> for Intrinsic<ArmIntrinsicType> {
    fn arguments(&self) -> ArgumentList<ArmIntrinsicType> {
        self.arguments.clone()
    }

    fn results(&self) -> ArmIntrinsicType {
        self.results.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
