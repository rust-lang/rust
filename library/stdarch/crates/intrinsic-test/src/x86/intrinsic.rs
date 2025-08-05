use crate::common::argument::ArgumentList;
use crate::common::indentation::Indentation;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
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

impl IntrinsicDefinition<X86IntrinsicType> for Intrinsic<X86IntrinsicType> {
    fn arguments(&self) -> ArgumentList<X86IntrinsicType> {
        self.arguments.clone()
    }

    fn results(&self) -> X86IntrinsicType {
        self.results.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, _indentation: Indentation, _additional: &str) -> String {
        todo!("print_result_c in Intrinsic<X86IntrinsicType> needs to be implemented!");
    }
}
