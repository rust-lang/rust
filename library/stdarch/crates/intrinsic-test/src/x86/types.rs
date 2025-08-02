use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::IntrinsicTypeDefinition;
use crate::x86::xml_parser::Parameter;

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the type in C format.
    /// This function assumes that this value is present in the metadata hashmap.
    fn c_type(&self) -> String {
        todo!("c_type from IntrinsicTypeDefinition is not defined!")
    }

    fn c_single_vector_type(&self) -> String {
        // matches __m128, __m256 and similar types
        todo!("c_type from IntrinsicTypeDefinition is not defined!")
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, _language: Language) -> String {
        todo!("get_load_function from IntrinsicTypeDefinition is not defined!")
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        todo!("get_lane_function for X86IntrinsicType needs to be implemented!");
    }

    fn from_c(s: &str, target: &str) -> Result<Self, String> {
        todo!("from_c from IntrinsicTypeDefinition is not defined!")
    }
}

impl X86IntrinsicType {
    pub fn from_param(param: &Parameter) -> Result<Self, String> {
        todo!("from_param from X86IntrinsicType is not defined!")
    }
}