use crate::common::argument::ArgumentList;
use crate::common::indentation::Indentation;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, TypeKind};
use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
pub struct ArmIntrinsicType(pub IntrinsicType);

impl Deref for ArmIntrinsicType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.0
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

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, indentation: Indentation, additional: &str) -> String {
        let lanes = if self.results().num_vectors() > 1 {
            (0..self.results().num_vectors())
                .map(|vector| {
                    format!(
                        r#""{ty}(" << {lanes} << ")""#,
                        ty = self.results().c_single_vector_type(),
                        lanes = (0..self.results().num_lanes())
                            .map(move |idx| -> std::string::String {
                                format!(
                                    "{cast}{lane_fn}(__return_value.val[{vector}], {lane})",
                                    cast = self.results().c_promotion(),
                                    lane_fn = self.results().get_lane_function(),
                                    lane = idx,
                                    vector = vector,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(r#" << ", " << "#)
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else if self.results().num_lanes() > 1 {
            (0..self.results().num_lanes())
                .map(|idx| -> std::string::String {
                    format!(
                        "{cast}{lane_fn}(__return_value, {lane})",
                        cast = self.results().c_promotion(),
                        lane_fn = self.results().get_lane_function(),
                        lane = idx
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else {
            format!(
                "{promote}cast<{cast}>(__return_value)",
                cast = match self.results.kind() {
                    TypeKind::Float if self.results().inner_size() == 16 => "float16_t".to_string(),
                    TypeKind::Float if self.results().inner_size() == 32 => "float".to_string(),
                    TypeKind::Float if self.results().inner_size() == 64 => "double".to_string(),
                    TypeKind::Int => format!("int{}_t", self.results().inner_size()),
                    TypeKind::UInt => format!("uint{}_t", self.results().inner_size()),
                    TypeKind::Poly => format!("poly{}_t", self.results().inner_size()),
                    ty => todo!("print_result_c - Unknown type: {:#?}", ty),
                },
                promote = self.results().c_promotion(),
            )
        };

        format!(
            r#"{indentation}std::cout << "Result {additional}-" << i+1 << ": {ty}" << std::fixed << std::setprecision(150) <<  {lanes} << "{close}" << std::endl;"#,
            ty = if self.results().is_simd() {
                format!("{}(", self.results().c_type())
            } else {
                String::from("")
            },
            close = if self.results.is_simd() { ")" } else { "" },
        )
    }
}
