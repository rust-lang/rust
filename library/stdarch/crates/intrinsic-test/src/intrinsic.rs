use crate::types::{IntrinsicType, TypeKind};

use super::argument::ArgumentList;

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsinc.
    pub arguments: ArgumentList,

    /// The return type of this intrinsic.
    pub results: IntrinsicType,
}

impl Intrinsic {
    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type.
    pub fn print_result_c(&self, index: usize, additional: &str) -> String {
        let lanes = if self.results.num_vectors() > 1 {
            (0..self.results.num_vectors())
                .map(|vector| {
                    format!(
                        r#""{ty}(" << {lanes} << ")""#,
                        ty = self.results.c_single_vector_type(),
                        lanes = (0..self.results.num_lanes())
                            .map(move |idx| -> std::string::String {
                                format!(
                                    "{cast}{lane_fn}(__return_value.val[{vector}], {lane})",
                                    cast = self.results.c_promotion(),
                                    lane_fn = self.results.get_lane_function(),
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
        } else if self.results.num_lanes() > 1 {
            (0..self.results.num_lanes())
                .map(|idx| -> std::string::String {
                    format!(
                        "{cast}{lane_fn}(__return_value, {lane})",
                        cast = self.results.c_promotion(),
                        lane_fn = self.results.get_lane_function(),
                        lane = idx
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else {
            format!(
                "{promote}cast<{cast}>(__return_value)",
                cast = match self.results.kind() {
                    TypeKind::Float if self.results.inner_size() == 32 => "float".to_string(),
                    TypeKind::Float if self.results.inner_size() == 64 => "double".to_string(),
                    TypeKind::Int => format!("int{}_t", self.results.inner_size()),
                    TypeKind::UInt => format!("uint{}_t", self.results.inner_size()),
                    TypeKind::Poly => format!("poly{}_t", self.results.inner_size()),
                    ty => todo!("print_result_c - Unknown type: {:#?}", ty),
                },
                promote = self.results.c_promotion(),
            )
        };

        format!(
            r#"std::cout << "Result {additional}-{idx}: {ty}" << std::fixed << std::setprecision(150) <<  {lanes} << "{close}" << std::endl;"#,
            ty = if self.results.is_simd() {
                format!("{}(", self.results.c_type())
            } else {
                String::from("")
            },
            close = if self.results.is_simd() { ")" } else { "" },
            lanes = lanes,
            additional = additional,
            idx = index,
        )
    }

    pub fn generate_pass_rust(&self, index: usize, additional: &str) -> String {
        let constraints = self.arguments.as_constraint_parameters_rust();
        let constraints = if !constraints.is_empty() {
            format!("::<{}>", constraints)
        } else {
            constraints
        };

        format!(
            r#"
    unsafe {{
        {initialized_args}
        let res = {intrinsic_call}{const}({args});
        println!("Result {additional}-{idx}: {{:.150?}}", res);
    }}"#,
            initialized_args = self.arguments.init_random_values_rust(index),
            intrinsic_call = self.name,
            args = self.arguments.as_call_param_rust(),
            additional = additional,
            idx = index,
            const = constraints,
        )
    }

    pub fn generate_pass_c(&self, index: usize, additional: &str) -> String {
        format!(
            r#"  {{
    {initialized_args}
    auto __return_value = {intrinsic_call}({args});
    {print_result}
  }}"#,
            initialized_args = self.arguments.init_random_values_c(index),
            intrinsic_call = self.name,
            args = self.arguments.as_call_param_c(),
            print_result = self.print_result_c(index, additional)
        )
    }
}
