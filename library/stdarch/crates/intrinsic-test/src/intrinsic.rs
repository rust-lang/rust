use crate::types::{IntrinsicType, TypeKind};

use super::argument::ArgumentList;
use serde::de::Unexpected;
use serde::{de, Deserialize, Deserializer};

/// An intrinsic
#[derive(Deserialize, Debug, PartialEq, Clone)]
pub struct Intrinsic {
    /// If the intrinsic should be tested.
    #[serde(deserialize_with = "bool_from_string")]
    pub enabled: bool,

    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsinc.
    #[serde(rename = "args")]
    pub arguments: ArgumentList,

    /// The return type of this intrinsic.
    #[serde(rename = "return")]
    pub results: IntrinsicType,
}

impl Intrinsic {
    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type.
    pub fn print_result_c(&self, index: usize) -> String {
        let lanes = if self.results.num_lanes() > 1 {
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
            r#"std::cout << "Result {idx}: {ty}" << std::fixed << std::setprecision(150) <<  {lanes} << "{close}" << std::endl;"#,
            ty = if self.results.is_simd() {
                format!("{}(", self.results.c_type())
            } else {
                String::from("")
            },
            close = if self.results.is_simd() { ")" } else { "" },
            lanes = lanes,
            idx = index,
        )
    }

    pub fn generate_pass_rust(&self, index: usize) -> String {
        format!(
            r#"
    unsafe {{
        {initialized_args}
        let res = {intrinsic_call}({args});
        println!("Result {idx}: {{:.150?}}", res);
    }}"#,
            initialized_args = self.arguments.init_random_values_rust(index),
            intrinsic_call = self.name,
            args = self.arguments.as_call_param_rust(),
            idx = index,
        )
    }

    pub fn generate_pass_c(&self, index: usize) -> String {
        format!(
            r#"  {{
    {initialized_args}
    auto __return_value = {intrinsic_call}({args});
    {print_result}
  }}"#,
            initialized_args = self.arguments.init_random_values_c(index),
            intrinsic_call = self.name,
            args = self.arguments.as_call_param_c(),
            print_result = self.print_result_c(index)
        )
    }
}

fn bool_from_string<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    match String::deserialize(deserializer)?.to_uppercase().as_ref() {
        "TRUE" => Ok(true),
        "FALSE" => Ok(false),
        other => Err(de::Error::invalid_value(
            Unexpected::Str(other),
            &"TRUE or FALSE",
        )),
    }
}
