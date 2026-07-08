use super::intrinsic::ArmType;
use crate::arm::Arm;
use crate::arm::types::parse_intrinsic_type;
use crate::common::argument::{Argument, ArgumentList};
use crate::common::constraint::Constraint;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::{IntrinsicType, TypeKind};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ReturnType {
    value: String,
    #[serde(rename = "element_bit_size")]
    _element_bit_size: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged, deny_unknown_fields)]
pub enum ArgPrep {
    Register {
        #[serde(rename = "register")]
        #[allow(dead_code)]
        reg: String,
    },
    Immediate {
        #[serde(rename = "minimum")]
        min: i64,
        #[serde(rename = "maximum")]
        max: i64,
    },
    Nothing {},
}

impl TryFrom<Value> for ArgPrep {
    type Error = serde_json::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[derive(Deserialize, Debug)]
struct JsonIntrinsic {
    #[serde(rename = "SIMD_ISA")]
    simd_isa: String,
    name: String,
    arguments: Vec<String>,
    return_type: ReturnType,
    #[serde(rename = "Arguments_Preparation")]
    args_prep: Option<HashMap<String, Value>>,
    #[serde(rename = "Architectures")]
    architectures: Vec<String>,
    #[serde(rename = "instructions")]
    _instructions: Option<Vec<Vec<String>>>,
}

pub fn get_neon_intrinsics(
    filename: &Path,
) -> Result<Vec<Intrinsic<Arm>>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let json: Vec<JsonIntrinsic> = serde_json::from_reader(reader).expect("Couldn't parse JSON");

    let parsed = json
        .into_iter()
        .filter_map(|intr| {
            if intr.simd_isa == "Neon" {
                Some(json_to_intrinsic(intr).expect("Couldn't parse JSON"))
            } else {
                None
            }
        })
        .collect();
    Ok(parsed)
}

fn json_to_intrinsic(
    mut intr: JsonIntrinsic,
) -> Result<Intrinsic<Arm>, Box<dyn std::error::Error>> {
    let name = intr.name.replace(['[', ']'], "");

    let result_ty = ArmType(parse_intrinsic_type(&intr.return_type.value)?);

    let args = intr
        .arguments
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            let (type_name, arg_name) = {
                let split_index = arg
                    .rfind([' ', '*'])
                    .expect("Couldn't split type and argname");

                (arg[..split_index + 1].trim_end(), &arg[split_index + 1..])
            };

            let arg_ty = parse_intrinsic_type(type_name)
                .unwrap_or_else(|_| panic!("Failed to parse argument '{arg}'"));

            let metadata = intr.args_prep.as_mut();
            let metadata = metadata.and_then(|a| a.remove(arg_name));
            let arg_prep: Option<ArgPrep> = metadata.and_then(|a| a.try_into().ok());
            let constraint: Option<Constraint> =
                arg_prep.and_then(|a| a.try_into().ok()).or_else(|| {
                    if arg_ty.kind() == TypeKind::SvPattern {
                        Some(Constraint::SvPattern)
                    } else if arg_ty.kind() == TypeKind::SvPrefetchOp {
                        Some(Constraint::SvPrefetchOp)
                    } else if arg_name == "imm_rotation" {
                        if name.starts_with("svcadd_") || name.starts_with("svqcadd_") {
                            Some(Constraint::SvImmRotationAdd)
                        } else {
                            Some(Constraint::SvImmRotation)
                        }
                    } else {
                        None
                    }
                });

            let mut arg =
                Argument::<Arm>::new(i, String::from(arg_name), ArmType(arg_ty), constraint);

            // The JSON doesn't list immediates as const
            let IntrinsicType {
                ref mut constant, ..
            } = *arg.ty;
            if arg.name.starts_with("imm") {
                *constant = true
            }
            arg
        })
        .collect();

    let arguments = ArgumentList::<Arm> { args };

    Ok(Intrinsic {
        name,
        arguments,
        results: result_ty,
        arch_tags: intr.architectures,
        extension: intr.simd_isa,
    })
}

/// ARM-specific
impl TryFrom<ArgPrep> for Constraint {
    type Error = ();

    fn try_from(prep: ArgPrep) -> Result<Self, Self::Error> {
        let parsed_ints = match prep {
            ArgPrep::Immediate { min, max } => Ok((min, max)),
            _ => Err(()),
        };
        if let Ok((min, max)) = parsed_ints {
            if min == max {
                Ok(Constraint::Equal(min))
            } else {
                Ok(Constraint::Range(min..max + 1))
            }
        } else {
            Err(())
        }
    }
}
