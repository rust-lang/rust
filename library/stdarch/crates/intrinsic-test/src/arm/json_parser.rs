use super::constraint::Constraint;
use super::intrinsic::ArmIntrinsicType;
use crate::common::argument::{Argument, ArgumentList};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_types::{IntrinsicType, IntrinsicTypeDefinition};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ReturnType {
    value: String,
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
}

pub fn get_neon_intrinsics(
    filename: &Path,
    target: &String,
) -> Result<Vec<Intrinsic<ArmIntrinsicType, Constraint>>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let json: Vec<JsonIntrinsic> = serde_json::from_reader(reader).expect("Couldn't parse JSON");

    let parsed = json
        .into_iter()
        .filter_map(|intr| {
            if intr.simd_isa == "Neon" {
                Some(json_to_intrinsic(intr, target).expect("Couldn't parse JSON"))
            } else {
                None
            }
        })
        .collect();
    Ok(parsed)
}

fn json_to_intrinsic(
    mut intr: JsonIntrinsic,
    target: &String,
) -> Result<Intrinsic<ArmIntrinsicType, Constraint>, Box<dyn std::error::Error>> {
    let name = intr.name.replace(['[', ']'], "");

    let results = ArmIntrinsicType::from_c(&intr.return_type.value, target)?;

    let args = intr
        .arguments
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            let arg_name = Argument::<ArmIntrinsicType, Constraint>::type_and_name_from_c(&arg).1;
            let metadata = intr.args_prep.as_mut();
            let metadata = metadata.and_then(|a| a.remove(arg_name));
            let mut arg =
                Argument::<ArmIntrinsicType, Constraint>::from_c(i, &arg, target, metadata);

            // The JSON doesn't list immediates as const
            let IntrinsicType {
                ref mut constant, ..
            } = arg.ty.0;
            if arg.name.starts_with("imm") {
                *constant = true
            }
            arg
        })
        .collect();

    let arguments = ArgumentList::<ArmIntrinsicType, Constraint> { args };

    Ok(Intrinsic {
        name,
        arguments,
        results: *results,
        a64_only: intr.architectures == vec!["A64".to_string()],
    })
}
