use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::argument::{Argument, ArgumentList};
use crate::intrinsic::Intrinsic;
use crate::types::IntrinsicType;

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

#[derive(Deserialize, Debug)]
struct JsonIntrinsic {
    #[serde(rename = "SIMD_ISA")]
    simd_isa: String,
    name: String,
    arguments: Vec<String>,
    return_type: ReturnType,
    #[serde(rename = "Arguments_Preparation")]
    args_prep: Option<HashMap<String, ArgPrep>>,
    #[serde(rename = "Architectures")]
    architectures: Vec<String>,
}

pub fn get_neon_intrinsics(filename: &Path) -> Result<Vec<Intrinsic>, Box<dyn std::error::Error>> {
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

fn json_to_intrinsic(mut intr: JsonIntrinsic) -> Result<Intrinsic, Box<dyn std::error::Error>> {
    let name = intr.name.replace(['[', ']'], "");

    let results = IntrinsicType::from_c(&intr.return_type.value)?;

    let mut args_prep = intr.args_prep.as_mut();
    let args = intr
        .arguments
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            let arg_name = Argument::type_and_name_from_c(&arg).1;
            let arg_prep = args_prep.as_mut().and_then(|a| a.remove(arg_name));
            let mut arg = Argument::from_c(i, &arg, arg_prep);
            // The JSON doesn't list immediates as const
            if let IntrinsicType::Type {
                ref mut constant, ..
            } = arg.ty
            {
                if arg.name.starts_with("imm") {
                    *constant = true
                }
            }
            arg
        })
        .collect();

    let arguments = ArgumentList { args };

    Ok(Intrinsic {
        name,
        arguments,
        results,
        a64_only: intr.architectures == vec!["A64".to_string()],
    })
}
