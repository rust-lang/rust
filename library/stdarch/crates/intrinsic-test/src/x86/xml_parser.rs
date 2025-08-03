use crate::common::argument::{Argument, ArgumentList};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use crate::x86::constraint::map_constraints;

use serde::{Deserialize, Deserializer};
use std::path::Path;

use super::intrinsic::X86IntrinsicType;

// Custom deserializer function to convert strings to u32
fn string_to_u32<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    return s.as_str().parse::<u32>().or(Ok(0u32));
}

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<XMLIntrinsic>,
}

#[derive(Deserialize)]
struct XMLIntrinsic {
    #[serde(rename = "return")]
    pub return_data: Parameter,
    #[serde(rename = "@name")]
    pub name: String,
    // #[serde(rename = "@tech")]
    // tech: String,
    #[serde(rename = "CPUID", default)]
    cpuid: Vec<String>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
pub struct Parameter {
    #[serde(rename = "@varname")]
    pub var_name: String,
    #[serde(rename = "@type")]
    pub type_data: String,
    #[serde(rename = "@etype", default)]
    pub etype: String,
    #[serde(rename = "@memwidth", default, deserialize_with = "string_to_u32")]
    pub memwidth: u32,
    #[serde(rename = "@immtype", default)]
    pub imm_type: String,
}

pub fn get_xml_intrinsics(
    filename: &Path,
) -> Result<Vec<Intrinsic<X86IntrinsicType>>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let data: Data =
        quick_xml::de::from_reader(reader).expect("failed to deserialize the source XML file");

    let parsed_intrinsics: Vec<Intrinsic<X86IntrinsicType>> = data
        .intrinsics
        .into_iter()
        .filter_map(|intr| {
            // Some(xml_to_intrinsic(intr, target).expect("Couldn't parse XML properly!"))
            xml_to_intrinsic(intr).ok()
        })
        .collect();

    Ok(parsed_intrinsics)
}

fn xml_to_intrinsic(
    intr: XMLIntrinsic,
) -> Result<Intrinsic<X86IntrinsicType>, Box<dyn std::error::Error>> {
    let name = intr.name;
    let result = X86IntrinsicType::from_param(&intr.return_data);
    let args_check = intr.parameters.into_iter().enumerate().map(|(i, param)| {
        let ty = X86IntrinsicType::from_param(&param);
        if ty.is_err() {
            None
        } else {
            let constraint = map_constraints(&param.imm_type);
            let arg = Argument::<X86IntrinsicType>::new(
                i,
                param.var_name.clone(),
                ty.unwrap(),
                constraint,
            );
            Some(arg)
        }
    });

    let args = args_check.collect::<Vec<_>>();
    if args.iter().any(|elem| elem.is_none()) {
        return Err(Box::from("intrinsic isn't fully supported in this test!"));
    }
    let args = args
        .into_iter()
        .map(|e| e.unwrap())
        .filter(|arg| arg.ty.ptr || arg.ty.kind != TypeKind::Void)
        .collect::<Vec<_>>();
    let arguments = ArgumentList::<X86IntrinsicType> { args };

    if let Err(message) = result {
        return Err(Box::from(message));
    }
    Ok(Intrinsic {
        name,
        arguments,
        results: result.unwrap(),
        arch_tags: intr.cpuid,
    })
}
