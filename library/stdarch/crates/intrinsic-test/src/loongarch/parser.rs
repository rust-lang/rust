use std::path::Path;

use super::intrinsic::LoongArchType;
use super::types::parse_intrinsic_type;
use crate::common::argument::{Argument, ArgumentList};
use crate::common::constraint::Constraint;
use crate::common::intrinsic::Intrinsic;
use crate::loongarch::LoongArch;

pub fn get_intrinsics(
    filename: &Path,
) -> Result<Vec<Intrinsic<LoongArch>>, Box<dyn std::error::Error>> {
    parse_spec_file(filename)
}

fn parse_spec_file(
    filename: &Path,
) -> Result<Vec<Intrinsic<LoongArch>>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(filename)?;
    parse_spec_contents(&contents)
}

fn parse_spec_contents(
    contents: &str,
) -> Result<Vec<Intrinsic<LoongArch>>, Box<dyn std::error::Error>> {
    let mut intrinsics = Vec::new();
    let mut record = Vec::new();

    for line in contents.lines().chain(std::iter::once("")) {
        let line = line.trim();
        if line.is_empty() {
            if !record.is_empty() {
                if let Some(intrinsic) = parse_record(&record)? {
                    intrinsics.push(intrinsic);
                }
                record.clear();
            }
            continue;
        }
        record.push(line);
    }

    Ok(intrinsics)
}

fn parse_record(
    record: &[&str],
) -> Result<Option<Intrinsic<LoongArch>>, Box<dyn std::error::Error>> {
    let mut name = None;
    let mut asm_formats = Vec::new();
    let mut data_types = None;

    for line in record {
        if let Some(value) = line.strip_prefix("name = ") {
            name = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("asm-fmts = ") {
            asm_formats = value
                .split(',')
                .map(|part| part.trim().to_string())
                .collect();
        } else if let Some(value) = line.strip_prefix("data-types = ") {
            data_types = Some(value);
        }
    }

    let Some(data_types) = data_types else {
        return Ok(None);
    };

    let name = name.ok_or("missing name before data-types")?;
    Ok(Some(parse_intrinsic(&name, &asm_formats, data_types)?))
}

fn parse_intrinsic(
    name: &str,
    asm_formats: &[String],
    data_types: &str,
) -> Result<Intrinsic<LoongArch>, Box<dyn std::error::Error>> {
    let data_types = data_types
        .split(',')
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let Some((result_type, argument_types)) = data_types.split_first() else {
        return Err("missing data-types for intrinsic".into());
    };
    let result = LoongArchType(parse_intrinsic_type(result_type)?);
    let asm_offset = asm_formats
        .len()
        .checked_sub(argument_types.len())
        .ok_or_else(|| format!("{name}: fewer asm formats than arguments"))?;
    let arguments = argument_types
        .iter()
        .enumerate()
        .map(|(pos, data_type)| {
            let constraint = asm_formats
                .get(pos + asm_offset)
                .and_then(|format| parse_constraint(name, format));
            let mut ty = LoongArchType(parse_intrinsic_type(data_type)?);
            if constraint.is_some() {
                ty.constant = true;
            }
            Ok(Argument::new(
                pos,
                format!("arg_{pos}"),
                ty,
                constraint,
                false,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(Intrinsic {
        name: name.to_string(),
        arguments: ArgumentList { args: arguments },
        results: result,
        arch_tags: Vec::new(),
        extension: extension_for(name)?.to_string(),
    })
}

fn parse_constraint(name: &str, asm_format: &str) -> Option<Constraint> {
    if let Some(cons) = special_constraint(name) {
        return Some(cons);
    }
    if let Some(bits) = asm_format.strip_prefix("ui") {
        return Some(unsigned_constraint(bits.parse::<u32>().ok()?));
    }
    if let Some(bits) = asm_format
        .strip_prefix("si")
        .or_else(|| asm_format.strip_prefix('i'))
    {
        return Some(signed_constraint(bits.parse::<u32>().ok()?));
    }
    None
}

fn special_constraint(name: &str) -> Option<Constraint> {
    match name {
        "lsx_vldi" | "lasx_xvldi" => {
            // CC: imm13 only support 0000 ~ 1100 in bits 9 ~ 12 when bit ‘13’ is 1
            let values: Vec<i64> = (0i64..8192i64)
                .filter(|&x| x < 4096 || ((x >> 8) & 0xf) < 13)
                // sign extend
                .map(|x| ((x << 51) as i64) >> 51)
                .collect();
            Some(Constraint::Set(values))
        }
        _ => None,
    }
}

fn unsigned_constraint(bits: u32) -> Constraint {
    Constraint::Range(0..(1i64 << bits.min(63)))
}

fn signed_constraint(bits: u32) -> Constraint {
    let min = if bits == 64 {
        i64::MIN
    } else {
        -(1i64 << (bits - 1))
    };
    let max = if bits == 64 {
        i64::MAX
    } else {
        1i64 << (bits - 1)
    };
    Constraint::Range(min..max)
}

fn extension_for(name: &str) -> Result<&'static str, Box<dyn std::error::Error>> {
    if name.starts_with("lasx_") {
        Ok("LASX")
    } else if name.starts_with("lsx_") {
        Ok("LSX")
    } else {
        Err(format!("unsupported LoongArch intrinsic name {name}").into())
    }
}
