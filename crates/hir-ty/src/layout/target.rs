//! Target dependent parameters needed for layouts

use std::sync::Arc;

use base_db::CrateId;
use hir_def::layout::{TargetDataLayout, TargetDataLayoutErrors};

use crate::db::HirDatabase;

use hir_def::layout::{AbiAndPrefAlign, AddressSpace, Align, Endian, Size};

pub fn target_data_layout_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<TargetDataLayout> {
    let crate_graph = db.crate_graph();
    let target_layout = &crate_graph[krate].target_layout;
    let cfg_options = &crate_graph[krate].cfg_options;
    Arc::new(
        target_layout
            .as_ref()
            .and_then(|it| parse_from_llvm_datalayout_string(it).ok())
            .unwrap_or_else(|| {
                let endian = match cfg_options.get_cfg_values("target_endian").next() {
                    Some(x) if x.as_str() == "big" => Endian::Big,
                    _ => Endian::Little,
                };
                let pointer_size = Size::from_bytes(
                    match cfg_options.get_cfg_values("target_pointer_width").next() {
                        Some(x) => match x.as_str() {
                            "16" => 2,
                            "32" => 4,
                            _ => 8,
                        },
                        _ => 8,
                    },
                );
                TargetDataLayout { endian, pointer_size, ..TargetDataLayout::default() }
            }),
    )
}

/// copied from rustc as it is not exposed yet
fn parse_from_llvm_datalayout_string<'a>(
    input: &'a str,
) -> Result<TargetDataLayout, TargetDataLayoutErrors<'a>> {
    // Parse an address space index from a string.
    let parse_address_space = |s: &'a str, cause: &'a str| {
        s.parse::<u32>().map(AddressSpace).map_err(|err| {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space: s, cause, err }
        })
    };

    // Parse a bit count from a string.
    let parse_bits = |s: &'a str, kind: &'a str, cause: &'a str| {
        s.parse::<u64>().map_err(|err| TargetDataLayoutErrors::InvalidBits {
            kind,
            bit: s,
            cause,
            err,
        })
    };

    // Parse a size string.
    let size = |s: &'a str, cause: &'a str| parse_bits(s, "size", cause).map(Size::from_bits);

    // Parse an alignment string.
    let align = |s: &[&'a str], cause: &'a str| {
        if s.is_empty() {
            return Err(TargetDataLayoutErrors::MissingAlignment { cause });
        }
        let align_from_bits = |bits| {
            Align::from_bits(bits)
                .map_err(|err| TargetDataLayoutErrors::InvalidAlignment { cause, err })
        };
        let abi = parse_bits(s[0], "alignment", cause)?;
        let pref = s.get(1).map_or(Ok(abi), |pref| parse_bits(pref, "alignment", cause))?;
        Ok(AbiAndPrefAlign { abi: align_from_bits(abi)?, pref: align_from_bits(pref)? })
    };

    let mut dl = TargetDataLayout::default();
    let mut i128_align_src = 64;
    for spec in input.split('-') {
        let spec_parts = spec.split(':').collect::<Vec<_>>();

        match &*spec_parts {
            ["e"] => dl.endian = Endian::Little,
            ["E"] => dl.endian = Endian::Big,
            [p] if p.starts_with('P') => {
                dl.instruction_address_space = parse_address_space(&p[1..], "P")?
            }
            ["a", ref a @ ..] => dl.aggregate_align = align(a, "a")?,
            ["f32", ref a @ ..] => dl.f32_align = align(a, "f32")?,
            ["f64", ref a @ ..] => dl.f64_align = align(a, "f64")?,
            [p @ "p", s, ref a @ ..] | [p @ "p0", s, ref a @ ..] => {
                dl.pointer_size = size(s, p)?;
                dl.pointer_align = align(a, p)?;
            }
            [s, ref a @ ..] if s.starts_with('i') => {
                let Ok(bits) = s[1..].parse::<u64>() else {
                    size(&s[1..], "i")?; // For the user error.
                    continue;
                };
                let a = align(a, s)?;
                match bits {
                    1 => dl.i1_align = a,
                    8 => dl.i8_align = a,
                    16 => dl.i16_align = a,
                    32 => dl.i32_align = a,
                    64 => dl.i64_align = a,
                    _ => {}
                }
                if bits >= i128_align_src && bits <= 128 {
                    // Default alignment for i128 is decided by taking the alignment of
                    // largest-sized i{64..=128}.
                    i128_align_src = bits;
                    dl.i128_align = a;
                }
            }
            [s, ref a @ ..] if s.starts_with('v') => {
                let v_size = size(&s[1..], "v")?;
                let a = align(a, s)?;
                if let Some(v) = dl.vector_align.iter_mut().find(|v| v.0 == v_size) {
                    v.1 = a;
                    continue;
                }
                // No existing entry, add a new one.
                dl.vector_align.push((v_size, a));
            }
            _ => {} // Ignore everything else.
        }
    }
    Ok(dl)
}
