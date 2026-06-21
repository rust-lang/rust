use super::intrinsic::ArmType;
use crate::common::intrinsic_helpers::{IntrinsicType, Sign, SimdLen, TypeDefinition, TypeKind};

impl TypeDefinition for ArmType {
    /// Gets a string containing the typename for this type in C format.
    fn c_type(&self) -> String {
        let prefix = self.kind.c_prefix();

        if let Some(bit_len) = self.bit_len {
            match (self.simd_len, self.vec_len) {
                (None, None) => format!("{prefix}{bit_len}_t"),
                (Some(SimdLen::Fixed(simd)), None) => format!("{prefix}{bit_len}x{simd}_t"),
                (Some(SimdLen::Fixed(simd)), Some(vec)) => {
                    format!("{prefix}{bit_len}x{simd}x{vec}_t")
                }
                (Some(SimdLen::Scalable), None) => format!("sv{prefix}{bit_len}_t"),
                (Some(SimdLen::Scalable), Some(vec)) => {
                    format!("sv{prefix}{bit_len}x{vec}_t")
                }
                (None, Some(_)) => todo!("{self:#?}"), // Likely an invalid case
            }
        } else {
            todo!("{self:#?}")
        }
    }

    fn rust_type(&self) -> String {
        let rust_prefix = self.kind.rust_prefix();
        let c_prefix = self.kind.c_prefix();

        if let Some(bit_len) = self.bit_len {
            match (self.simd_len, self.vec_len) {
                (None, None) => format!("{rust_prefix}{bit_len}"),
                (Some(SimdLen::Fixed(simd)), None) => format!("{c_prefix}{bit_len}x{simd}_t"),
                (Some(SimdLen::Fixed(simd)), Some(vec)) => {
                    format!("{c_prefix}{bit_len}x{simd}x{vec}_t")
                }
                (Some(SimdLen::Scalable), None) => format!("sv{c_prefix}{bit_len}_t"),
                (Some(SimdLen::Scalable), Some(vec)) => {
                    format!("sv{c_prefix}{bit_len}x{vec}_t")
                }
                (None, Some(_)) => todo!("{self:#?}"), // Likely an invalid case
            }
        } else {
            todo!("{self:#?}")
        }
    }

    /// Determines the load function for this type.
    fn get_load_function(&self) -> String {
        if let IntrinsicType {
            kind: k,
            bit_len: Some(bl),
            vec_len,
            ..
        } = **self
        {
            let quad = if self.num_lanes() * bl > 64 { "q" } else { "" };

            format!(
                "vld{len}{quad}_{type}{size}",
                type = match k {
                    TypeKind::Int(Sign::Unsigned) => "u",
                    TypeKind::Int(Sign::Signed) => "s",
                    TypeKind::Float => "f",
                    TypeKind::Poly => "p",
                    x => todo!("get_load_function TypeKind: {x:#?}"),
                },
                size = bl,
                quad = quad,
                len = vec_len.unwrap_or(1),
            )
        } else {
            todo!("get_load_function IntrinsicType: {self:#?}")
        }
    }
}

pub fn parse_intrinsic_type(s: &str) -> Result<IntrinsicType, String> {
    const CONST_STR: &str = "const";
    const ENUM_STR: &str = "enum ";

    // Recurse to handle pointers..
    if let Some(s) = s.strip_suffix('*') {
        let s = s.trim();
        let (s, constant) = if s.ends_with(CONST_STR) || s.starts_with(CONST_STR) {
            (
                s.trim_start_matches(CONST_STR).trim_end_matches(CONST_STR),
                true,
            )
        } else {
            (s, false)
        };

        let mut ty = parse_intrinsic_type(s.trim())?;
        ty.ptr = true;
        ty.ptr_constant = constant;
        return Ok(ty);
    }

    // [const ][sv]TYPE[{element_bits}[x{num_lanes}[x{num_vecs}]]][_t]
    //   | [enum ]TYPE
    let (mut s, constant) = match (s.strip_prefix(CONST_STR), s.strip_prefix(ENUM_STR)) {
        (Some(const_strip), _) => (const_strip, true),
        (_, Some(enum_strip)) => (enum_strip, true),
        (None, None) => (s, false),
    };
    s = s.trim();
    s = s.strip_suffix("_t").unwrap_or(s);

    // Consider the following types as examples:
    // A) `svuint32x3_t`
    // B) `float16x4x2_t`
    // C) `svbool_t`

    let sve = s.starts_with("sv");

    let mut parts = s.split('x');
    let start = parts.next().ok_or("failed to parse type")?;

    // Continuing the previous examples..
    // A) kind=TypeKind::Int(Sign::Unsigned), bit_len=Some(32)
    // B) kind=TypeKind::Float, bit_len=Some(16)
    // C) kind=TypeKind::Bool, bit_len=None
    let (kind, bit_len) = if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
        let (element_kind, element_bits) = start.split_at(digit_start);
        let element_kind = element_kind.parse::<TypeKind>()?;
        let element_bits = element_bits.parse::<u32>().map_err(|err| err.to_string())?;
        (element_kind, Some(element_bits))
    } else {
        let element_kind = start.parse::<TypeKind>()?;
        (element_kind, None)
    };

    let bit_len = match (bit_len, kind) {
        (None, TypeKind::SvPattern | TypeKind::SvPrefetchOp | TypeKind::Int(_)) => Some(32),
        (None, TypeKind::Bool) => Some(8),
        _ => bit_len,
    };

    // Continuing the previous examples..
    // A) second_len=Some(3)
    // B) second_len=Some(4)
    // C) second_len=None
    let second_len = parts.next().map(|part| {
        part.parse::<u32>()
            .expect("failed to parse second part of type")
    });

    // Continuing the previous examples..
    // A) third_len=None
    // B) third_len=Some(2)
    // C) third_len=None
    let third_len = parts.next().map(|part| {
        part.parse::<u32>()
            .expect("failed to parse third part of type")
    });

    let (simd_len, vec_len) = if sve {
        (Some(SimdLen::Scalable), second_len)
    } else {
        (second_len.map(SimdLen::Fixed), third_len)
    };

    Ok(IntrinsicType {
        ptr: false,
        ptr_constant: false,
        constant,
        kind,
        bit_len,
        simd_len,
        vec_len,
    })
}
