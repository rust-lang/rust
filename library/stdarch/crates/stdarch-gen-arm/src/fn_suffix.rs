use std::fmt::{self};

/* This file is acting as a bridge between the old neon types and how they
 * have a fairly complex way of picking suffixes and the new world. If possible
 * it would be good to clean this up. At least it is self contained and the
 * logic simple */
use crate::typekinds::{BaseType, BaseTypeKind, TypeKind, VectorType};
use serde::{Deserialize, Serialize};

use std::str::FromStr;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Deserialize, Serialize)]
pub enum SuffixKind {
    Normal,
    Base,
    NoQ,
    NSuffix,
    NoQNSuffix,
    DupNox,
    Dup,
    /* Get the number of lanes or panic if there are not any Lanes */
    Lane,
    Rot270,
    Rot270Lane,
    Rot270LaneQ,
    Rot180,
    Rot180Lane,
    Rot180LaneQ,
    Rot90,
    Rot90Lane,
    Rot90LaneQ,
    /* Force the type to be unsigned */
    Unsigned,
    Tuple,
    NoX,
    BaseByteSize,
    LaneNoX,
    LaneQNoX,
}

pub fn type_to_size(str_type: &str) -> i32 {
    match str_type {
        "int8x8_t" | "int8x16_t" | "i8" | "s8" | "uint8x8_t" | "uint8x16_t" | "u8"
        | "poly8x8_t" | "poly8x16_t" => 8,
        "int16x4_t" | "int16x8_t" | "i16" | "s16" | "uint16x4_t" | "uint16x8_t" | "u16"
        | "float16x4_t" | "float16x8_t" | "_f16" | "poly16x4_t" | "poly16x8_t" => 16,
        "int32x2_t" | "int32x4_t" | "i32" | "s32" | "uint32x2_t" | "uint32x4_t" | "u32"
        | "float32x2_t" | "float32x4_t" | "f32" => 32,
        "int64x1_t" | "int64x2_t" | "i64" | "s64" | "uint64x1_t" | "uint64x2_t" | "u64"
        | "float64x1_t" | "float64x2_t" | "f64" | "poly64x1_t" | "poly64x2_t" | "p64" => 64,
        "p128" => 128,
        _ => panic!("unknown type: {str_type}"),
    }
}

fn neon_get_base_and_char(ty: &VectorType) -> (u32, char, bool) {
    let lanes = ty.lanes();
    match ty.base_type() {
        BaseType::Sized(BaseTypeKind::Float, size) => (*size, 'f', *size * lanes == 128),
        BaseType::Sized(BaseTypeKind::Int, size) => (*size, 's', *size * lanes == 128),
        BaseType::Sized(BaseTypeKind::UInt, size) => (*size, 'u', *size * lanes == 128),
        BaseType::Sized(BaseTypeKind::Poly, size) => (*size, 'p', *size * lanes == 128),
        _ => panic!("Unhandled {:?}", ty),
    }
}

/* @TODO
 * for the chained enum types we can safely delete them as we can index the
 * types array */
pub fn make_neon_suffix(type_kind: TypeKind, suffix_kind: SuffixKind) -> String {
    match type_kind {
        TypeKind::Vector(ty) => {
            let tuple_size = ty.tuple_size().map_or(0, |t| t.to_int());
            let (base_size, prefix_char, requires_q) = neon_get_base_and_char(&ty);
            let lanes = ty.lanes();
            match suffix_kind {
                SuffixKind::Normal => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    if tuple_size > 0 {
                        str_suffix.push_str("_x");
                        str_suffix.push_str(tuple_size.to_string().as_str());
                    }
                    return str_suffix;
                }
                SuffixKind::NSuffix => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push_str("_n_");
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    return str_suffix;
                }

                SuffixKind::NoQ => format!("_{}{}", prefix_char, base_size),
                SuffixKind::NoQNSuffix => format!("_n{}{}", prefix_char, base_size),

                SuffixKind::Unsigned => {
                    let t = type_kind.to_string();
                    if t.starts_with("u") {
                        return t;
                    }
                    return format!("u{}", t);
                }
                SuffixKind::Lane => {
                    if lanes == 0 {
                        panic!("type {} has no lanes!", type_kind.to_string())
                    } else {
                        format!("{}", lanes)
                    }
                }
                SuffixKind::Tuple => {
                    if tuple_size == 0 {
                        panic!("type {} has no lanes!", type_kind.to_string())
                    } else {
                        format!("{}", tuple_size)
                    }
                }
                SuffixKind::Base => base_size.to_string(),
                SuffixKind::NoX => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    return str_suffix;
                }
                SuffixKind::Dup => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push_str("dup_");
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    if tuple_size > 0 {
                        str_suffix.push_str("_x");
                        str_suffix.push_str(tuple_size.to_string().as_str());
                    }
                    return str_suffix;
                }
                SuffixKind::DupNox => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push_str("dup_");
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    return str_suffix;
                }
                SuffixKind::LaneNoX => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push_str("lane_");
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    return str_suffix;
                }
                SuffixKind::LaneQNoX => {
                    let mut str_suffix: String = String::new();
                    if requires_q {
                        str_suffix.push('q');
                    }
                    str_suffix.push('_');
                    str_suffix.push_str("laneq_");
                    str_suffix.push(prefix_char);
                    str_suffix.push_str(base_size.to_string().as_str());
                    return str_suffix;
                }
                SuffixKind::Rot270 => {
                    if requires_q {
                        return format!("q_rot270_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot270_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot270Lane => {
                    if requires_q {
                        return format!("q_rot270_lane_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot270_lane_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot270LaneQ => {
                    if requires_q {
                        return format!("q_rot270_laneq_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot270_laneq_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot180 => {
                    if requires_q {
                        return format!("q_rot180_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot180_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot180Lane => {
                    if requires_q {
                        return format!("q_rot180_lane_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot180_lane_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot180LaneQ => {
                    if requires_q {
                        return format!("q_rot180_laneq_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot180_laneq_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot90 => {
                    if requires_q {
                        return format!("q_rot90_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot90_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot90Lane => {
                    if requires_q {
                        return format!("q_rot90_lane_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot90_lane_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::Rot90LaneQ => {
                    if requires_q {
                        return format!("q_rot90_laneq_{}{}", prefix_char, base_size.to_string());
                    }
                    return format!("_rot90_laneq_{}{}", prefix_char, base_size.to_string());
                }
                SuffixKind::BaseByteSize => format!("{}", base_size / 8),
            }
        }
        _ => panic!("Cannot only make neon vector types suffixed"),
    }
}

impl FromStr for SuffixKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "no" => Ok(SuffixKind::Normal),
            "noq" => Ok(SuffixKind::NoQ),
            "N" => Ok(SuffixKind::NSuffix),
            "noq_N" => Ok(SuffixKind::NoQNSuffix),
            "dup_nox" => Ok(SuffixKind::DupNox),
            "dup" => Ok(SuffixKind::Dup),
            "lane" => Ok(SuffixKind::Lane),
            "base" => Ok(SuffixKind::Base),
            "tuple" => Ok(SuffixKind::Tuple),
            "rot270" => Ok(SuffixKind::Rot270),
            "rot270_lane" => Ok(SuffixKind::Rot270Lane),
            "rot270_laneq" => Ok(SuffixKind::Rot270LaneQ),
            "rot90" => Ok(SuffixKind::Rot90),
            "rot90_lane" => Ok(SuffixKind::Rot90Lane),
            "rot90_laneq" => Ok(SuffixKind::Rot90LaneQ),
            "rot180" => Ok(SuffixKind::Rot180),
            "rot180_lane" => Ok(SuffixKind::Rot180LaneQ),
            "rot180_laneq" => Ok(SuffixKind::Rot180LaneQ),
            "u" => Ok(SuffixKind::Unsigned),
            "nox" => Ok(SuffixKind::NoX),
            "base_byte_size" => Ok(SuffixKind::BaseByteSize),
            "lane_nox" => Ok(SuffixKind::LaneNoX),
            "laneq_nox" => Ok(SuffixKind::LaneQNoX),
            _ => Err(format!("unknown suffix type: {}", s)),
        }
    }
}

impl fmt::Display for SuffixKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuffixKind::Normal => write!(f, "normal"),
            SuffixKind::NoQ => write!(f, "NoQ"),
            SuffixKind::NSuffix => write!(f, "NSuffix"),
            SuffixKind::NoQNSuffix => write!(f, "NoQNSuffix"),
            SuffixKind::DupNox => write!(f, "DupNox"),
            SuffixKind::Dup => write!(f, "Dup",),
            SuffixKind::Lane => write!(f, "Lane"),
            SuffixKind::LaneNoX => write!(f, "LaneNoX"),
            SuffixKind::LaneQNoX => write!(f, "LaneQNoX"),
            SuffixKind::Base => write!(f, "Base"),
            SuffixKind::Rot270 => write!(f, "Rot270",),
            SuffixKind::Rot270Lane => write!(f, "Rot270Lane"),
            SuffixKind::Rot270LaneQ => write!(f, "Rot270LaneQ"),
            SuffixKind::Rot90 => write!(f, "Rot90",),
            SuffixKind::Rot90Lane => write!(f, "Rot90Lane"),
            SuffixKind::Rot90LaneQ => write!(f, "Rot90LaneQ"),
            SuffixKind::Rot180 => write!(f, "Rot180",),
            SuffixKind::Rot180Lane => write!(f, "Rot180Lane"),
            SuffixKind::Rot180LaneQ => write!(f, "Rot180LaneQ"),
            SuffixKind::Unsigned => write!(f, "Unsigned"),
            SuffixKind::Tuple => write!(f, "Tuple"),
            SuffixKind::NoX => write!(f, "NoX"),
            SuffixKind::BaseByteSize => write!(f, "BaseByteSize"),
        }
    }
}
