use super::intrinsic::LoongArchType;
use crate::common::intrinsic_helpers::{IntrinsicType, Sign, SimdLen, TypeDefinition, TypeKind};

impl TypeDefinition for LoongArchType {
    fn c_type(&self) -> String {
        if self.ptr {
            return if self.ptr_constant {
                "const void*".to_string()
            } else {
                "void*".to_string()
            };
        }

        match (self.kind(), self.simd_len) {
            (_, Some(SimdLen::Fixed(lanes))) => {
                format!(
                    "__{}",
                    vector_type_name(lanes, self.inner_size(), self.kind())
                )
            }
            (TypeKind::Int(Sign::Signed), None) => {
                scalar_type_name(true, scalar_signature_bits(self.inner_size())).to_string()
            }
            (TypeKind::Int(Sign::Unsigned), None) => {
                scalar_type_name(false, scalar_signature_bits(self.inner_size())).to_string()
            }
            (TypeKind::Float, None) => match self.inner_size() {
                32 => "float".to_string(),
                64 => "double".to_string(),
                bits => unreachable!("unsupported scalar float width {bits}"),
            },
            (TypeKind::Void, None) => "void".to_string(),
            _ => unreachable!("unsupported LoongArch type {self:#?}"),
        }
    }

    fn rust_type(&self) -> String {
        if self.ptr {
            return format!(
                "*{} core::ffi::c_void",
                if self.ptr_constant { "const" } else { "mut" },
            );
        }

        match (self.kind(), self.simd_len) {
            (_, Some(SimdLen::Fixed(lanes))) => {
                vector_type_name(lanes, self.inner_size(), self.kind()).to_string()
            }
            (TypeKind::Int(Sign::Signed), None) => {
                semantic_rust_scalar_type(true, scalar_signature_bits(self.inner_size()))
                    .to_string()
            }
            (TypeKind::Int(Sign::Unsigned), None) => {
                semantic_rust_scalar_type(false, scalar_signature_bits(self.inner_size()))
                    .to_string()
            }
            (TypeKind::Float, None) => match self.inner_size() {
                32 => "f32".to_string(),
                64 => "f64".to_string(),
                bits => unreachable!("unsupported scalar float width {bits}"),
            },
            (TypeKind::Void, None) => "()".to_string(),
            _ => unreachable!("unsupported LoongArch type {self:#?}"),
        }
    }

    fn rust_scalar_type(&self) -> String {
        match self.kind() {
            TypeKind::Int(Sign::Signed) => {
                semantic_rust_scalar_type(true, self.inner_size()).to_string()
            }
            TypeKind::Int(Sign::Unsigned) => {
                semantic_rust_scalar_type(false, self.inner_size()).to_string()
            }
            TypeKind::Float => match self.inner_size() {
                32 => "f32".to_string(),
                64 => "f64".to_string(),
                bits => unreachable!("unsupported scalar float width {bits}"),
            },
            _ => unreachable!("unsupported LoongArch scalar type {self:#?}"),
        }
    }

    fn load_function(&self) -> String {
        let Some(SimdLen::Fixed(lanes)) = self.simd_len else {
            unreachable!("LoongArch loads are only used for SIMD types")
        };

        match (lanes * self.inner_size(), self.kind()) {
            (128, TypeKind::Float) if self.inner_size() == 64 => "lsx_vld_to_m128d".to_string(),
            (128, TypeKind::Float) => "lsx_vld_to_m128".to_string(),
            (128, _) => "lsx_vld_to_m128i".to_string(),
            (256, TypeKind::Float) if self.inner_size() == 64 => "lasx_xvld_to_m256d".to_string(),
            (256, TypeKind::Float) => "lasx_xvld_to_m256".to_string(),
            (256, _) => "lasx_xvld_to_m256i".to_string(),
            bits => unreachable!("unsupported LoongArch vector width {bits:?}"),
        }
    }
}

fn vector_type_name(lanes: u32, bit_len: u32, kind: TypeKind) -> &'static str {
    match (lanes * bit_len, kind, bit_len) {
        (128, TypeKind::Float, 32) => "m128",
        (128, TypeKind::Float, 64) => "m128d",
        (128, _, _) => "m128i",
        (256, TypeKind::Float, 32) => "m256",
        (256, TypeKind::Float, 64) => "m256d",
        (256, _, _) => "m256i",
        _ => unreachable!("unsupported LoongArch vector shape {kind:?}x{bit_len}x{lanes}"),
    }
}

fn scalar_type_name(signed: bool, bit_len: u32) -> &'static str {
    match (signed, bit_len) {
        (true, 32) => "int32_t",
        (true, 64) => "int64_t",
        (false, 32) => "uint32_t",
        (false, 64) => "uint64_t",
        _ => unreachable!("unsupported LoongArch scalar width {bit_len}"),
    }
}

fn scalar_signature_bits(bit_len: u32) -> u32 {
    match bit_len {
        8 | 16 | 32 => 32,
        64 => 64,
        _ => unreachable!("unsupported LoongArch scalar width {bit_len}"),
    }
}

fn semantic_rust_scalar_type(signed: bool, bit_len: u32) -> &'static str {
    match (signed, bit_len) {
        (true, 8) => "i8",
        (true, 16) => "i16",
        (true, 32) => "i32",
        (true, 64) => "i64",
        (false, 8) => "u8",
        (false, 16) => "u16",
        (false, 32) => "u32",
        (false, 64) => "u64",
        _ => unreachable!("unsupported LoongArch scalar width {bit_len}"),
    }
}

pub fn parse_intrinsic_type(s: &str) -> Result<IntrinsicType, String> {
    let (kind, bit_len, simd_len, ptr, ptr_constant) = match s {
        "V16QI" => (
            TypeKind::Int(Sign::Signed),
            Some(8),
            Some(SimdLen::Fixed(16)),
            false,
            false,
        ),
        "V32QI" => (
            TypeKind::Int(Sign::Signed),
            Some(8),
            Some(SimdLen::Fixed(32)),
            false,
            false,
        ),
        "V8HI" => (
            TypeKind::Int(Sign::Signed),
            Some(16),
            Some(SimdLen::Fixed(8)),
            false,
            false,
        ),
        "V16HI" => (
            TypeKind::Int(Sign::Signed),
            Some(16),
            Some(SimdLen::Fixed(16)),
            false,
            false,
        ),
        "V4SI" => (
            TypeKind::Int(Sign::Signed),
            Some(32),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "V8SI" => (
            TypeKind::Int(Sign::Signed),
            Some(32),
            Some(SimdLen::Fixed(8)),
            false,
            false,
        ),
        "V2DI" => (
            TypeKind::Int(Sign::Signed),
            Some(64),
            Some(SimdLen::Fixed(2)),
            false,
            false,
        ),
        "V4DI" => (
            TypeKind::Int(Sign::Signed),
            Some(64),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "UV16QI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(8),
            Some(SimdLen::Fixed(16)),
            false,
            false,
        ),
        "UV32QI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(8),
            Some(SimdLen::Fixed(32)),
            false,
            false,
        ),
        "UV8HI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(16),
            Some(SimdLen::Fixed(8)),
            false,
            false,
        ),
        "UV16HI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(16),
            Some(SimdLen::Fixed(16)),
            false,
            false,
        ),
        "UV4SI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(32),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "UV8SI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(32),
            Some(SimdLen::Fixed(8)),
            false,
            false,
        ),
        "UV2DI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(64),
            Some(SimdLen::Fixed(2)),
            false,
            false,
        ),
        "UV4DI" => (
            TypeKind::Int(Sign::Unsigned),
            Some(64),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "V4SF" => (
            TypeKind::Float,
            Some(32),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "V8SF" => (
            TypeKind::Float,
            Some(32),
            Some(SimdLen::Fixed(8)),
            false,
            false,
        ),
        "V2DF" => (
            TypeKind::Float,
            Some(64),
            Some(SimdLen::Fixed(2)),
            false,
            false,
        ),
        "V4DF" => (
            TypeKind::Float,
            Some(64),
            Some(SimdLen::Fixed(4)),
            false,
            false,
        ),
        "QI" => (TypeKind::Int(Sign::Signed), Some(8), None, false, false),
        "HI" => (TypeKind::Int(Sign::Signed), Some(16), None, false, false),
        "SI" => (TypeKind::Int(Sign::Signed), Some(32), None, false, false),
        "UQI" => (TypeKind::Int(Sign::Unsigned), Some(8), None, false, false),
        "UHI" => (TypeKind::Int(Sign::Unsigned), Some(16), None, false, false),
        "USI" => (TypeKind::Int(Sign::Unsigned), Some(32), None, false, false),
        "DI" => (TypeKind::Int(Sign::Signed), Some(64), None, false, false),
        "UDI" => (TypeKind::Int(Sign::Unsigned), Some(64), None, false, false),
        "CVPOINTER" => (TypeKind::Int(Sign::Signed), Some(8), None, true, true),
        "VOID" => (TypeKind::Void, None, None, false, false),
        _ => return Err(format!("unsupported LoongArch type {s}")),
    };

    Ok(IntrinsicType {
        constant: false,
        ptr_constant,
        ptr,
        kind,
        bit_len,
        simd_len,
        vec_len: None,
    })
}
