use std::str::FromStr;

use itertools::Itertools;

use super::intrinsic::X86IntrinsicType;
use crate::common::intrinsic_helpers::{
    IntrinsicType, IntrinsicTypeDefinition, Sign, SimdLen, TypeKind,
};
use crate::x86::xml_parser::Parameter;

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the type in C format.
    /// This function assumes that this value is present in the metadata hashmap.
    fn c_type(&self) -> String {
        self.param
            .type_data
            .replace("unsigned __int64", "uint64_t")
            .replace("unsigned __int32", "uint32_t")
            .replace("unsigned __int16", "uint16_t")
            .replace("unsigned __int8", "uint8_t")
            .replace("__int64", "int64_t")
            .replace("__int32", "int32_t")
            .replace("__int16", "int16_t")
            .replace("__int8", "int8_t")
            .replace("const ", "")
    }

    fn rust_type(&self) -> String {
        let type_data = &*self.param.type_data;
        if type_data.starts_with("__m") {
            return type_data.to_owned();
        }
        match &*type_data.replace("const ", "") {
            "_Float16" => "f16",
            "__bfloat16" => "bf16",
            "float" => "f32",
            "double" => "f64",
            "__int8" | "char" => "i8",
            "unsigned char" => "u8",
            "__int16" | "short" => "i16",
            "unsigned short" => "u16",
            "__int32" | "int" => "i32",
            "unsigned __int32" | "unsigned int" | "unsigned long" => "u32",
            "__int64" | "long long" => "i64",
            "unsigned __int64" => "u64",
            "size_t" => "usize",
            _ => todo!("unknown type {type_data}"),
        }
        .to_string()
    }

    /// Determines the load function for this type.
    fn get_load_function(&self) -> String {
        let type_value = self.param.type_data.clone();
        if type_value.len() == 0 {
            unimplemented!("the value for key 'type' is not present!");
        }
        if type_value.starts_with("__mmask") {
            // no need of loads, since they work directly
            // with hex constants
            String::from("*")
        } else if type_value.starts_with("__m") {
            // the structure is like the follows:
            // if "type" starts with __m<num>{h/i/<null>},
            // then use either _mm_set1_epi64,
            // _mm256_set1_epi64 or _mm512_set1_epi64

            let type_val_filtered = type_value
                .chars()
                .filter(|c| c.is_numeric())
                .join("")
                .replace("128", "")
                .replace("64", "");
            {
                let suffix = match (self.bit_len, self.kind) {
                    (Some(16), TypeKind::Float)
                        if ["__m128i", "__m256i", "__m512i"]
                            .contains(&self.param.type_data.as_str()) =>
                    {
                        format!("ph_to_{}", self.param.type_data)
                    }
                    (Some(32), TypeKind::Float)
                        if ["__m128h", "__m256h", "__m512h"]
                            .contains(&self.param.type_data.as_str()) =>
                    {
                        format!("ps_to_{}", self.param.type_data)
                    }
                    (Some(bit_len @ (16 | 32 | 64)), TypeKind::Int(_) | TypeKind::Mask)
                        if ["__m128d", "__m256d", "__m512d"]
                            .contains(&self.param.type_data.as_str()) =>
                    {
                        format!("epi{bit_len}_to_{}", self.param.type_data)
                    }
                    (Some(bit_len @ (16 | 32 | 64)), TypeKind::Int(_) | TypeKind::Mask)
                        if ["__m128", "__m256", "__m512"]
                            .contains(&self.param.type_data.as_str()) =>
                    {
                        format!("epi{bit_len}_to_{}", self.param.type_data)
                    }
                    (Some(bit_len @ (8 | 16 | 32 | 64)), TypeKind::Int(_)) => {
                        format!("epi{bit_len}")
                    }
                    (Some(bit_len), TypeKind::Mask) => format!("epi{bit_len}"),
                    (Some(16), TypeKind::Float) => format!("ph"),
                    (Some(32), TypeKind::Float) => format!("ps"),
                    (Some(64), TypeKind::Float) => format!("pd"),
                    (Some(128 | 256 | 512), TypeKind::Vector) => format!("epi32"),
                    _ => unreachable!("Invalid element type for a vector type! {:?}", self.param),
                };
                format!("_mm{type_val_filtered}_loadu_{suffix}")
            }
        } else {
            // if it is a pointer, then rely on type conversion
            // If it is not any of the above type (__int<num>, __bfloat16, unsigned short, etc)
            // then typecast it.
            format!("({type_value})")
        }
    }

    fn rust_scalar_type(&self) -> String {
        if self.is_simd() {
            format!(
                "{prefix}{bits}",
                prefix = self.kind().rust_prefix(),
                bits = self.inner_size()
            )
        } else {
            self.rust_type().replace("__mmask", "u")
        }
    }
}

impl X86IntrinsicType {
    fn from_c(s: &str) -> Result<IntrinsicType, String> {
        let mut s_copy = s.to_string();
        s_copy = s_copy
            .replace("*", "")
            .replace("_", "")
            .replace("constexpr", "")
            .replace("const", "")
            .replace("literal", "");

        let s_split = s_copy
            .split(" ")
            .filter_map(|s| if s.len() == 0 { None } else { Some(s) })
            .last();

        let s_split = s_split.map(|s| s.chars().filter(|c| !c.is_numeric()).join(""));

        // TODO: make the unwrapping safe
        let kind = TypeKind::from_str(s_split.unwrap().trim()).unwrap_or(TypeKind::Void);

        let kind = if s.find("unsigned").is_some() {
            match kind {
                TypeKind::Int(_) => TypeKind::Int(Sign::Unsigned),
                TypeKind::Char(_) => TypeKind::Char(Sign::Unsigned),
                a => a,
            }
        } else {
            kind
        };

        let ptr_constant = false;
        let constant = s.matches("const").next().is_some();
        let ptr = s.matches("*").next().is_some();

        Ok(IntrinsicType {
            ptr,
            ptr_constant,
            constant,
            kind,
            bit_len: None,
            simd_len: None,
            vec_len: None,
        })
    }

    pub fn update_simd_len(&mut self) {
        let mut type_processed = self.param.type_data.clone();
        type_processed.retain(|c| c.is_numeric());

        // check the param.type and extract numeric part if there are double
        // underscores. divide this number with bit-len and set this as simd-len.
        // Only __m<int> types can have a simd-len.
        if self.param.type_data.contains("__m") && !self.param.type_data.contains("__mmask") {
            self.data.simd_len = match str::parse::<u32>(type_processed.as_str()) {
                // If bit_len is None, simd_len will be None.
                // Else simd_len will be (num_bits / bit_len).
                Ok(num_bits) => self
                    .data
                    .bit_len
                    .and_then(|bit_len| Some(SimdLen::Fixed(num_bits / bit_len))),
                Err(_) => None,
            };
        }
    }

    pub fn from_param(param: &Parameter) -> Result<Self, String> {
        match Self::from_c(param.type_data.as_str()) {
            Err(message) => Err(message),
            Ok(mut data) => {
                // First correct the type of the parameter using param.etype.
                // The assumption is that the parameter of type void may have param.type
                // as "__m128i", "__mmask8" and the like.
                if !param.etype.is_empty() {
                    match TypeKind::from_str(param.etype.as_str()) {
                        Ok(value) => {
                            data.kind = value;
                        }
                        Err(_) => {}
                    };
                }

                // check for param.etype.
                // extract the numeric part and set as bit-len
                // If param.etype is not present, guess the default bit-len

                let mut etype_processed = param.etype.clone();
                etype_processed.retain(|c| c.is_numeric());

                let mut type_processed = param.type_data.clone();
                type_processed.retain(|c| c.is_numeric());

                match str::parse::<u32>(etype_processed.as_str()) {
                    Ok(value) => data.bit_len = Some(value),
                    Err(_) => {
                        data.bit_len = match data.kind() {
                            TypeKind::Char(_) => Some(8),
                            TypeKind::BFloat => Some(16),
                            TypeKind::Int(_) => Some(32),
                            TypeKind::Float => Some(32),
                            _ => None,
                        };
                    }
                }

                if param.type_data.contains("__mmask") {
                    data.bit_len = str::parse::<u32>(type_processed.as_str()).ok();
                }

                if vec!["M512", "M256", "M128"].contains(&param.etype.as_str()) {
                    match param.type_data.chars().last() {
                        Some('i') => {
                            data.kind = TypeKind::Int(Sign::Signed);
                            data.bit_len = Some(32);
                        }
                        Some('h') => {
                            data.kind = TypeKind::Float;
                            data.bit_len = Some(16);
                        }
                        Some('d') => {
                            data.kind = TypeKind::Float;
                            data.bit_len = Some(64);
                        }
                        _ => (),
                    }
                }

                // default settings for "void *" parameters
                // often used by intrinsics to denote memory address or so.
                if data.kind == TypeKind::Void && data.ptr {
                    data.kind = TypeKind::Int(Sign::Unsigned);
                    data.bit_len = Some(8);
                }

                // default settings for "void *" parameters
                // often used by intrinsics to denote memory address or so.
                if data.kind == TypeKind::Mask && data.bit_len.is_none() {
                    data.bit_len = Some(32);
                }

                if param.etype == "IMM" || param.imm_width > 0 || param.imm_type.len() > 0 {
                    data.kind = TypeKind::Int(Sign::Unsigned);
                    data.constant = true;
                }

                // Rust defaults to signed variants, unless they are explicitly mentioned
                // the `type` field are C++ types.
                if data.kind == TypeKind::Int(Sign::Unsigned)
                    && !(param.type_data.contains("unsigned") || param.type_data.contains("uint"))
                {
                    data.kind = TypeKind::Int(Sign::Signed)
                }

                // default settings for IMM parameters
                if param.etype == "IMM" {
                    data.bit_len = if param.imm_width > 0 {
                        Some(param.imm_width)
                    } else {
                        Some(8)
                    }
                }

                // a few intrinsics have wrong `etype` field in the XML
                // - _mm512_reduce_add_ph
                // - _mm512_reduce_mul_ph
                // - _mm512_reduce_min_ph
                // - _mm512_reduce_max_ph
                // - _mm512_conj_pch
                if param.type_data == "__m512h" && param.etype == "FP32" {
                    data.bit_len = Some(16);
                    data.simd_len = Some(SimdLen::Fixed(32));
                }

                let mut result = X86IntrinsicType {
                    data,
                    param: param.clone(),
                };

                result.update_simd_len();
                Ok(result)
            }
        }
        // Tile types won't currently reach here, since the intrinsic that involve them
        // often return "null" type. Such intrinsics are not tested in `intrinsic-test`
        // currently and are filtered out at `mod.rs`.
    }
}
