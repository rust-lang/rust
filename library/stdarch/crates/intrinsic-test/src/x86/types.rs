use std::str::FromStr;

use itertools::Itertools;
use regex::Regex;

use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, TypeKind};
use crate::x86::xml_parser::Parameter;

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the type in C format.
    /// This function assumes that this value is present in the metadata hashmap.
    fn c_type(&self) -> String {
        self.param.type_data.clone()
    }

    fn c_single_vector_type(&self) -> String {
        // matches __m128, __m256 and similar types
        let re = Regex::new(r"__m\d+").unwrap();
        if re.is_match(self.param.type_data.as_str()) {
            self.param.type_data.clone()
        } else {
            unreachable!("Shouldn't be called on this type")
        }
    }

    // fn rust_type(&self) -> String {
    //     // handling edge cases first
    //     // the general handling is implemented below
    //     if let Some(val) = self.metadata.get("type") {
    //         match val.as_str() {
    //             "__m128 const *" => {
    //                 return "&__m128".to_string();
    //             }
    //             "__m128d const *" => {
    //                 return "&__m128d".to_string();
    //             }
    //             "const void*" => {
    //                 return "&__m128d".to_string();
    //             }
    //             _ => {}
    //         }
    //     }

    //     if self.kind() == TypeKind::Void && self.ptr {
    //         // this has been handled by default settings in
    //         // the from_param function of X86IntrinsicType
    //         unreachable!()
    //     }

    //     // general handling cases
    //     let core_part = if self.kind() == TypeKind::Mask {
    //         // all types of __mmask<int> are handled here
    //         format!("__mask{}", self.bit_len.unwrap())
    //     } else if self.simd_len.is_some() {
    //         // all types of __m<int> vector types are handled here
    //         let re = Regex::new(r"\__m\d+[a-z]*").unwrap();
    //         let rust_type = self
    //             .metadata
    //             .get("type")
    //             .map(|val| re.find(val).unwrap().as_str());
    //         rust_type.unwrap().to_string()
    //     } else {
    //         format!(
    //             "{}{}",
    //             self.kind.rust_prefix().to_string(),
    //             self.bit_len.unwrap()
    //         )
    //     };

    //     // extracting "memsize" so that even vector types can be involved
    //     let memwidth = self
    //         .metadata
    //         .get("memwidth")
    //         .map(|n| str::parse::<u32>(n).unwrap());
    //     let prefix_part = if self.ptr && self.constant && self.bit_len.eq(&memwidth) {
    //         "&"
    //     } else if self.ptr && self.bit_len.eq(&memwidth) {
    //         "&mut "
    //     } else if self.ptr && self.constant {
    //         "*const "
    //     } else if self.ptr {
    //         "*mut "
    //     } else {
    //         ""
    //     };

    //     return prefix_part.to_string() + core_part.as_str();
    // }

    /// Determines the load function for this type.
    fn get_load_function(&self, _language: Language) -> String {
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
                .replace("128", "");
            format!("_mm{type_val_filtered}_set1_epi64")
        } else {
            // if it is a pointer, then rely on type conversion
            // If it is not any of the above type (__int<num>, __bfloat16, unsigned short, etc)
            // then typecast it.
            format!("({type_value})")
        }
        // Look for edge cases (constexpr, literal, etc)
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        let total_vector_bits: Option<u32> = self
            .vec_len
            .zip(self.bit_len)
            .and_then(|(vec_len, bit_len)| Some(vec_len * bit_len));

        match (self.bit_len, total_vector_bits) {
            (Some(8), Some(128)) => String::from("_mm_extract_epi8"),
            (Some(16), Some(128)) => String::from("_mm_extract_epi16"),
            (Some(32), Some(128)) => String::from("_mm_extract_epi32"),
            (Some(64), Some(128)) => String::from("_mm_extract_epi64"),
            (Some(8), Some(256)) => String::from("_mm256_extract_epi8"),
            (Some(16), Some(256)) => String::from("_mm256_extract_epi16"),
            (Some(32), Some(256)) => String::from("_mm256_extract_epi32"),
            (Some(64), Some(256)) => String::from("_mm256_extract_epi64"),
            (Some(8), Some(512)) => String::from("_mm512_extract_intrinsic_test_epi8"),
            (Some(16), Some(512)) => String::from("_mm512_extract_intrinsic_test_epi16"),
            (Some(32), Some(512)) => String::from("_mm512_extract_intrinsic_test_epi32"),
            (Some(64), Some(512)) => String::from("_mm512_extract_intrinsic_test_epi64"),
            _ => unreachable!(
                "invalid length for vector argument: {:?}, {:?}",
                self.bit_len, self.vec_len
            ),
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

                if param.type_data.matches("__mmask").next().is_some() {
                    data.bit_len = str::parse::<u32>(type_processed.as_str()).ok();
                }

                // then check the param.type and extract numeric part if there are double
                // underscores. divide this number with bit-len and set this as simd-len.
                // Only __m<int> types can have a simd-len.
                if param.type_data.matches("__m").next().is_some()
                    && param.type_data.matches("__mmask").next().is_none()
                {
                    data.vec_len = match str::parse::<u32>(type_processed.as_str()) {
                        // If bit_len is None, vec_len will be None.
                        // Else vec_len will be (num_bits / bit_len).
                        Ok(num_bits) => data.bit_len.and_then(|bit_len| Some(num_bits / bit_len)),
                        Err(_) => None,
                    };
                }

                // default settings for "void *" parameters
                // often used by intrinsics to denote memory address or so.
                if data.kind == TypeKind::Void && data.ptr {
                    data.kind = TypeKind::Int(Sign::Unsigned);
                    data.bit_len = Some(8);
                }

                // if param.etype == IMM, then it is a constant.
                // else it stays unchanged.
                data.constant |= param.etype == "IMM";
                Ok(X86IntrinsicType {
                    data,
                    param: param.clone(),
                })
            }
        }
        // Tile types won't currently reach here, since the intrinsic that involve them
        // often return "null" type. Such intrinsics are not tested in `intrinsic-test`
        // currently and are filtered out at `mod.rs`.
    }
}
