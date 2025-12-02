use std::str::FromStr;

use itertools::Itertools;
use regex::Regex;

use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::indentation::Indentation;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, TypeKind};
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
            if type_value.contains("__m64") {
                return String::from("*(__m64*)");
            }

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

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, indentation: Indentation, additional: &str) -> String {
        let lanes = if self.num_vectors() > 1 {
            (0..self.num_vectors())
                .map(|vector| {
                    format!(
                        r#""{ty}(" << {lanes} << ")""#,
                        ty = self.c_single_vector_type(),
                        lanes = (0..self.num_lanes())
                            .map(move |idx| -> std::string::String {
                                format!(
                                    "{cast}{lane_fn}(__return_value.val[{vector}], {lane})",
                                    cast = self.generate_final_type_cast(),
                                    lane_fn = self.get_lane_function(),
                                    lane = idx,
                                    vector = vector,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(r#" << ", " << "#)
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else if self.num_lanes() > 1 {
            (0..self.num_lanes())
                .map(|idx| -> std::string::String {
                    let cast_type = self.c_promotion();
                    let lane_fn = self.get_lane_function();
                    if cast_type.len() > 2 {
                        format!("cast<{cast_type}>({lane_fn}(__return_value, {idx}))")
                    } else {
                        format!("{lane_fn}(__return_value, {idx})")
                    }
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else {
            format!(
                "{promote}cast<{cast}>(__return_value)",
                cast = match self.kind() {
                    TypeKind::Void => "void".to_string(),
                    TypeKind::Float if self.inner_size() == 64 => "double".to_string(),
                    TypeKind::Float if self.inner_size() == 32 => "float".to_string(),
                    TypeKind::Mask => format!(
                        "__mmask{}",
                        self.bit_len.expect(format!("self: {self:#?}").as_str())
                    ),
                    TypeKind::Vector => format!(
                        "__m{}i",
                        self.bit_len.expect(format!("self: {self:#?}").as_str())
                    ),
                    _ => self.c_scalar_type(),
                },
                promote = self.generate_final_type_cast(),
            )
        };

        format!(
            r#"{indentation}std::cout << "Result {additional}-" << i+1 << ": {ty}" << std::fixed << std::setprecision(150) <<  {lanes} << "{close}" << std::endl;"#,
            ty = if self.is_simd() {
                format!("{}(", self.c_type())
            } else {
                String::from("")
            },
            close = if self.is_simd() { ")" } else { "" },
        )
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        let total_vector_bits: Option<u32> = self
            .simd_len
            .zip(self.bit_len)
            .and_then(|(simd_len, bit_len)| Some(simd_len * bit_len));

        match (self.bit_len, total_vector_bits) {
            (Some(8), Some(128)) => String::from("(uint8_t)_mm_extract_epi8"),
            (Some(16), Some(128)) => String::from("(uint16_t)_mm_extract_epi16"),
            (Some(32), Some(128)) => String::from("(uint32_t)_mm_extract_epi32"),
            (Some(64), Some(128)) => String::from("(uint64_t)_mm_extract_epi64"),
            (Some(8), Some(256)) => String::from("(uint8_t)_mm256_extract_epi8"),
            (Some(16), Some(256)) => String::from("(uint16_t)_mm256_extract_epi16"),
            (Some(32), Some(256)) => String::from("(uint32_t)_mm256_extract_epi32"),
            (Some(64), Some(256)) => String::from("(uint64_t)_mm256_extract_epi64"),
            (Some(8), Some(512)) => String::from("(uint8_t)_mm512_extract_intrinsic_test_epi8"),
            (Some(16), Some(512)) => String::from("(uint16_t)_mm512_extract_intrinsic_test_epi16"),
            (Some(32), Some(512)) => String::from("(uint32_t)_mm512_extract_intrinsic_test_epi32"),
            (Some(64), Some(512)) => String::from("(uint64_t)_mm512_extract_intrinsic_test_epi64"),
            (Some(8), Some(64)) => String::from("(uint8_t)_mm64_extract_intrinsic_test_epi8"),
            (Some(16), Some(64)) => String::from("(uint16_t)_mm_extract_pi16"),
            (Some(32), Some(64)) => String::from("(uint32_t)_mm64_extract_intrinsic_test_epi32"),
            _ => unreachable!(
                "invalid length for vector argument: {:?}, {:?}",
                self.bit_len, self.simd_len
            ),
        }
    }

    fn rust_scalar_type(&self) -> String {
        let prefix = match self.data.kind {
            TypeKind::Mask => String::from("__mmask"),
            TypeKind::Vector => String::from("i"),
            _ => self.kind().rust_prefix().to_string(),
        };

        let bits = if self.inner_size() >= 128 {
            32
        } else {
            self.inner_size()
        };
        format!("{prefix}{bits}")
    }

    fn print_result_rust(&self) -> String {
        let return_value = match self.kind() {
            TypeKind::Float if self.inner_size() == 16 => "debug_f16(__return_value)".to_string(),
            TypeKind::Float
                if self.inner_size() == 32
                    && ["__m512h"].contains(&self.param.type_data.as_str()) =>
            {
                "debug_as::<_, f32>(__return_value)".to_string()
            }
            TypeKind::Int(_)
                if ["__m128i", "__m256i", "__m512i"].contains(&self.param.type_data.as_str()) =>
            {
                format!("debug_as::<_, u{}>(__return_value)", self.inner_size())
            }
            _ => "format_args!(\"{__return_value:.150?}\")".to_string(),
        };

        return_value
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
                    .and_then(|bit_len| Some(num_bits / bit_len)),
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
