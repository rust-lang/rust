use super::intrinsic::ArmIntrinsicType;
use crate::common::cli::Language;
use crate::common::indentation::Indentation;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, TypeKind};

impl IntrinsicTypeDefinition for ArmIntrinsicType {
    /// Gets a string containing the typename for this type in C format.
    fn c_type(&self) -> String {
        let prefix = self.kind.c_prefix();
        let const_prefix = if self.constant { "const " } else { "" };

        if let (Some(bit_len), simd_len, vec_len) = (self.bit_len, self.simd_len, self.vec_len) {
            match (simd_len, vec_len) {
                (None, None) => format!("{const_prefix}{prefix}{bit_len}_t"),
                (Some(simd), None) => format!("{prefix}{bit_len}x{simd}_t"),
                (Some(simd), Some(vec)) => format!("{prefix}{bit_len}x{simd}x{vec}_t"),
                (None, Some(_)) => todo!("{:#?}", self), // Likely an invalid case
            }
        } else {
            todo!("{:#?}", self)
        }
    }

    fn c_single_vector_type(&self) -> String {
        if let (Some(bit_len), Some(simd_len)) = (self.bit_len, self.simd_len) {
            format!(
                "{prefix}{bit_len}x{simd_len}_t",
                prefix = self.kind.c_prefix()
            )
        } else {
            unreachable!("Shouldn't be called on this type")
        }
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, language: Language) -> String {
        if let IntrinsicType {
            kind: k,
            bit_len: Some(bl),
            simd_len,
            vec_len,
            ..
        } = &self.data
        {
            let quad = if simd_len.unwrap_or(1) * bl > 64 {
                "q"
            } else {
                ""
            };

            let choose_workaround = language == Language::C && self.target.contains("v7");
            format!(
                "vld{len}{quad}_{type}{size}",
                type = match k {
                    TypeKind::Int(Sign::Unsigned) => "u",
                    TypeKind::Int(Sign::Signed) => "s",
                    TypeKind::Float => "f",
                    // The ACLE doesn't support 64-bit polynomial loads on Armv7
                    // if armv7 and bl == 64, use "s", else "p"
                    TypeKind::Poly => if choose_workaround && *bl == 64 {"s"} else {"p"},
                    x => todo!("get_load_function TypeKind: {:#?}", x),
                },
                size = bl,
                quad = quad,
                len = vec_len.unwrap_or(1),
            )
        } else {
            todo!("get_load_function IntrinsicType: {:#?}", self)
        }
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        if let IntrinsicType {
            kind: k,
            bit_len: Some(bl),
            simd_len,
            ..
        } = &self.data
        {
            let quad = if (simd_len.unwrap_or(1) * bl) > 64 {
                "q"
            } else {
                ""
            };
            format!(
                "vget{quad}_lane_{type}{size}",
                type = match k {
                    TypeKind::Int(Sign::Unsigned) => "u",
                    TypeKind::Int(Sign::Signed) => "s",
                    TypeKind::Float => "f",
                    TypeKind::Poly => "p",
                    x => todo!("get_load_function TypeKind: {:#?}", x),
                },
                size = bl,
                quad = quad,
            )
        } else {
            todo!("get_lane_function IntrinsicType: {:#?}", self)
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
                                    cast = self.c_promotion(),
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
                    format!(
                        "{cast}{lane_fn}(__return_value, {lane})",
                        cast = self.c_promotion(),
                        lane_fn = self.get_lane_function(),
                        lane = idx
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else {
            format!(
                "{promote}cast<{cast}>(__return_value)",
                cast = match self.kind() {
                    TypeKind::Float if self.inner_size() == 16 => "float16_t".to_string(),
                    TypeKind::Float if self.inner_size() == 32 => "float".to_string(),
                    TypeKind::Float if self.inner_size() == 64 => "double".to_string(),
                    TypeKind::Int(Sign::Signed) => format!("int{}_t", self.inner_size()),
                    TypeKind::Int(Sign::Unsigned) => format!("uint{}_t", self.inner_size()),
                    TypeKind::Poly => format!("poly{}_t", self.inner_size()),
                    ty => todo!("print_result_c - Unknown type: {:#?}", ty),
                },
                promote = self.c_promotion(),
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
}

impl ArmIntrinsicType {
    pub fn from_c(s: &str, target: &str) -> Result<Self, String> {
        const CONST_STR: &str = "const";
        if let Some(s) = s.strip_suffix('*') {
            let (s, constant) = match s.trim().strip_suffix(CONST_STR) {
                Some(stripped) => (stripped, true),
                None => (s, false),
            };
            let s = s.trim_end();
            let temp_return = ArmIntrinsicType::from_c(s, target);
            temp_return.map(|mut op| {
                op.ptr = true;
                op.ptr_constant = constant;
                op
            })
        } else {
            // [const ]TYPE[{bitlen}[x{simdlen}[x{vec_len}]]][_t]
            let (mut s, constant) = match s.strip_prefix(CONST_STR) {
                Some(stripped) => (stripped.trim(), true),
                None => (s, false),
            };
            s = s.strip_suffix("_t").unwrap_or(s);
            let mut parts = s.split('x'); // [[{bitlen}], [{simdlen}], [{vec_len}] ]
            let start = parts.next().ok_or("Impossible to parse type")?;
            if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
                let (arg_kind, bit_len) = start.split_at(digit_start);
                let arg_kind = arg_kind.parse::<TypeKind>()?;
                let bit_len = bit_len.parse::<u32>().map_err(|err| err.to_string())?;
                let simd_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse simd_len: {part}")?,
                    ),
                    None => None,
                };
                let vec_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse vec_len: {part}")?,
                    ),
                    None => None,
                };
                Ok(ArmIntrinsicType {
                    data: IntrinsicType {
                        ptr: false,
                        ptr_constant: false,
                        constant,
                        kind: arg_kind,
                        bit_len: Some(bit_len),
                        simd_len,
                        vec_len,
                    },
                    target: target.to_string(),
                })
            } else {
                let kind = start.parse::<TypeKind>()?;
                let bit_len = match kind {
                    TypeKind::Int(_) => Some(32),
                    _ => None,
                };
                Ok(ArmIntrinsicType {
                    data: IntrinsicType {
                        ptr: false,
                        ptr_constant: false,
                        constant,
                        kind: start.parse::<TypeKind>()?,
                        bit_len,
                        simd_len: None,
                        vec_len: None,
                    },
                    target: target.to_string(),
                })
            }
        }
    }
}
