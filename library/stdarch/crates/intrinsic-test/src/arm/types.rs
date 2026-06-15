use super::intrinsic::ArmType;
use crate::common::PREDICATE_LOCAL;
use crate::common::intrinsic_helpers::{
    IntrinsicType, Sign, SimdLen, TypeDefinition, TypeKind, default_fixed_vector_comparison,
};
use itertools::Itertools;

impl TypeDefinition for ArmType {
    /// Gets a string containing the typename for this type in C format.
    fn c_type(&self) -> String {
        let prefix = self.kind.c_prefix();

        match (self.bit_len, self.simd_len, self.vec_len) {
            // e.g. `bool`
            (Some(_), None, None) if matches!(self.kind, TypeKind::Bool) => {
                format!("{prefix}")
            }
            // e.g. `float32_t`, `int64_t`
            (Some(bit_len), None, None) => format!("{prefix}{bit_len}_t"),
            // e.g. `float32x2_t`, `int64x2_t`
            (Some(bit_len), Some(SimdLen::Fixed(simd)), None) => {
                format!("{prefix}{bit_len}x{simd}_t")
            }
            // e.g. `float32x2x3_t`, `int64x2x3_t`
            (Some(bit_len), Some(SimdLen::Fixed(simd)), Some(vec)) => {
                format!("{prefix}{bit_len}x{simd}x{vec}_t")
            }
            // e.g. `svbool_t`
            (Some(_), Some(SimdLen::Scalable), None) if matches!(self.kind, TypeKind::Bool) => {
                format!("sv{prefix}_t")
            }
            // e.g. `svfloat32_t`, `svint64_t`
            (Some(bit_len), Some(SimdLen::Scalable), None) => format!("sv{prefix}{bit_len}_t"),
            // e.g. `svfloat32x3_t`, `svint64x3_t`
            (Some(bit_len), Some(SimdLen::Scalable), Some(vec)) => {
                format!("sv{prefix}{bit_len}x{vec}_t")
            }
            _ => todo!("{self:#?}"),
        }
    }

    fn rust_type(&self) -> String {
        let rust_prefix = self.kind.rust_prefix();
        let c_prefix = self.kind.c_prefix();

        match (self.bit_len, self.simd_len, self.vec_len) {
            // e.g. `svpattern`
            (None, _, _) => format!("{rust_prefix}"),
            // e.g. `bool`
            (Some(_), None, None) if matches!(self.kind, TypeKind::Bool) => {
                format!("{rust_prefix}")
            }
            // e.g. `i32`
            (Some(bit_len), None, None) => format!("{rust_prefix}{bit_len}"),
            // e.g. `int32x2_t`
            (Some(bit_len), Some(SimdLen::Fixed(simd)), None) => {
                format!("{c_prefix}{bit_len}x{simd}_t")
            }
            // e.g. `int32x2x3_t`
            (Some(bit_len), Some(SimdLen::Fixed(simd)), Some(vec)) => {
                format!("{c_prefix}{bit_len}x{simd}x{vec}_t")
            }
            // e.g. `svbool_t`
            (Some(_), Some(SimdLen::Scalable), None) if matches!(self.kind, TypeKind::Bool) => {
                format!("sv{c_prefix}_t")
            }
            // e.g. `svint32_t`
            (Some(bit_len), Some(SimdLen::Scalable), None) => format!("sv{c_prefix}{bit_len}_t"),
            // e.g. `svint32x3_t`
            (Some(bit_len), Some(SimdLen::Scalable), Some(vec)) => {
                format!("sv{c_prefix}{bit_len}x{vec}_t")
            }
            (Some(_), None, Some(_)) => todo!("{self:#?}"),
        }
    }

    fn rust_scalar_type_for_test_value_array(&self) -> String {
        if self.kind() == TypeKind::Bool && self.num_lanes() == SimdLen::Scalable {
            let mut ty = self.clone();
            ty.kind = TypeKind::Int(Sign::Signed);
            ty.rust_scalar_type()
        } else {
            self.rust_scalar_type()
        }
    }

    /// Determines the load function for this type.
    fn load_function(&self) -> String {
        if let Some(bl) = self.bit_len {
            match self.num_lanes() {
                SimdLen::Scalable => {
                    format!(
                        "svld{len}_{type}{bl}",
                        len = self.num_vectors(),
                        type = self.rust_intrinsic_name_prefix(),
                    )
                }
                SimdLen::Fixed(num_lanes) => {
                    format!(
                        "vld{len}{quad}_{type}{bl}",
                        quad = if num_lanes * bl > 64 { "q" } else { "" },
                        len = self.num_vectors(),
                        type = self.rust_intrinsic_name_prefix(),
                    )
                }
            }
        } else {
            todo!("load_function IntrinsicType: {self:#?}")
        }
    }

    fn comparison_function(&self) -> String {
        if let SimdLen::Fixed(num_lanes) = self.num_lanes() {
            return default_fixed_vector_comparison(self, num_lanes);
        }

        if self.kind() == TypeKind::Bool {
            // There isn't a `svcmpeq` for `svbool_t` and there aren't `svboolxN_t` types, so just
            // do an XOR and test it is empty.
            return format!(
                r#"
let __eq = sveor_b_z({PREDICATE_LOCAL}, __rust_return_value, __c_return_value);
assert!(!svptest_any({PREDICATE_LOCAL}, __eq), "{{}}", id);
                    "#
            );
        }

        // Returns `of` when `num_vectors == 1` otherwise returns the appropriate `svget` invocation
        // for `of`.
        let get = |num_vectors: u32, idx: u32, from: &'static str| -> String {
            if num_vectors == 1 {
                return from.to_string();
            }

            format!(
                "svget{num_vectors}_{ty}{bl}::<{idx}>({from})",
                ty = self.rust_intrinsic_name_prefix(),
                bl = self.inner_size(),
            )
        };

        let n = self.num_vectors();
        (0..n)
            .format_with("\n", |i, fmt| {
                match self.kind() {
                    TypeKind::Float | TypeKind::BFloat => {
                        // Floats need special handling because `NaN != NaN` normally - this
                        // effectively does `(rust == c) || (isnan(rust) && isnan(c))`
                        fmt(&format_args!(
                            r#"
let __rust_eq_return_value = {rust_return_value};
let __c_eq_return_value = {c_return_value};
let __eq_sans_nan = svcmpeq_{ty}{bl}({PREDICATE_LOCAL}, __rust_eq_return_value, __c_eq_return_value);
let __rust_nan = svcmpuo_{ty}{bl}({PREDICATE_LOCAL}, __rust_eq_return_value, __rust_eq_return_value);
let __c_nan = svcmpuo_{ty}{bl}({PREDICATE_LOCAL}, __c_eq_return_value, __c_eq_return_value);
let __both_nan = svand_b_z({PREDICATE_LOCAL}, __rust_nan, __c_nan);
let __eq = svorr_b_z({PREDICATE_LOCAL}, __eq_sans_nan, __both_nan);
if !svptest_any(__pred, __eq) {{
  let __rust_pretty = debug_print_{ty}{bl}(__rust_eq_return_value);
  let __c_pretty = debug_print_{ty}{bl}(__c_eq_return_value);
  panic!("{{}}-{i_plus_one}/{n}\nRust: {{__rust_pretty}}\nC: {{__c_pretty}}", id);
}}
"#,
                            ty = self.rust_intrinsic_name_prefix(),
                            bl = self.inner_size(),
                            rust_return_value = get(n, i, "__rust_return_value"),
                            c_return_value = get(n, i, "__c_return_value"),
                            i_plus_one = i + 1, // so that the output is "1/2" and "2/2"
                        ))
                    }
                    _ => {
                        // Most types can just use `svcmpeq`
                        fmt(&format_args!(
                            r#"
let __rust_eq_return_value = {rust_return_value};
let __c_eq_return_value = {c_return_value};
let __eq = svcmpeq_{ty}{bl}({PREDICATE_LOCAL}, __rust_eq_return_value, __c_eq_return_value);
if !svptest_any(__pred, __eq) {{
  let __rust_pretty = debug_print_{ty}{bl}(__rust_eq_return_value);
  let __c_pretty = debug_print_{ty}{bl}(__c_eq_return_value);
  panic!("{{}}-{i_plus_one}/{n}\nRust: {{__rust_pretty}}\nC: {{__c_pretty}}", id);
}}
"#,
                            ty = self.rust_intrinsic_name_prefix(),
                            bl = self.inner_size(),
                            rust_return_value = get(n, i, "__rust_return_value"),
                            c_return_value = get(n, i, "__c_return_value"),
                            i_plus_one = i + 1, // so that the output is "1/2" and "2/2"
                        ))
                    }
                }
            })
            .to_string()
    }
}

impl ArmType {
    /// Returns the Rust prefix for the name of an intrinsic with this type kind (i.e. `s` for
    /// `i16`, or `u` for `u16`). For type kinds without any bit length at the end (e.g. `bool`),
    /// returns the whole type name.
    pub fn rust_intrinsic_name_prefix(&self) -> &str {
        match self.kind() {
            TypeKind::Char(Sign::Signed) => "s",
            TypeKind::Int(Sign::Signed) => "s",
            TypeKind::Poly => "p",
            TypeKind::Bool => "s",
            _ => self.kind.rust_prefix(),
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
