//! Hexagon HVX Code Generator
//!
//! This generator creates v64.rs and v128.rs from scratch using the LLVM HVX
//! header file as the sole source of truth. It parses the C intrinsic prototypes
//! and generates Rust wrapper functions with appropriate attributes.
//!
//! The two generated files provide:
//! - v64.rs: 64-byte vector mode intrinsics (512-bit vectors)
//! - v128.rs: 128-byte vector mode intrinsics (1024-bit vectors)
//!
//! Both modules are available unconditionally, but require the appropriate
//! target features to actually use the intrinsics.
//!
//! Usage:
//!     cd crates/stdarch-gen-hexagon
//!     cargo run
//!     # Output is written to ../core_arch/src/hexagon/v64.rs and v128.rs

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Mappings from HVX intrinsics to architecture-independent SIMD intrinsics.
/// These intrinsics have equivalent semantics and can be lowered to the generic form.
fn get_simd_intrinsic_mappings() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    // Bitwise operations (element-size independent)
    map.insert("vxor", "simd_xor");
    map.insert("vand", "simd_and");
    map.insert("vor", "simd_or");
    // Word (32-bit) arithmetic operations
    map.insert("vaddw", "simd_add");
    map.insert("vsubw", "simd_sub");
    map
}

/// The tracking issue number for the stdarch_hexagon feature
const TRACKING_ISSUE: &str = "151523";

/// HVX vector length mode
#[derive(Debug, Clone, Copy, PartialEq)]
enum VectorMode {
    /// 64-byte vectors (512 bits)
    V64,
    /// 128-byte vectors (1024 bits)
    V128,
}

impl VectorMode {
    fn bytes(&self) -> u32 {
        match self {
            VectorMode::V64 => 64,
            VectorMode::V128 => 128,
        }
    }

    fn bits(&self) -> u32 {
        self.bytes() * 8
    }

    fn lanes(&self) -> u32 {
        self.bytes() / 4 // 32-bit lanes
    }

    fn target_feature(&self) -> &'static str {
        match self {
            VectorMode::V64 => "hvx-length64b",
            VectorMode::V128 => "hvx-length128b",
        }
    }
}

/// LLVM version the header file is from (for reference)
/// Source: https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0-rc1/clang/lib/Headers/hvx_hexagon_protos.h
const LLVM_VERSION: &str = "22.1.0-rc1";

/// Maximum HVX architecture version supported by rustc
/// Check with: rustc --target=hexagon-unknown-linux-musl --print target-features
const MAX_SUPPORTED_ARCH: u32 = 79;

/// Local header file path (checked into the repository)
const HEADER_FILE: &str = "hvx_hexagon_protos.h";

/// Intrinsic information parsed from the LLVM header
#[derive(Debug, Clone)]
struct IntrinsicInfo {
    /// The Q6_* intrinsic name (e.g., "Q6_V_vadd_VV")
    q6_name: String,
    /// The LLVM builtin name without prefix (e.g., "V6_vaddb")
    builtin_name: String,
    /// The short instruction name for assert_instr (e.g., "vaddb")
    instr_name: String,
    /// The assembly syntax from the comment
    asm_syntax: String,
    /// Instruction type
    instr_type: String,
    /// Execution slots
    exec_slots: String,
    /// Minimum HVX architecture version required
    min_arch: u32,
    /// Return type
    return_type: RustType,
    /// Parameters (name, type)
    params: Vec<(String, RustType)>,
    /// Whether this is a compound intrinsic (multiple builtins)
    is_compound: bool,
    /// For compound intrinsics: the parsed expression tree
    compound_expr: Option<CompoundExpr>,
}

/// Expression tree for compound intrinsics
#[derive(Debug, Clone)]
enum CompoundExpr {
    /// A call to a builtin: (builtin_name without V6_ prefix, arguments)
    BuiltinCall(String, Vec<CompoundExpr>),
    /// A parameter reference by name
    Param(String),
    /// An integer literal (like -1)
    IntLiteral(i32),
}

/// Rust type mappings
#[derive(Debug, Clone, PartialEq)]
enum RustType {
    HvxVector,
    HvxVectorPair,
    HvxVectorPred,
    I32,
    MutPtrHvxVector,
    Unit,
}

impl RustType {
    fn from_c_type(c_type: &str) -> Option<Self> {
        match c_type.trim() {
            "HVX_Vector" => Some(RustType::HvxVector),
            "HVX_VectorPair" => Some(RustType::HvxVectorPair),
            "HVX_VectorPred" => Some(RustType::HvxVectorPred),
            "Word32" => Some(RustType::I32),
            "HVX_Vector*" => Some(RustType::MutPtrHvxVector),
            "void" => Some(RustType::Unit),
            _ => None,
        }
    }

    fn to_rust_str(&self) -> &'static str {
        match self {
            RustType::HvxVector => "HvxVector",
            RustType::HvxVectorPair => "HvxVectorPair",
            RustType::HvxVectorPred => "HvxVectorPred",
            RustType::I32 => "i32",
            RustType::MutPtrHvxVector => "*mut HvxVector",
            RustType::Unit => "()",
        }
    }

    fn to_extern_str(&self) -> &'static str {
        match self {
            RustType::HvxVector => "HvxVector",
            RustType::HvxVectorPair => "HvxVectorPair",
            RustType::HvxVectorPred => "HvxVectorPred",
            RustType::I32 => "i32",
            RustType::MutPtrHvxVector => "*mut HvxVector",
            RustType::Unit => "()",
        }
    }
}

/// Parse a compound macro expression into an expression tree
fn parse_compound_expr(expr: &str) -> Option<CompoundExpr> {
    let expr = expr.trim();

    // Try to match an integer literal (like -1)
    if let Ok(n) = expr.parse::<i32>() {
        return Some(CompoundExpr::IntLiteral(n));
    }

    // Try to match a simple parameter name (Vu, Vv, Rt, Qs, Qt, Qx, Vx, etc.)
    // These are typically short identifiers in the macro
    if expr.len() <= 3
        && expr.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !expr.contains("__")
    {
        return Some(CompoundExpr::Param(expr.to_lowercase()));
    }

    // Check if it's wrapped in extra parens first
    if expr.starts_with('(') && expr.ends_with(')') {
        // Check if these parens wrap the entire expression
        let inner = &expr[1..expr.len() - 1];
        // Count depth: if after removing outer parens the expression is balanced,
        // the outer parens were enclosing everything
        if is_balanced_parens(inner) {
            // But we also need to verify these aren't part of a function call
            // If the inner expression is balanced and the whole thing starts with (
            // and ends with ), it's a paren wrapper
            let result = parse_compound_expr(inner);
            if result.is_some() {
                return result;
            }
        }
    }

    // Try to match __BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_xxx)(args)
    // The args portion may contain nested calls, so we need to find the matching paren
    if expr.starts_with("__BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_") {
        // Find the end of the builtin name (after V6_)
        let prefix = "__BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_";
        let after_prefix = &expr[prefix.len()..];
        if let Some(paren_pos) = after_prefix.find(')') {
            let builtin_name = &after_prefix[..paren_pos];
            let rest = &after_prefix[paren_pos + 1..]; // Skip the closing ) of the WRAP
                                                       // rest should now be "(args)"
            if rest.starts_with('(') && rest.ends_with(')') {
                let args_str = &rest[1..rest.len() - 1];
                let args = parse_compound_args(args_str)?;
                return Some(CompoundExpr::BuiltinCall(builtin_name.to_string(), args));
            }
        }
    }

    // Try to match __builtin_HEXAGON_V6_xxx(args) without wrap
    if expr.starts_with("__builtin_HEXAGON_V6_") {
        let prefix = "__builtin_HEXAGON_V6_";
        let after_prefix = &expr[prefix.len()..];
        if let Some(paren_pos) = after_prefix.find('(') {
            let builtin_name = &after_prefix[..paren_pos];
            let rest = &after_prefix[paren_pos..];
            if rest.starts_with('(') && rest.ends_with(')') {
                let args_str = &rest[1..rest.len() - 1];
                let args = parse_compound_args(args_str)?;
                return Some(CompoundExpr::BuiltinCall(builtin_name.to_string(), args));
            }
        }
    }

    None
}

/// Check if parentheses are balanced in a string
fn is_balanced_parens(s: &str) -> bool {
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Parse comma-separated arguments, respecting nested parentheses
fn parse_compound_args(args_str: &str) -> Option<Vec<CompoundExpr>> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in args_str.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let arg = current.trim().to_string();
                if !arg.is_empty() {
                    args.push(parse_compound_expr(&arg)?);
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    // Don't forget the last argument
    let arg = current.trim().to_string();
    if !arg.is_empty() {
        args.push(parse_compound_expr(&arg)?);
    }

    Some(args)
}

/// Extract all builtin names used in a compound expression
fn collect_builtins_from_expr(expr: &CompoundExpr, builtins: &mut HashSet<String>) {
    match expr {
        CompoundExpr::BuiltinCall(name, args) => {
            builtins.insert(name.clone());
            for arg in args {
                collect_builtins_from_expr(arg, builtins);
            }
        }
        CompoundExpr::Param(_) | CompoundExpr::IntLiteral(_) => {}
    }
}

/// Read the local HVX header file
fn read_header(crate_dir: &Path) -> Result<String, String> {
    let header_path = crate_dir.join(HEADER_FILE);
    println!("Reading HVX header from: {}", header_path.display());
    println!("  (LLVM version: {})", LLVM_VERSION);

    std::fs::read_to_string(&header_path).map_err(|e| {
        format!(
            "Failed to read header file {}: {}",
            header_path.display(),
            e
        )
    })
}

/// Parse a C function prototype to extract return type and parameters
fn parse_prototype(prototype: &str) -> Option<(RustType, Vec<(String, RustType)>)> {
    // Pattern: ReturnType FunctionName(ParamType1 Param1, ParamType2 Param2, ...)
    let proto_re = Regex::new(r"(\w+(?:\*)?)\s+Q6_\w+\(([^)]*)\)").unwrap();

    if let Some(caps) = proto_re.captures(prototype) {
        let return_type_str = caps[1].trim();
        let params_str = &caps[2];

        let return_type = RustType::from_c_type(return_type_str)?;

        let mut params = Vec::new();
        if !params_str.trim().is_empty() {
            // Pattern: Type Name or Type* Name
            let param_re = Regex::new(r"(\w+\*?)\s+(\w+)").unwrap();
            for param in params_str.split(',') {
                let param = param.trim();
                if let Some(pcaps) = param_re.captures(param) {
                    let ptype_str = pcaps[1].trim();
                    let pname = pcaps[2].to_lowercase();
                    if let Some(ptype) = RustType::from_c_type(ptype_str) {
                        params.push((pname, ptype));
                    } else {
                        return None; // Unknown type
                    }
                }
            }
        }

        Some((return_type, params))
    } else {
        None
    }
}

/// Parse the LLVM header file to extract intrinsic information
fn parse_header(content: &str) -> Vec<IntrinsicInfo> {
    let mut intrinsics = Vec::new();

    let arch_re = Regex::new(r"#if __HVX_ARCH__ >= (\d+)").unwrap();

    // Regex to extract the simple builtin name from a macro body
    // Match: __BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_xxx)(args)
    let simple_builtin_re =
        Regex::new(r"__BUILTIN_VECTOR_WRAP\(__builtin_HEXAGON_(\w+)\)\([^)]*\)\s*$").unwrap();

    // Also handle builtins without VECTOR_WRAP
    let simple_builtin_re2 = Regex::new(r"__builtin_HEXAGON_(\w+)\([^)]*\)\s*$").unwrap();

    // Regex to extract Q6 name from #define
    let q6_name_re = Regex::new(r"#define\s+(Q6_\w+)").unwrap();

    // Regex to extract macro expression body
    let macro_expr_re = Regex::new(r"#define\s+Q6_\w+\([^)]*\)\s+(.+)").unwrap();

    let lines: Vec<&str> = content.lines().collect();
    let mut current_arch: u32 = 60;
    let mut i = 0;

    while i < lines.len() {
        // Track architecture version
        if let Some(caps) = arch_re.captures(lines[i]) {
            if let Ok(arch) = caps[1].parse() {
                current_arch = arch;
            }
        }

        // Look for Assembly Syntax comment block
        if lines[i].contains("Assembly Syntax:") {
            let mut asm_syntax = String::new();
            let mut prototype = String::new();
            let mut instr_type = String::new();
            let mut exec_slots = String::new();

            // Parse the comment block
            let mut j = i;
            while j < lines.len() && !lines[j].starts_with("#define") {
                let line = lines[j];
                if line.contains("Assembly Syntax:") {
                    if let Some(pos) = line.find("Assembly Syntax:") {
                        asm_syntax = line[pos + 16..].trim().to_string();
                    }
                } else if line.contains("C Intrinsic Prototype:") {
                    if let Some(pos) = line.find("C Intrinsic Prototype:") {
                        prototype = line[pos + 22..].trim().to_string();
                    }
                } else if line.contains("Instruction Type:") {
                    if let Some(pos) = line.find("Instruction Type:") {
                        instr_type = line[pos + 17..].trim().to_string();
                    }
                } else if line.contains("Execution Slots:") {
                    if let Some(pos) = line.find("Execution Slots:") {
                        exec_slots = line[pos + 16..].trim().to_string();
                    }
                }
                j += 1;
            }

            // Now find the #define line
            while j < lines.len() && !lines[j].starts_with("#define") {
                j += 1;
            }

            if j < lines.len() {
                let define_line = lines[j];

                // Extract Q6 name and check if it's simple or compound
                if let Some(caps) = q6_name_re.captures(define_line) {
                    let q6_name = caps[1].to_string();

                    // Get the full macro body (handle line continuations)
                    let mut macro_body = define_line.to_string();
                    let mut k = j;
                    while macro_body.trim_end().ends_with('\\') && k + 1 < lines.len() {
                        k += 1;
                        macro_body.push_str(lines[k]);
                    }

                    // Try to extract simple builtin name
                    let builtin_name = simple_builtin_re
                        .captures(&macro_body)
                        .or_else(|| simple_builtin_re2.captures(&macro_body))
                        .map(|bcaps| bcaps[1].to_string());

                    // Check if it's a compound intrinsic (multiple __builtin calls)
                    let builtin_count = macro_body.matches("__builtin_HEXAGON_").count();
                    let is_compound = builtin_count > 1;

                    // Parse prototype
                    if let Some((return_type, params)) = parse_prototype(&prototype) {
                        if is_compound {
                            // For compound intrinsics, parse the expression
                            // Extract the macro body after the parameter list
                            if let Some(expr_caps) = macro_expr_re.captures(&macro_body) {
                                let expr_str = expr_caps[1].trim().replace(['\n', '\\'], " ");
                                let expr_str = expr_str.trim();

                                if let Some(compound_expr) = parse_compound_expr(expr_str) {
                                    // For compound intrinsics, we use the outermost builtin
                                    // as the "primary" for the instruction name
                                    let (primary_builtin, instr_name) = match &compound_expr {
                                        CompoundExpr::BuiltinCall(name, _) => {
                                            (name.clone(), name.clone())
                                        }
                                        _ => continue,
                                    };

                                    intrinsics.push(IntrinsicInfo {
                                        q6_name,
                                        builtin_name: format!("V6_{}", primary_builtin),
                                        instr_name,
                                        asm_syntax,
                                        instr_type,
                                        exec_slots,
                                        min_arch: current_arch,
                                        return_type,
                                        params,
                                        is_compound: true,
                                        compound_expr: Some(compound_expr),
                                    });
                                }
                            }
                        } else if let Some(builtin) = builtin_name {
                            // Extract short instruction name
                            let instr_name = builtin
                                .strip_prefix("V6_")
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| builtin.clone());

                            intrinsics.push(IntrinsicInfo {
                                q6_name,
                                builtin_name: builtin,
                                instr_name,
                                asm_syntax,
                                instr_type,
                                exec_slots,
                                min_arch: current_arch,
                                return_type,
                                params,
                                is_compound: false,
                                compound_expr: None,
                            });
                        }
                    }
                }
            }
            i = j;
        }
        i += 1;
    }

    intrinsics
}

/// Convert Q6 name to Rust function name (lowercase with underscores)
fn q6_to_rust_name(q6_name: &str) -> String {
    // Q6_V_hi_W -> q6_v_hi_w
    q6_name.to_lowercase()
}

/// Generate the module documentation
fn generate_module_doc(mode: VectorMode) -> String {
    format!(
        r#"//! Hexagon HVX {bytes}-byte vector mode intrinsics
//!
//! This module provides intrinsics for the Hexagon Vector Extensions (HVX)
//! in {bytes}-byte vector mode ({bits}-bit vectors).
//!
//! HVX is a wide vector extension designed for high-performance signal processing.
//! [Hexagon HVX Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61)
//!
//! ## Vector Types
//!
//! In {bytes}-byte mode:
//! - `HvxVector` is {bits} bits ({bytes} bytes) containing {lanes} x 32-bit values
//! - `HvxVectorPair` is {pair_bits} bits ({pair_bytes} bytes)
//! - `HvxVectorPred` is {bits} bits ({bytes} bytes) for predicate operations
//!
//! To use this module, compile with `-C target-feature=+{target_feature}`.
//!
//! ## Architecture Versions
//!
//! Different intrinsics require different HVX architecture versions. Use the
//! appropriate target feature to enable the required version:
//! - HVX v60: `-C target-feature=+hvxv60` (basic HVX operations)
//! - HVX v62: `-C target-feature=+hvxv62`
//! - HVX v65: `-C target-feature=+hvxv65` (includes floating-point support)
//! - HVX v66: `-C target-feature=+hvxv66`
//! - HVX v68: `-C target-feature=+hvxv68`
//! - HVX v69: `-C target-feature=+hvxv69`
//! - HVX v73: `-C target-feature=+hvxv73`
//! - HVX v79: `-C target-feature=+hvxv79`
//!
//! Each version includes all features from previous versions.
"#,
        bytes = mode.bytes(),
        bits = mode.bits(),
        lanes = mode.lanes(),
        pair_bytes = mode.bytes() * 2,
        pair_bits = mode.bits() * 2,
        target_feature = mode.target_feature(),
    )
}

/// Generate the type definitions for a specific vector mode
fn generate_types(mode: VectorMode) -> String {
    let lanes = mode.lanes();
    let pair_lanes = lanes * 2;
    let bits = mode.bits();
    let bytes = mode.bytes();
    let pair_bits = bits * 2;
    let pair_bytes = bytes * 2;

    format!(
        r#"
#![allow(non_camel_case_types)]

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::intrinsics::simd::{{simd_add, simd_and, simd_or, simd_sub, simd_xor}};

// HVX type definitions for {bytes}-byte vector mode
types! {{
    #![unstable(feature = "stdarch_hexagon", issue = "{TRACKING_ISSUE}")]

    /// HVX vector type ({bits} bits / {bytes} bytes)
    ///
    /// This type represents a single HVX vector register containing {lanes} x 32-bit values.
    pub struct HvxVector({lanes} x i32);

    /// HVX vector pair type ({pair_bits} bits / {pair_bytes} bytes)
    ///
    /// This type represents a pair of HVX vector registers, often used for
    /// operations that produce double-width results.
    pub struct HvxVectorPair({pair_lanes} x i32);

    /// HVX vector predicate type ({bits} bits / {bytes} bytes)
    ///
    /// This type represents a predicate vector used for conditional operations.
    /// Each bit corresponds to a lane in the vector.
    pub struct HvxVectorPred({lanes} x i32);
}}
"#,
        bytes = bytes,
        bits = bits,
        lanes = lanes,
        pair_bits = pair_bits,
        pair_bytes = pair_bytes,
        pair_lanes = pair_lanes,
        TRACKING_ISSUE = TRACKING_ISSUE,
    )
}

/// Builtin signature information for extern declarations
struct BuiltinSignature {
    /// The V6_ prefixed name
    full_name: String,
    /// The short name (without V6_)
    short_name: String,
    /// Return type
    return_type: RustType,
    /// Parameter types
    param_types: Vec<RustType>,
}

/// Get known signatures for builtins used in compound operations
/// These are the helper builtins that don't have their own Q6_ wrapper
fn get_compound_helper_signatures() -> HashMap<String, BuiltinSignature> {
    let mut map = HashMap::new();

    // vandvrt: HVX_Vector -> i32 -> HVX_Vector
    // Converts predicate to vector representation. LLVM uses HVX_Vector for both.
    map.insert(
        "vandvrt".to_string(),
        BuiltinSignature {
            full_name: "V6_vandvrt".to_string(),
            short_name: "vandvrt".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::I32],
        },
    );

    // vandqrt: HVX_Vector -> i32 -> HVX_Vector
    // Converts vector representation back to predicate. LLVM uses HVX_Vector for both.
    map.insert(
        "vandqrt".to_string(),
        BuiltinSignature {
            full_name: "V6_vandqrt".to_string(),
            short_name: "vandqrt".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::I32],
        },
    );

    // vandvrt_acc: HVX_Vector -> HVX_Vector -> i32 -> HVX_Vector
    map.insert(
        "vandvrt_acc".to_string(),
        BuiltinSignature {
            full_name: "V6_vandvrt_acc".to_string(),
            short_name: "vandvrt_acc".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector, RustType::I32],
        },
    );

    // vandqrt_acc: HVX_Vector -> HVX_Vector -> i32 -> HVX_Vector
    map.insert(
        "vandqrt_acc".to_string(),
        BuiltinSignature {
            full_name: "V6_vandqrt_acc".to_string(),
            short_name: "vandqrt_acc".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector, RustType::I32],
        },
    );

    // pred_and: HVX_Vector -> HVX_Vector -> HVX_Vector
    map.insert(
        "pred_and".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_and".to_string(),
            short_name: "pred_and".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // pred_and_n: HVX_Vector -> HVX_Vector -> HVX_Vector
    map.insert(
        "pred_and_n".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_and_n".to_string(),
            short_name: "pred_and_n".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // pred_or: HVX_Vector -> HVX_Vector -> HVX_Vector
    map.insert(
        "pred_or".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_or".to_string(),
            short_name: "pred_or".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // pred_or_n: HVX_Vector -> HVX_Vector -> HVX_Vector
    map.insert(
        "pred_or_n".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_or_n".to_string(),
            short_name: "pred_or_n".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // pred_xor: HVX_Vector -> HVX_Vector -> HVX_Vector
    map.insert(
        "pred_xor".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_xor".to_string(),
            short_name: "pred_xor".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // pred_not: HVX_Vector -> HVX_Vector
    map.insert(
        "pred_not".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_not".to_string(),
            short_name: "pred_not".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector],
        },
    );

    // pred_scalar2: i32 -> HVX_Vector
    map.insert(
        "pred_scalar2".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_scalar2".to_string(),
            short_name: "pred_scalar2".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::I32],
        },
    );

    // Conditional store operations
    map.insert(
        "vS32b_qpred_ai".to_string(),
        BuiltinSignature {
            full_name: "V6_vS32b_qpred_ai".to_string(),
            short_name: "vS32b_qpred_ai".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vS32b_nqpred_ai".to_string(),
        BuiltinSignature {
            full_name: "V6_vS32b_nqpred_ai".to_string(),
            short_name: "vS32b_nqpred_ai".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vS32b_nt_qpred_ai".to_string(),
        BuiltinSignature {
            full_name: "V6_vS32b_nt_qpred_ai".to_string(),
            short_name: "vS32b_nt_qpred_ai".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vS32b_nt_nqpred_ai".to_string(),
        BuiltinSignature {
            full_name: "V6_vS32b_nt_nqpred_ai".to_string(),
            short_name: "vS32b_nt_nqpred_ai".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
            ],
        },
    );

    // Conditional accumulation operations
    for (suffix, _elem) in [("b", "byte"), ("h", "halfword"), ("w", "word")] {
        // vaddbq, vaddhq, vaddwq
        map.insert(
            format!("vadd{}q", suffix),
            BuiltinSignature {
                full_name: format!("V6_vadd{}q", suffix),
                short_name: format!("vadd{}q", suffix),
                return_type: RustType::HvxVector,
                param_types: vec![
                    RustType::HvxVector,
                    RustType::HvxVector,
                    RustType::HvxVector,
                ],
            },
        );
        // vaddbnq, vaddhnq, vaddwnq
        map.insert(
            format!("vadd{}nq", suffix),
            BuiltinSignature {
                full_name: format!("V6_vadd{}nq", suffix),
                short_name: format!("vadd{}nq", suffix),
                return_type: RustType::HvxVector,
                param_types: vec![
                    RustType::HvxVector,
                    RustType::HvxVector,
                    RustType::HvxVector,
                ],
            },
        );
    }

    // Comparison operations with accumulation
    // veqb_and, veqb_or, veqb_xor, etc.
    for elem in ["b", "h", "w", "ub", "uh", "uw"] {
        for op in ["and", "or", "xor"] {
            // veq*_and, veq*_or, veq*_xor
            map.insert(
                format!("veq{}_{}", elem, op),
                BuiltinSignature {
                    full_name: format!("V6_veq{}_{}", elem, op),
                    short_name: format!("veq{}_{}", elem, op),
                    return_type: RustType::HvxVector,
                    param_types: vec![
                        RustType::HvxVector,
                        RustType::HvxVector,
                        RustType::HvxVector,
                    ],
                },
            );
            // vgt*_and, vgt*_or, vgt*_xor
            map.insert(
                format!("vgt{}_{}", elem, op),
                BuiltinSignature {
                    full_name: format!("V6_vgt{}_{}", elem, op),
                    short_name: format!("vgt{}_{}", elem, op),
                    return_type: RustType::HvxVector,
                    param_types: vec![
                        RustType::HvxVector,
                        RustType::HvxVector,
                        RustType::HvxVector,
                    ],
                },
            );
        }
    }

    // Floating-point comparison operations (hf = half-float, sf = single-float)
    for elem in ["hf", "sf"] {
        // Basic comparison: vgt*
        map.insert(
            format!("vgt{}", elem),
            BuiltinSignature {
                full_name: format!("V6_vgt{}", elem),
                short_name: format!("vgt{}", elem),
                return_type: RustType::HvxVector,
                param_types: vec![RustType::HvxVector, RustType::HvxVector],
            },
        );

        for op in ["and", "or", "xor"] {
            // vgt*_and, vgt*_or, vgt*_xor
            map.insert(
                format!("vgt{}_{}", elem, op),
                BuiltinSignature {
                    full_name: format!("V6_vgt{}_{}", elem, op),
                    short_name: format!("vgt{}_{}", elem, op),
                    return_type: RustType::HvxVector,
                    param_types: vec![
                        RustType::HvxVector,
                        RustType::HvxVector,
                        RustType::HvxVector,
                    ],
                },
            );
        }
    }

    // Prefix operations with predicate
    for elem in ["b", "h", "w"] {
        map.insert(
            format!("vprefixq{}", elem),
            BuiltinSignature {
                full_name: format!("V6_vprefixq{}", elem),
                short_name: format!("vprefixq{}", elem),
                return_type: RustType::HvxVector,
                param_types: vec![RustType::HvxVector],
            },
        );
    }

    // Scatter operations with predicate
    map.insert(
        "vscattermhq".to_string(),
        BuiltinSignature {
            full_name: "V6_vscattermhq".to_string(),
            short_name: "vscattermhq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVector,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vscattermhwq".to_string(),
        BuiltinSignature {
            full_name: "V6_vscattermhwq".to_string(),
            short_name: "vscattermhwq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVectorPair,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vscattermwq".to_string(),
        BuiltinSignature {
            full_name: "V6_vscattermwq".to_string(),
            short_name: "vscattermwq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVector,
                RustType::HvxVector,
            ],
        },
    );

    // Add with carry saturation
    map.insert(
        "vaddcarrysat".to_string(),
        BuiltinSignature {
            full_name: "V6_vaddcarrysat".to_string(),
            short_name: "vaddcarrysat".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![
                RustType::HvxVector,
                RustType::HvxVector,
                RustType::HvxVector,
            ],
        },
    );

    // Gather operations with predicate
    map.insert(
        "vgathermhq".to_string(),
        BuiltinSignature {
            full_name: "V6_vgathermhq".to_string(),
            short_name: "vgathermhq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVector,
            ],
        },
    );

    map.insert(
        "vgathermhwq".to_string(),
        BuiltinSignature {
            full_name: "V6_vgathermhwq".to_string(),
            short_name: "vgathermhwq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVectorPair,
            ],
        },
    );

    map.insert(
        "vgathermwq".to_string(),
        BuiltinSignature {
            full_name: "V6_vgathermwq".to_string(),
            short_name: "vgathermwq".to_string(),
            return_type: RustType::Unit,
            param_types: vec![
                RustType::MutPtrHvxVector,
                RustType::HvxVector,
                RustType::I32,
                RustType::I32,
                RustType::HvxVector,
            ],
        },
    );

    // Basic comparison operations (without accumulation)
    for elem in ["b", "h", "w", "ub", "uh", "uw"] {
        // vgt* - greater than
        map.insert(
            format!("vgt{}", elem),
            BuiltinSignature {
                full_name: format!("V6_vgt{}", elem),
                short_name: format!("vgt{}", elem),
                return_type: RustType::HvxVector,
                param_types: vec![RustType::HvxVector, RustType::HvxVector],
            },
        );
        // veq* - equal
        map.insert(
            format!("veq{}", elem),
            BuiltinSignature {
                full_name: format!("V6_veq{}", elem),
                short_name: format!("veq{}", elem),
                return_type: RustType::HvxVector,
                param_types: vec![RustType::HvxVector, RustType::HvxVector],
            },
        );
    }

    // Conditional subtraction operations (vsub*q, vsub*nq)
    for elem in ["b", "h", "w"] {
        map.insert(
            format!("vsub{}q", elem),
            BuiltinSignature {
                full_name: format!("V6_vsub{}q", elem),
                short_name: format!("vsub{}q", elem),
                return_type: RustType::HvxVector,
                param_types: vec![
                    RustType::HvxVector,
                    RustType::HvxVector,
                    RustType::HvxVector,
                ],
            },
        );
        map.insert(
            format!("vsub{}nq", elem),
            BuiltinSignature {
                full_name: format!("V6_vsub{}nq", elem),
                short_name: format!("vsub{}nq", elem),
                return_type: RustType::HvxVector,
                param_types: vec![
                    RustType::HvxVector,
                    RustType::HvxVector,
                    RustType::HvxVector,
                ],
            },
        );
    }

    // vmux - vector mux (select based on predicate)
    map.insert(
        "vmux".to_string(),
        BuiltinSignature {
            full_name: "V6_vmux".to_string(),
            short_name: "vmux".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![
                RustType::HvxVector,
                RustType::HvxVector,
                RustType::HvxVector,
            ],
        },
    );

    // vswap - vector swap based on predicate
    map.insert(
        "vswap".to_string(),
        BuiltinSignature {
            full_name: "V6_vswap".to_string(),
            short_name: "vswap".to_string(),
            return_type: RustType::HvxVectorPair,
            param_types: vec![
                RustType::HvxVector,
                RustType::HvxVector,
                RustType::HvxVector,
            ],
        },
    );

    // shuffeq operations - take vectors (internal pred representation) and return vector
    for elem in ["h", "w"] {
        map.insert(
            format!("shuffeq{}", elem),
            BuiltinSignature {
                full_name: format!("V6_shuffeq{}", elem),
                short_name: format!("shuffeq{}", elem),
                return_type: RustType::HvxVector,
                param_types: vec![RustType::HvxVector, RustType::HvxVector],
            },
        );
    }

    // Predicate AND with vector operations
    map.insert(
        "vandvqv".to_string(),
        BuiltinSignature {
            full_name: "V6_vandvqv".to_string(),
            short_name: "vandvqv".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    map.insert(
        "vandvnqv".to_string(),
        BuiltinSignature {
            full_name: "V6_vandvnqv".to_string(),
            short_name: "vandvnqv".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector],
        },
    );

    // vandnqrt and vandnqrt_acc
    map.insert(
        "vandnqrt".to_string(),
        BuiltinSignature {
            full_name: "V6_vandnqrt".to_string(),
            short_name: "vandnqrt".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::I32],
        },
    );

    map.insert(
        "vandnqrt_acc".to_string(),
        BuiltinSignature {
            full_name: "V6_vandnqrt_acc".to_string(),
            short_name: "vandnqrt_acc".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::HvxVector, RustType::HvxVector, RustType::I32],
        },
    );

    // pred_scalar2v2
    map.insert(
        "pred_scalar2v2".to_string(),
        BuiltinSignature {
            full_name: "V6_pred_scalar2v2".to_string(),
            short_name: "pred_scalar2v2".to_string(),
            return_type: RustType::HvxVector,
            param_types: vec![RustType::I32],
        },
    );

    map
}

/// Generate extern declarations for all intrinsics for a specific vector mode
fn generate_extern_block(intrinsics: &[IntrinsicInfo], mode: VectorMode) -> String {
    let mut output = String::new();

    // Collect unique builtins to avoid duplicates
    let mut seen_builtins: HashSet<String> = HashSet::new();
    let mut decls: Vec<(String, String, RustType, Vec<RustType>)> = Vec::new();

    // First, add simple intrinsics
    for info in intrinsics.iter().filter(|i| !i.is_compound) {
        if seen_builtins.contains(&info.builtin_name) {
            continue;
        }
        seen_builtins.insert(info.builtin_name.clone());

        let param_types: Vec<RustType> = info.params.iter().map(|(_, t)| t.clone()).collect();
        decls.push((
            info.builtin_name.clone(),
            info.instr_name.clone(),
            info.return_type.clone(),
            param_types,
        ));
    }

    // Then, collect all builtins used in compound expressions
    let helper_sigs = get_compound_helper_signatures();
    let mut compound_builtins: HashSet<String> = HashSet::new();

    for info in intrinsics.iter().filter(|i| i.is_compound) {
        if let Some(ref expr) = info.compound_expr {
            collect_builtins_from_expr(expr, &mut compound_builtins);
        }
    }

    // Add compound helper builtins
    let mut missing_builtins = Vec::new();
    for builtin_name in compound_builtins {
        let full_name = format!("V6_{}", builtin_name);
        if seen_builtins.contains(&full_name) {
            continue;
        }
        seen_builtins.insert(full_name.clone());

        if let Some(sig) = helper_sigs.get(&builtin_name) {
            decls.push((
                sig.full_name.clone(),
                sig.short_name.clone(),
                sig.return_type.clone(),
                sig.param_types.clone(),
            ));
        } else {
            missing_builtins.push(builtin_name);
        }
    }

    // Report missing builtins (for development purposes)
    if !missing_builtins.is_empty() {
        eprintln!("Warning: Missing helper signatures for compound builtins:");
        for name in &missing_builtins {
            eprintln!("  - {}", name);
        }
    }

    // Sort by builtin name for consistent output
    decls.sort_by(|a, b| a.0.cmp(&b.0));

    // Generate intrinsic declarations for the specified mode
    output.push_str(&format!(
        "// LLVM intrinsic declarations for {}-byte vector mode\n",
        mode.bytes()
    ));
    output.push_str("#[allow(improper_ctypes)]\n");
    output.push_str("unsafe extern \"unadjusted\" {\n");

    for (builtin_name, instr_name, return_type, param_types) in &decls {
        let base_link = builtin_name.replace('_', ".");
        // 128-byte mode uses .128B suffix, 64-byte mode doesn't
        let link_name = if builtin_name.starts_with("V6_") && mode == VectorMode::V128 {
            format!("llvm.hexagon.{}.128B", base_link)
        } else {
            format!("llvm.hexagon.{}", base_link)
        };

        let params_str = if param_types.is_empty() {
            String::new()
        } else {
            param_types
                .iter()
                .map(|t| format!("_: {}", t.to_extern_str()))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let return_str = if *return_type == RustType::Unit {
            " -> ()".to_string()
        } else {
            format!(" -> {}", return_type.to_extern_str())
        };

        output.push_str(&format!(
            "    #[link_name = \"{}\"]\n    fn {}({}){};\n",
            link_name, instr_name, params_str, return_str
        ));
    }

    output.push_str("}\n");
    output
}

/// Generate Rust code for a compound expression
/// `params` maps parameter names to their types in the function signature
/// Get the type of an expression
fn get_expr_type(
    expr: &CompoundExpr,
    params: &HashMap<String, RustType>,
    helper_sigs: &HashMap<String, BuiltinSignature>,
) -> Option<RustType> {
    match expr {
        CompoundExpr::BuiltinCall(name, _) => {
            helper_sigs.get(name).map(|sig| sig.return_type.clone())
        }
        CompoundExpr::Param(name) => params.get(name).cloned(),
        CompoundExpr::IntLiteral(_) => Some(RustType::I32),
    }
}

fn generate_compound_expr_code(
    expr: &CompoundExpr,
    params: &HashMap<String, RustType>,
    helper_sigs: &HashMap<String, BuiltinSignature>,
) -> String {
    match expr {
        CompoundExpr::BuiltinCall(name, args) => {
            // Get the expected parameter types for this builtin
            let expected_types = helper_sigs
                .get(name)
                .map(|sig| sig.param_types.clone())
                .unwrap_or_default();

            let args_code: Vec<String> = args
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    let arg_code = generate_compound_expr_code(arg, params, helper_sigs);

                    // Check if we need to transmute this argument
                    let expected_type = expected_types.get(i);
                    let actual_type = get_expr_type(arg, params, helper_sigs);

                    // If the builtin expects HvxVector but the arg is HvxVectorPred, transmute
                    if expected_type == Some(&RustType::HvxVector)
                        && actual_type == Some(RustType::HvxVectorPred)
                    {
                        format!(
                            "core::mem::transmute::<HvxVectorPred, HvxVector>({})",
                            arg_code
                        )
                    } else {
                        arg_code
                    }
                })
                .collect();
            format!("{}({})", name, args_code.join(", "))
        }
        CompoundExpr::Param(name) => name.clone(),
        CompoundExpr::IntLiteral(n) => n.to_string(),
    }
}

/// Get the primary instruction name from a compound expression (innermost significant op)
fn get_compound_primary_instr(expr: &CompoundExpr) -> Option<String> {
    match expr {
        CompoundExpr::BuiltinCall(name, args) => {
            // For vandqrt wrapper, look inside
            if name == "vandqrt" && !args.is_empty() {
                if let Some(inner) = get_compound_primary_instr(&args[0]) {
                    return Some(inner);
                }
            }
            // For store operations, use the store name
            if name.starts_with("vS32b") {
                return Some(name.clone());
            }
            // For conditional accumulation, use the add name
            if name.starts_with("vadd") && (name.ends_with("q") || name.ends_with("nq")) {
                return Some(name.clone());
            }
            // For predicate operations
            if name.starts_with("pred_") {
                return Some(name.clone());
            }
            // For comparison operations with accumulation
            if (name.starts_with("veq") || name.starts_with("vgt"))
                && (name.ends_with("_and") || name.ends_with("_or") || name.ends_with("_xor"))
            {
                return Some(name.clone());
            }
            Some(name.clone())
        }
        _ => None,
    }
}

/// Get override implementations for specific compound intrinsics.
/// Some C macros rely on implicit type conversions that don't work with
/// our stricter Rust types, so we provide corrected implementations.
fn get_compound_overrides() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // Q6_V_vand_QR: takes pred, returns vec
    // Use transmute to convert pred to vec for LLVM, call vandvrt
    map.insert(
        "Q6_V_vand_QR",
        "vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qu), rt)",
    );

    // Q6_V_vandor_VQR: takes vec and pred, returns vec
    map.insert(
        "Q6_V_vandor_VQR",
        "vandvrt_acc(vx, core::mem::transmute::<HvxVectorPred, HvxVector>(qu), rt)",
    );

    // Q6_Q_vand_VR: takes vec, returns pred
    map.insert(
        "Q6_Q_vand_VR",
        "core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vu, rt))",
    );

    // Q6_Q_vandor_QVR: takes pred and vec, returns pred
    map.insert(
        "Q6_Q_vandor_QVR",
        "core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt_acc(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), vu, rt))",
    );

    map
}

/// Generate wrapper functions for all intrinsics
fn generate_functions(intrinsics: &[IntrinsicInfo]) -> String {
    let mut output = String::new();
    let simd_mappings = get_simd_intrinsic_mappings();

    // Generate simple intrinsics
    for info in intrinsics.iter().filter(|i| !i.is_compound) {
        let rust_name = q6_to_rust_name(&info.q6_name);

        // Generate doc comment
        output.push_str(&format!("/// `{}`\n", info.asm_syntax));
        output.push_str("///\n");
        output.push_str(&format!("/// Instruction Type: {}\n", info.instr_type));
        output.push_str(&format!("/// Execution Slots: {}\n", info.exec_slots));

        // Generate attributes
        output.push_str("#[inline(always)]\n");
        output.push_str(&format!(
            "#[cfg_attr(target_arch = \"hexagon\", target_feature(enable = \"hvxv{}\"))]\n",
            info.min_arch
        ));

        // Check if we should use simd intrinsic instead
        let use_simd = simd_mappings.get(info.instr_name.as_str());

        // assert_instr uses the original instruction name
        output.push_str(&format!(
            "#[cfg_attr(test, assert_instr({}))]\n",
            info.instr_name
        ));

        output.push_str(&format!(
            "#[unstable(feature = \"stdarch_hexagon\", issue = \"{}\")]\n",
            TRACKING_ISSUE
        ));

        // Generate function signature
        let params_str = info
            .params
            .iter()
            .map(|(name, ty)| format!("{}: {}", name, ty.to_rust_str()))
            .collect::<Vec<_>>()
            .join(", ");

        let return_str = if info.return_type == RustType::Unit {
            String::new()
        } else {
            format!(" -> {}", info.return_type.to_rust_str())
        };

        output.push_str(&format!(
            "pub unsafe fn {}({}){} {{\n",
            rust_name, params_str, return_str
        ));

        // Generate function body
        let args_str = info
            .params
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(simd_fn) = use_simd {
            // Use architecture-independent simd intrinsic
            output.push_str(&format!("    {}({})\n", simd_fn, args_str));
        } else {
            // Use the LLVM intrinsic
            output.push_str(&format!("    {}({})\n", info.instr_name, args_str));
        }

        output.push_str("}\n\n");
    }

    // Generate compound intrinsics
    let helper_sigs = get_compound_helper_signatures();
    let overrides = get_compound_overrides();
    for info in intrinsics.iter().filter(|i| i.is_compound) {
        if let Some(ref compound_expr) = info.compound_expr {
            let rust_name = q6_to_rust_name(&info.q6_name);

            // Get the primary instruction for assert_instr
            let _primary_instr = get_compound_primary_instr(compound_expr)
                .unwrap_or_else(|| info.instr_name.clone());

            // Generate doc comment
            output.push_str(&format!("/// `{}`\n", info.asm_syntax));
            output.push_str("///\n");
            output.push_str(
                "/// This is a compound operation composed of multiple HVX instructions.\n",
            );
            if !info.instr_type.is_empty() {
                output.push_str(&format!("/// Instruction Type: {}\n", info.instr_type));
            }
            if !info.exec_slots.is_empty() {
                output.push_str(&format!("/// Execution Slots: {}\n", info.exec_slots));
            }

            // Generate attributes
            output.push_str("#[inline(always)]\n");
            output.push_str(&format!(
                "#[cfg_attr(target_arch = \"hexagon\", target_feature(enable = \"hvxv{}\"))]\n",
                info.min_arch
            ));

            // For compound ops, we skip assert_instr since they emit multiple instructions
            // output.push_str(&format!(
            //     "#[cfg_attr(test, assert_instr({}))]\n",
            //     primary_instr
            // ));

            output.push_str(&format!(
                "#[unstable(feature = \"stdarch_hexagon\", issue = \"{}\")]\n",
                TRACKING_ISSUE
            ));

            // Generate function signature
            let params_str = info
                .params
                .iter()
                .map(|(name, ty)| format!("{}: {}", name, ty.to_rust_str()))
                .collect::<Vec<_>>()
                .join(", ");

            let return_str = if info.return_type == RustType::Unit {
                String::new()
            } else {
                format!(" -> {}", info.return_type.to_rust_str())
            };

            output.push_str(&format!(
                "pub unsafe fn {}({}){} {{\n",
                rust_name, params_str, return_str
            ));

            // Check if we have an override for this intrinsic
            let body = if let Some(override_body) = overrides.get(info.q6_name.as_str()) {
                override_body.to_string()
            } else {
                // Build param type map for expression code generation
                let param_types: HashMap<String, RustType> = info.params.iter().cloned().collect();
                // Generate function body from compound expression
                let expr_body =
                    generate_compound_expr_code(compound_expr, &param_types, &helper_sigs);

                // Check if we need to transmute the result
                let expr_return_type = get_expr_type(compound_expr, &param_types, &helper_sigs);
                if info.return_type == RustType::HvxVectorPred
                    && expr_return_type == Some(RustType::HvxVector)
                {
                    format!(
                        "core::mem::transmute::<HvxVector, HvxVectorPred>({})",
                        expr_body
                    )
                } else {
                    expr_body
                }
            };
            output.push_str(&format!("    {}\n", body));

            output.push_str("}\n\n");
        }
    }

    output
}

/// Generate a module file for a specific vector mode
fn generate_module_file(
    intrinsics: &[IntrinsicInfo],
    output_path: &Path,
    mode: VectorMode,
) -> Result<(), String> {
    let mut output =
        File::create(output_path).map_err(|e| format!("Failed to create output: {}", e))?;

    writeln!(output, "{}", generate_module_doc(mode)).map_err(|e| e.to_string())?;
    writeln!(output, "{}", generate_types(mode)).map_err(|e| e.to_string())?;
    writeln!(output, "{}", generate_extern_block(intrinsics, mode)).map_err(|e| e.to_string())?;
    writeln!(output, "{}", generate_functions(intrinsics)).map_err(|e| e.to_string())?;

    // Ensure file is flushed before running rustfmt
    drop(output);

    // Run rustfmt on the generated file
    let status = std::process::Command::new("rustfmt")
        .arg(output_path)
        .status()
        .map_err(|e| format!("Failed to run rustfmt: {}", e))?;

    if !status.success() {
        return Err("rustfmt failed".to_string());
    }

    Ok(())
}

fn main() -> Result<(), String> {
    println!("=== Hexagon HVX Code Generator ===\n");

    // Get the crate directory first (needed for both reading header and writing output)
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    // Read and parse the local LLVM header
    println!("Step 1: Reading LLVM HVX header...");
    let header_content = read_header(&crate_dir)?;
    println!("  Read {} bytes", header_content.len());

    println!("\nStep 2: Parsing intrinsic definitions...");
    let all_intrinsics = parse_header(&header_content);
    println!("  Found {} intrinsic definitions", all_intrinsics.len());

    // Filter out intrinsics requiring architecture versions not yet supported by rustc
    let intrinsics: Vec<_> = all_intrinsics
        .into_iter()
        .filter(|i| i.min_arch <= MAX_SUPPORTED_ARCH)
        .collect();
    let filtered_count = intrinsics.len();
    println!(
        "  Filtered to {} intrinsics (max supported: hvxv{})",
        filtered_count, MAX_SUPPORTED_ARCH
    );

    // Count simple vs compound
    let simple_count = intrinsics.iter().filter(|i| !i.is_compound).count();
    let compound_count = intrinsics.iter().filter(|i| i.is_compound).count();
    println!("  Simple intrinsics: {}", simple_count);
    println!("  Compound intrinsics: {}", compound_count);

    // Print some sample intrinsics for verification
    println!("\n  Sample simple intrinsics:");
    for info in intrinsics.iter().filter(|i| !i.is_compound).take(5) {
        println!(
            "    {} -> {} ({})",
            info.q6_name, info.builtin_name, info.asm_syntax
        );
    }

    println!("\n  Sample compound intrinsics:");
    for info in intrinsics.iter().filter(|i| i.is_compound).take(5) {
        println!("    {} ({})", info.q6_name, info.asm_syntax);
    }

    // Count architecture versions
    let mut arch_counts: HashMap<u32, usize> = HashMap::new();
    for info in &intrinsics {
        *arch_counts.entry(info.min_arch).or_insert(0) += 1;
    }
    println!("\n  By architecture version:");
    let mut archs: Vec<_> = arch_counts.iter().collect();
    archs.sort_by_key(|(k, _)| *k);
    for (arch, count) in archs {
        println!("    HVX v{}: {} intrinsics", arch, count);
    }

    // Generate output files
    let hexagon_dir = crate_dir.join("../core_arch/src/hexagon");

    // Generate v64.rs (64-byte vector mode)
    let v64_path = hexagon_dir.join("v64.rs");
    println!("\nStep 3: Generating v64.rs (64-byte mode)...");
    generate_module_file(&intrinsics, &v64_path, VectorMode::V64)?;
    println!("  Output: {}", v64_path.display());

    // Generate v128.rs (128-byte vector mode)
    let v128_path = hexagon_dir.join("v128.rs");
    println!("\nStep 4: Generating v128.rs (128-byte mode)...");
    generate_module_file(&intrinsics, &v128_path, VectorMode::V128)?;
    println!("  Output: {}", v128_path.display());

    println!("\n=== Results ===");
    println!(
        "  Generated {} simple wrapper functions per module",
        simple_count
    );
    println!(
        "  Generated {} compound wrapper functions per module",
        compound_count
    );
    println!(
        "  Total: {} functions per module",
        simple_count + compound_count
    );
    println!("  Output files: v64.rs, v128.rs");

    Ok(())
}
