//! Hexagon Scalar Code Generator
//!
//! This generator creates scalar.rs from the LLVM `hexagon_protos.h` header file.
//! It parses the C intrinsic prototypes and generates Rust wrapper functions
//! with appropriate attributes for all scalar (non-HVX) Hexagon intrinsics.
//!
//! The generated module provides ~901 scalar intrinsic wrappers covering:
//! - Arithmetic, multiply, shift, saturate operations
//! - Compare, floating-point, and other scalar operations
//!
//! Intrinsics with `void*`/`void**` parameters (circular-addressing) are skipped
//! because they have no corresponding LLVM intrinsic.
//!
//! Usage:
//!     cd crates/stdarch-gen-hexagon-scalar
//!     cargo run
//!     # Output is written to ../core_arch/src/hexagon/scalar.rs

use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Extract the instruction mnemonic from the assembly syntax string.
///
/// Examples:
/// - `Rd32=abs(Rs32)` → Some("abs")
/// - `Rd32=add(Rs32,Rt32):sat` → Some("add")
/// - `Rx32+=mpy(Rs32,Rt32)` → Some("mpy")
/// - `Rd32=dmpause` → Some("dmpause")
/// - `dmlink(Rs32,Rt32)` → Some("dmlink")
/// - `Rd32=Rs32` → None (simple transfer)
/// - `Rx32.h=#u16` → None (immediate load)
/// - `Rdd32=#s8` → None (immediate load)
fn extract_instr_name(asm_syntax: &str) -> Option<String> {
    // Find the operator: +=, -=, or =
    let after_op = if let Some(pos) = asm_syntax.find("+=") {
        &asm_syntax[pos + 2..]
    } else if let Some(pos) = asm_syntax.find("-=") {
        &asm_syntax[pos + 2..]
    } else if let Some(pos) = asm_syntax.find('=') {
        &asm_syntax[pos + 1..]
    } else {
        // No assignment operator: try function-call-style syntax like "dmlink(Rs32,Rt32)".
        // The mnemonic is the leading lowercase identifier.
        return extract_leading_mnemonic(asm_syntax);
    };

    // After the operator, we expect a lowercase letter starting the mnemonic.
    // Skip if it starts with uppercase (register name like Rs32) or # (immediate).
    extract_leading_mnemonic(after_op)
}

/// Extract a leading lowercase mnemonic from the given string.
///
/// Returns `Some(mnemonic)` if the string starts with a lowercase ASCII letter,
/// collecting all subsequent alphanumeric/underscore characters. Returns `None`
/// if the string is empty or starts with an uppercase letter, `#`, etc.
fn extract_leading_mnemonic(s: &str) -> Option<String> {
    let first_char = s.chars().next()?;
    if !first_char.is_ascii_lowercase() {
        return None;
    }
    let mnemonic: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if mnemonic.is_empty() {
        None
    } else {
        Some(mnemonic)
    }
}

/// The tracking issue number for the stdarch_hexagon feature
const TRACKING_ISSUE: &str = "151523";

/// LLVM version the header file is from (for reference)
const LLVM_VERSION: &str = "22.1.0";

/// Local header file path (checked into the repository)
const HEADER_FILE: &str = "hexagon_protos.h";

/// Rust type representation for scalar intrinsics
#[derive(Debug, Clone, PartialEq)]
enum RustType {
    I32,
    I64,
    F32,
    F64,
    Unit,
}

impl RustType {
    fn from_c_type(c_type: &str) -> Option<Self> {
        match c_type.trim() {
            "Word32" | "UWord32" | "Byte" | "Address" => Some(RustType::I32),
            "Word64" | "UWord64" => Some(RustType::I64),
            "Float32" => Some(RustType::F32),
            "Float64" => Some(RustType::F64),
            "void" => Some(RustType::Unit),
            _ => None,
        }
    }

    fn to_rust_str(&self) -> &'static str {
        match self {
            RustType::I32 => "i32",
            RustType::I64 => "i64",
            RustType::F32 => "f32",
            RustType::F64 => "f64",
            RustType::Unit => "()",
        }
    }
}

/// Information about an immediate operand parameter.
///
/// Detected from C prototype parameter names like `Is16` (signed 16-bit),
/// `Iu5` (unsigned 5-bit), `IU5` (unsigned 5-bit secondary), `Iu6_2`
/// (unsigned 6-bit with 2-bit alignment).
#[derive(Debug, Clone)]
struct ImmediateInfo {
    /// Whether this is a signed immediate
    signed: bool,
    /// Number of bits in the immediate field
    bits: u32,
}

/// Arch guard for an intrinsic
#[derive(Debug, Clone, PartialEq)]
enum ArchGuard {
    /// No guard (base v5/v55 intrinsics)
    None,
    /// `#if __HEXAGON_ARCH__ >= N`
    Arch(u32),
    /// `#if __HEXAGON_ARCH__ >= N && defined __HEXAGON_AUDIO__`
    ArchAudio(u32),
}

impl ArchGuard {
    /// Returns a doc comment describing the required architecture version,
    /// or None if no specific version is needed.
    fn requires_doc(&self) -> Option<String> {
        match self {
            ArchGuard::None => Option::None,
            ArchGuard::Arch(ver) => Some(format!("Requires: V{}", ver)),
            ArchGuard::ArchAudio(ver) => Some(format!("Requires: V{}, Audio", ver)),
        }
    }

    /// Returns a `#[cfg_attr(target_arch = "hexagon", target_feature(enable = "..."))]`
    /// attribute string, or None for base intrinsics that have no user-facing feature gate.
    fn target_feature_attr(&self) -> Option<String> {
        match self {
            ArchGuard::None => None,
            ArchGuard::Arch(ver) => Some(format!(
                "#[cfg_attr(target_arch = \"hexagon\", target_feature(enable = \"v{}\"))]",
                ver
            )),
            ArchGuard::ArchAudio(ver) => Some(format!(
                "#[cfg_attr(target_arch = \"hexagon\", target_feature(enable = \"v{},audio\"))]",
                ver
            )),
        }
    }
}

/// Parsed scalar intrinsic information
#[derive(Debug, Clone)]
struct ScalarIntrinsic {
    /// Q6 name (e.g., "Q6_R_add_RR")
    q6_name: String,
    /// Builtin suffix (e.g., "A2_add") - from __builtin_HEXAGON_A2_add
    builtin_name: String,
    /// Assembly syntax
    asm_syntax: String,
    /// Instruction type
    instr_type: String,
    /// Execution slots
    exec_slots: String,
    /// Return type
    return_type: RustType,
    /// Parameters (name, type, optional immediate info)
    params: Vec<(String, RustType, Option<ImmediateInfo>)>,
    /// Architecture guard
    arch_guard: ArchGuard,
}

impl ScalarIntrinsic {
    /// Generate the LLVM link name: A2_add -> llvm.hexagon.A2.add
    fn llvm_link_name(&self) -> String {
        format!("llvm.hexagon.{}", self.builtin_name.replace('_', "."))
    }

    /// Generate the Rust function name: Q6_R_add_RR -> Q6_R_add_RR
    ///
    /// We preserve the original case because the Q6 naming convention uses
    /// case to distinguish register types:
    /// - `P` (uppercase) = 64-bit register pair (Word64)
    /// - `p` (lowercase) = predicate register (Byte)
    fn rust_fn_name(&self) -> String {
        self.q6_name.clone()
    }

    /// Generate the extern function name: A2_add -> hexagon_A2_add
    fn extern_fn_name(&self) -> String {
        format!("hexagon_{}", self.builtin_name)
    }
}

/// Read the local header file
fn read_header(crate_dir: &Path) -> Result<String, String> {
    let header_path = crate_dir.join(HEADER_FILE);
    println!("Reading scalar header from: {}", header_path.display());
    println!("  (LLVM version: {})", LLVM_VERSION);

    std::fs::read_to_string(&header_path).map_err(|e| {
        format!(
            "Failed to read header file {}: {}",
            header_path.display(),
            e
        )
    })
}

/// Detect whether a C parameter name represents an immediate operand.
///
/// C prototype parameter names follow the pattern `I[usUS]\d+` for immediates:
/// - `Is16` → signed 16-bit
/// - `Iu5` → unsigned 5-bit
/// - `IS8` → signed 8-bit (secondary)
/// - `IU5` → unsigned 5-bit (secondary)
/// - `Iu6_2` → unsigned 6-bit (with alignment suffix)
fn detect_immediate(original_name: &str, imm_re: &Regex) -> Option<ImmediateInfo> {
    imm_re.captures(original_name).map(|caps| {
        let sign_char = &caps[1];
        let bits: u32 = caps[2].parse().unwrap();
        ImmediateInfo {
            signed: sign_char == "s" || sign_char == "S",
            bits,
        }
    })
}

/// Parse a C function prototype to extract return type and parameters
fn parse_prototype(
    prototype: &str,
    proto_re: &Regex,
    param_re: &Regex,
    imm_re: &Regex,
) -> Option<(RustType, Vec<(String, RustType, Option<ImmediateInfo>)>)> {
    if let Some(caps) = proto_re.captures(prototype) {
        let return_type_str = caps[1].trim();
        let params_str = &caps[2];

        // Skip if return type is unknown
        let return_type = RustType::from_c_type(return_type_str)?;

        let mut params = Vec::new();
        if !params_str.trim().is_empty() {
            let mut name_counts: HashMap<String, u32> = HashMap::new();
            for param in params_str.split(',') {
                let param = param.trim();
                if let Some(pcaps) = param_re.captures(param) {
                    let ptype_str = &pcaps[1];
                    let original_name = &pcaps[2];
                    let base_name = original_name.to_lowercase();

                    // Skip intrinsics with void* or void** params
                    if ptype_str.contains("void") {
                        return None;
                    }

                    if let Some(ptype) = RustType::from_c_type(ptype_str) {
                        // Detect immediate operands from the original C name
                        let imm_info = detect_immediate(original_name, imm_re);

                        // De-duplicate parameter names by appending a suffix
                        let count = name_counts.entry(base_name.clone()).or_insert(0);
                        *count += 1;
                        let pname = if *count > 1 {
                            format!("{}_{}", base_name, count)
                        } else {
                            base_name
                        };
                        params.push((pname, ptype, imm_info));
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

/// Parse the header file to extract all scalar intrinsics
fn parse_header(content: &str) -> Vec<ScalarIntrinsic> {
    let mut intrinsics = Vec::new();

    // Pre-compile all regexes once
    let arch_guard_re = Regex::new(r"#if __HEXAGON_ARCH__ >= (\d+)(.*)").unwrap();
    let q6_define_re = Regex::new(r"#define\s+(Q6_\w+)\s+__builtin_HEXAGON_(\w+)").unwrap();
    let proto_re = Regex::new(r"(\w+)\s+Q6_\w+\(([^)]*)\)").unwrap();
    let param_re = Regex::new(r"(\w+\*{0,2})\s+(\w+)").unwrap();
    let imm_re = Regex::new(r"^I([uUsS])(\d+)").unwrap();

    let lines: Vec<&str> = content.lines().collect();
    let mut current_guard = ArchGuard::None;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Track #if guards
        if let Some(caps) = arch_guard_re.captures(line) {
            let arch_ver: u32 = caps[1].parse().unwrap_or(0);
            let rest = &caps[2];
            if rest.contains("__HEXAGON_AUDIO__") {
                current_guard = ArchGuard::ArchAudio(arch_ver);
            } else {
                current_guard = ArchGuard::Arch(arch_ver);
            }
            i += 1;
            continue;
        }

        // Track #endif to reset guard
        if line.starts_with("#endif")
            && !line.contains("__HEXAGON_PROTOS_H_")
            && !line.contains("__HVX__")
        {
            current_guard = ArchGuard::None;
            i += 1;
            continue;
        }

        // Look for comment blocks with Assembly Syntax
        if line.contains("Assembly Syntax:") {
            let mut asm_syntax = String::new();
            let mut prototype = String::new();
            let mut instr_type = String::new();
            let mut exec_slots = String::new();

            // Parse the comment block
            let mut j = i;
            while j < lines.len() && !lines[j].trim().starts_with("#define") {
                let cline = lines[j];
                if cline.contains("Assembly Syntax:") {
                    if let Some(pos) = cline.find("Assembly Syntax:") {
                        asm_syntax = cline[pos + 16..].trim().to_string();
                    }
                } else if cline.contains("C Intrinsic Prototype:") {
                    if let Some(pos) = cline.find("C Intrinsic Prototype:") {
                        prototype = cline[pos + 22..].trim().to_string();
                    }
                } else if cline.contains("Instruction Type:") {
                    if let Some(pos) = cline.find("Instruction Type:") {
                        instr_type = cline[pos + 17..].trim().to_string();
                    }
                } else if cline.contains("Execution Slots:") {
                    if let Some(pos) = cline.find("Execution Slots:") {
                        exec_slots = cline[pos + 16..].trim().to_string();
                    }
                }
                j += 1;
            }

            // Find the #define line
            while j < lines.len() && !lines[j].trim().starts_with("#define") {
                j += 1;
            }

            if j < lines.len() {
                let define_line = lines[j];

                if let Some(caps) = q6_define_re.captures(define_line) {
                    let q6_name = caps[1].to_string();
                    let builtin_name = caps[2].to_string();

                    // Parse the C prototype
                    if let Some((return_type, params)) =
                        parse_prototype(&prototype, &proto_re, &param_re, &imm_re)
                    {
                        intrinsics.push(ScalarIntrinsic {
                            q6_name,
                            builtin_name,
                            asm_syntax,
                            instr_type,
                            exec_slots,
                            return_type,
                            params,
                            arch_guard: current_guard.clone(),
                        });
                    }
                }
            }
            i = j + 1;
            continue;
        }

        i += 1;
    }

    intrinsics
}

/// Generate the module documentation
fn generate_module_doc() -> String {
    r#"//! Hexagon scalar intrinsics
//!
//! This module provides intrinsics for scalar (non-HVX) Hexagon DSP operations,
//! including arithmetic, multiply, shift, saturate, compare, and floating-point
//! operations.
//!
//! [Hexagon V68 Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-45)
//!
//! ## Naming Convention
//!
//! Function names preserve the original Q6 naming case because the convention
//! uses case to distinguish register types:
//! - `P` (uppercase) = 64-bit register pair (`Word64`)
//! - `p` (lowercase) = predicate register (`Byte`)
//!
//! For example, `Q6_P_and_PP` operates on 64-bit pairs while `Q6_p_and_pp`
//! operates on predicate registers.
//!
//! ## Architecture Versions
//!
//! Most scalar intrinsics are available on all Hexagon architectures.
//! Some intrinsics require specific architecture versions (v60, v62, v65,
//! v66, v67, v68, or v67+audio) and carry
//! `#[target_feature(enable = "v68")]` (or the appropriate version).
//! Enable these with `-C target-feature=+v68` or by setting the target CPU
//! via `-C target-cpu=hexagonv68`.
//!
//! Each version includes all features from previous versions.

#![allow(non_snake_case)]

#[cfg(test)]
use stdarch_test::assert_instr;
"#
    .to_string()
}

/// Generate the extern block with LLVM intrinsic declarations
fn generate_extern_block(intrinsics: &[ScalarIntrinsic]) -> String {
    let mut output = String::new();

    output.push_str("// LLVM intrinsic declarations for Hexagon scalar operations\n");
    output.push_str("#[allow(improper_ctypes)]\n");
    output.push_str("unsafe extern \"unadjusted\" {\n");

    for info in intrinsics {
        let link_name = info.llvm_link_name();
        let fn_name = info.extern_fn_name();

        let params_str = if info.params.is_empty() {
            String::new()
        } else {
            info.params
                .iter()
                .map(|(_, t, _)| format!("_: {}", t.to_rust_str()))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let return_str = if info.return_type == RustType::Unit {
            String::new()
        } else {
            format!(" -> {}", info.return_type.to_rust_str())
        };

        output.push_str(&format!(
            "    #[link_name = \"{}\"]\n    fn {}({}){return_str};\n",
            link_name, fn_name, params_str
        ));
    }

    output.push_str("}\n");
    output
}

/// Generate wrapper functions for all intrinsics
fn generate_functions(intrinsics: &[ScalarIntrinsic]) -> String {
    let mut output = String::new();

    for info in intrinsics {
        let rust_name = info.rust_fn_name();
        let extern_name = info.extern_fn_name();

        // Collect immediate parameter info: (original_index, const_name, ImmediateInfo)
        let imm_params: Vec<(usize, String, &ImmediateInfo)> = info
            .params
            .iter()
            .enumerate()
            .filter_map(|(i, (name, _, imm))| imm.as_ref().map(|im| (i, name.to_uppercase(), im)))
            .collect();

        // Doc comment
        output.push_str(&format!("/// `{}`\n", info.asm_syntax));
        output.push_str("///\n");
        output.push_str(&format!("/// Instruction Type: {}\n", info.instr_type));
        output.push_str(&format!("/// Execution Slots: {}\n", info.exec_slots));
        if let Some(req) = info.arch_guard.requires_doc() {
            output.push_str(&format!("/// {}\n", req));
        }

        // Attributes
        output.push_str("#[inline(always)]\n");
        if let Some(tf_attr) = info.arch_guard.target_feature_attr() {
            output.push_str(&format!("{}\n", tf_attr));
        }

        // Immediate parameters become const generics but are passed as positional
        // arguments at the call site: Q6_R_add_RI(rs, 42) rather than Q6_R_add_RI::<42>(rs).
        // This matches the assembly syntax where the immediate is an operand.
        if !imm_params.is_empty() {
            let indices: Vec<String> = imm_params.iter().map(|(i, _, _)| i.to_string()).collect();
            output.push_str(&format!(
                "#[rustc_legacy_const_generics({})]\n",
                indices.join(", ")
            ));
        }

        if let Some(instr) = extract_instr_name(&info.asm_syntax) {
            if imm_params.is_empty() {
                output.push_str(&format!("#[cfg_attr(test, assert_instr({}))]\n", instr));
            } else {
                // Provide default values for const generics in assert_instr
                let defaults: Vec<String> = imm_params
                    .iter()
                    .map(|(_, name, _)| format!("{} = 0", name))
                    .collect();
                output.push_str(&format!(
                    "#[cfg_attr(test, assert_instr({}, {}))]\n",
                    instr,
                    defaults.join(", ")
                ));
            }
        }

        output.push_str(&format!(
            "#[unstable(feature = \"stdarch_hexagon\", issue = \"{}\")]\n",
            TRACKING_ISSUE
        ));

        // Function signature: regular params exclude immediates, const generics added
        let regular_params_str = info
            .params
            .iter()
            .filter(|(_, _, imm)| imm.is_none())
            .map(|(name, ty, _)| format!("{}: {}", name, ty.to_rust_str()))
            .collect::<Vec<_>>()
            .join(", ");

        let return_str = if info.return_type == RustType::Unit {
            String::new()
        } else {
            format!(" -> {}", info.return_type.to_rust_str())
        };

        if imm_params.is_empty() {
            output.push_str(&format!(
                "pub unsafe fn {}({}){} {{\n",
                rust_name, regular_params_str, return_str
            ));
        } else {
            let const_generics: Vec<String> = imm_params
                .iter()
                .map(|(_, name, imm)| {
                    let ty = if imm.signed { "i32" } else { "u32" };
                    format!("const {}: {}", name, ty)
                })
                .collect();
            output.push_str(&format!(
                "pub unsafe fn {}<{}>({}){} {{\n",
                rust_name,
                const_generics.join(", "),
                regular_params_str,
                return_str
            ));
        }

        // Function body: static assertions then call
        for (_, const_name, imm_info) in &imm_params {
            if imm_info.signed {
                output.push_str(&format!(
                    "    static_assert_simm_bits!({}, {});\n",
                    const_name, imm_info.bits
                ));
            } else {
                output.push_str(&format!(
                    "    static_assert_uimm_bits!({}, {});\n",
                    const_name, imm_info.bits
                ));
            }
        }

        // Call args: use original order, using const generic names for immediates.
        // Unsigned const generics (u32) need a cast to i32 for the extern call.
        let args_str = info
            .params
            .iter()
            .map(|(name, _, imm)| match imm {
                Some(info) if !info.signed => format!("{} as i32", name.to_uppercase()),
                Some(_) => name.to_uppercase(),
                None => name.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ");

        output.push_str(&format!("    {}({})\n", extern_name, args_str));
        output.push_str("}\n\n");
    }

    output
}

/// Generate the complete scalar.rs file
fn generate_scalar_file(intrinsics: &[ScalarIntrinsic], output_path: &Path) -> Result<(), String> {
    let mut output =
        File::create(output_path).map_err(|e| format!("Failed to create output: {}", e))?;

    writeln!(output, "{}", generate_module_doc()).map_err(|e| e.to_string())?;
    writeln!(output, "").map_err(|e| e.to_string())?;
    writeln!(output, "{}", generate_extern_block(intrinsics)).map_err(|e| e.to_string())?;
    writeln!(output, "{}", generate_functions(intrinsics)).map_err(|e| e.to_string())?;

    // Flush before running rustfmt
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
    println!("=== Hexagon Scalar Code Generator ===\n");

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    let header_content = read_header(&crate_dir)?;
    println!("Read {} bytes", header_content.len());

    let intrinsics = parse_header(&header_content);
    println!("Parsed {} scalar intrinsics", intrinsics.len());

    let hexagon_dir = crate_dir.join("../core_arch/src/hexagon");
    let scalar_path = hexagon_dir.join("scalar.rs");

    generate_scalar_file(&intrinsics, &scalar_path)?;
    println!("Generated scalar.rs at {}", scalar_path.display());

    Ok(())
}
