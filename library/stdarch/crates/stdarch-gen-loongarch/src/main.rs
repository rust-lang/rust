use std::env;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::PathBuf;

/// Complete lines of generated source.
///
/// This enables common generation tasks to be factored out without precluding basic
/// context-specific formatting.
///
/// The convention in this generator is to prefix (not suffix) lines with a newline, so the
/// implementation of `std::fmt::Display` behaves in the same way.
struct Lines {
    indent: usize,
    lines: Vec<String>,
}

impl Lines {
    fn single(line: String) -> Self {
        Self::from(vec![line])
    }
}

impl From<Vec<String>> for Lines {
    fn from(lines: Vec<String>) -> Self {
        Self { indent: 0, lines }
    }
}

impl std::fmt::Display for Lines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        for line in self.lines.iter() {
            write!(f, "\n{:width$}{line}", "", width = self.indent)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TargetFeature {
    Lsx,
    Lasx,
}

impl TargetFeature {
    fn new(ext: &str) -> TargetFeature {
        match ext {
            "lasx" => Self::Lasx,
            _ => Self::Lsx,
        }
    }

    /// A string for use with `#[target_feature(...)]`.
    fn as_target_feature_arg(&self, ins: &str) -> String {
        let vec = match *self {
            // Features included with LoongArch64 LSX and LASX.
            Self::Lsx => "lsx",
            Self::Lasx => "lasx",
        };
        let frecipe = match ins {
            "lsx_vfrecipe_s" | "lsx_vfrecipe_d" | "lsx_vfrsqrte_s" | "lsx_vfrsqrte_d"
            | "lasx_xvfrecipe_s" | "lasx_xvfrecipe_d" | "lasx_xvfrsqrte_s" | "lasx_xvfrsqrte_d" => {
                ",frecipe"
            }
            _ => "",
        };
        format!("{vec}{frecipe}")
    }

    fn attr(name: &str, value: impl fmt::Display) -> String {
        format!(r#"#[{name}(enable = "{value}")]"#)
    }

    /// Generate a target_feature attribute
    fn to_target_feature_attr(&self, ins: &str) -> Lines {
        Lines::single(Self::attr(
            "target_feature",
            self.as_target_feature_arg(ins),
        ))
    }

    fn bytes(&self) -> u8 {
        match *self {
            // Features included with LoongArch64 LSX and LASX.
            Self::Lsx => 16,
            Self::Lasx => 32,
        }
    }
}

fn gen_spec(in_file: String, ext_name: &str) -> io::Result<()> {
    let f = File::open(in_file.clone()).unwrap_or_else(|_| panic!("Failed to open {in_file}"));
    let f = BufReader::new(f);
    let mut out = format!(
        r#"// This code is automatically generated. DO NOT MODIFY.
// ```
// OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- {in_file}
// ```
"#
    );
    out.push('\n');

    let mut asm_fmts = String::new();
    let mut data_types = String::new();
    let fn_pat = format!("__{ext_name}_");
    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }

        if let Some(s) = line.find("/* Assembly instruction format:") {
            let e = line.find('.').unwrap();
            asm_fmts = line.get(s + 31..e).unwrap().trim().to_string();
        } else if let Some(s) = line.find("/* Data types in instruction templates:") {
            let e = line.find('.').unwrap();
            data_types = line.get(s + 39..e).unwrap().trim().to_string();
        } else if let Some(s) = line.find(fn_pat.as_str()) {
            let e = line.find('(').unwrap();
            let name = line.get(s + 2..e).unwrap().trim().to_string();
            out.push_str(&format!("/// {name}\n"));
            out.push_str(&format!("name = {name}\n"));
            out.push_str(&format!("asm-fmts = {asm_fmts}\n"));
            out.push_str(&format!("data-types = {data_types}\n"));
            out.push('\n');
        }
    }

    let out_dir_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::create_dir_all(&out_dir_path)?;
    let mut f = File::create(out_dir_path.join(format!("{ext_name}.spec")))?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

fn gen_bind(in_file: String, ext_name: &str) -> io::Result<()> {
    let f = File::open(in_file.clone()).unwrap_or_else(|_| panic!("Failed to open {in_file}"));
    let f = BufReader::new(f);

    let target: TargetFeature = TargetFeature::new(ext_name);
    let mut para_num;
    let mut current_name: Option<String> = None;
    let mut asm_fmts: Vec<String> = Vec::new();
    let mut link_function_str = String::new();
    let mut function_str = String::new();
    let mut out = String::new();

    out.push_str(&format!(
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `{in_file}` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- {in_file}
// ```

use super::types::*;
"#
    ));

    out.push_str(
        r#"
#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
"#,
    );

    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix("name = ") {
            current_name = Some(String::from(name));
        } else if line.starts_with("asm-fmts = ") {
            asm_fmts = line[10..]
                .split(',')
                .map(|v| v.trim().to_string())
                .collect();
        } else if line.starts_with("data-types = ") {
            let current_name = current_name.clone().unwrap();
            let data_types: Vec<&str> = line
                .get(12..)
                .unwrap()
                .split(',')
                .map(|e| e.trim())
                .collect();
            let in_t;
            let out_t;
            if data_types.len() == 2 {
                in_t = [data_types[1], "NULL", "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 1;
            } else if data_types.len() == 3 {
                in_t = [data_types[1], data_types[2], "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 2;
            } else if data_types.len() == 4 {
                in_t = [data_types[1], data_types[2], data_types[3], "NULL"];
                out_t = data_types[0];
                para_num = 3;
            } else if data_types.len() == 5 {
                in_t = [data_types[1], data_types[2], data_types[3], data_types[4]];
                out_t = data_types[0];
                para_num = 4;
            } else {
                panic!("DEBUG: line: {0} len: {1}", line, data_types.len());
            }

            let (link_function, function) =
                gen_bind_body(&current_name, &asm_fmts, &in_t, out_t, para_num, target);
            link_function_str.push_str(&link_function);
            function_str.push_str(&function);
        }
    }
    out.push_str(&link_function_str);
    out.push_str("}\n");
    out.push_str(&function_str);

    let out_path: PathBuf =
        PathBuf::from(env::var("OUT_DIR").unwrap_or("crates/core_arch".to_string()))
            .join("src")
            .join("loongarch64")
            .join(ext_name);
    std::fs::create_dir_all(&out_path)?;

    let mut file = File::create(out_path.join("generated.rs"))?;
    file.write_all(out.as_bytes())?;
    Ok(())
}

fn gen_bind_body(
    current_name: &str,
    asm_fmts: &[String],
    in_t: &[&str; 4],
    out_t: &str,
    para_num: i32,
    target: TargetFeature,
) -> (String, String) {
    let type_to_rst = |t: &str, s: bool| -> &str {
        match (t, s) {
            ("V16QI", _) => "v16i8",
            ("V32QI", _) => "v32i8",
            ("V8HI", _) => "v8i16",
            ("V16HI", _) => "v16i16",
            ("V4SI", _) => "v4i32",
            ("V8SI", _) => "v8i32",
            ("V2DI", _) => "v2i64",
            ("V4DI", _) => "v4i64",
            ("UV16QI", _) => "v16u8",
            ("UV32QI", _) => "v32u8",
            ("UV8HI", _) => "v8u16",
            ("UV16HI", _) => "v16u16",
            ("UV4SI", _) => "v4u32",
            ("UV8SI", _) => "v8u32",
            ("UV2DI", _) => "v2u64",
            ("UV4DI", _) => "v4u64",
            ("SI", _) => "i32",
            ("DI", _) => "i64",
            ("USI", _) => "u32",
            ("UDI", _) => "u64",
            ("V4SF", _) => "v4f32",
            ("V8SF", _) => "v8f32",
            ("V2DF", _) => "v2f64",
            ("V4DF", _) => "v4f64",
            ("UQI", _) => "u32",
            ("QI", _) => "i32",
            ("CVPOINTER", false) => "*const i8",
            ("CVPOINTER", true) => "*mut i8",
            ("HI", _) => "i32",
            (_, _) => panic!("unknown type: {t}"),
        }
    };

    let is_store = current_name.to_string().contains("vst");
    let link_function = {
        let fn_decl = {
            let fn_output = if out_t.to_lowercase() == "void" {
                String::new()
            } else {
                format!("-> {}", type_to_rst(out_t, is_store))
            };
            let fn_inputs = match para_num {
                1 => format!("(a: {})", type_to_rst(in_t[0], is_store)),
                2 => format!(
                    "(a: {}, b: {})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store)
                ),
                3 => format!(
                    "(a: {}, b: {}, c: {})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store)
                ),
                4 => format!(
                    "(a: {}, b: {}, c: {}, d: {})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store),
                    type_to_rst(in_t[3], is_store)
                ),
                _ => panic!("unsupported parameter number"),
            };
            format!("fn __{current_name}{fn_inputs} {fn_output};")
        };
        let function = format!(
            r#"    #[link_name = "llvm.loongarch.{}"]
    {fn_decl}
"#,
            current_name.replace('_', ".")
        );
        function
    };

    let type_to_imm = |t| -> i8 {
        match t {
            'b' => 4,
            'h' => 3,
            'w' => 2,
            'd' => 1,
            _ => panic!("unsupported type"),
        }
    };
    let mut rustc_legacy_const_generics = "";
    let fn_decl = {
        let fn_output = if out_t.to_lowercase() == "void" {
            String::new()
        } else {
            format!("-> {} ", type_to_rst(out_t, is_store))
        };
        let mut fn_inputs = match para_num {
            1 => format!("(a: {})", type_to_rst(in_t[0], is_store)),
            2 => format!(
                "(a: {}, b: {})",
                type_to_rst(in_t[0], is_store),
                type_to_rst(in_t[1], is_store)
            ),
            3 => format!(
                "(a: {}, b: {}, c: {})",
                type_to_rst(in_t[0], is_store),
                type_to_rst(in_t[1], is_store),
                type_to_rst(in_t[2], is_store)
            ),
            4 => format!(
                "(a: {}, b: {}, c: {}, d: {})",
                type_to_rst(in_t[0], is_store),
                type_to_rst(in_t[1], is_store),
                type_to_rst(in_t[2], is_store),
                type_to_rst(in_t[3], is_store)
            ),
            _ => panic!("unsupported parameter number"),
        };
        if para_num == 1 && in_t[0] == "HI" {
            fn_inputs = match asm_fmts[1].as_str() {
                "si13" | "i13" => format!("<const IMM_S13: {}>()", type_to_rst(in_t[0], is_store)),
                "si10" => format!("<const IMM_S10: {}>()", type_to_rst(in_t[0], is_store)),
                _ => panic!("unsupported assembly format: {}", asm_fmts[1]),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(0)";
        } else if para_num == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
            fn_inputs = if asm_fmts[2].starts_with("ui") {
                format!(
                    "<const IMM{2}: {1}>(a: {0})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[1] == "QI" {
            fn_inputs = if asm_fmts[2].starts_with("si") {
                format!(
                    "<const IMM_S{2}: {1}>(a: {0})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
            fn_inputs = if asm_fmts[2].starts_with("si") {
                format!(
                    "<const IMM_S{2}: {1}>(mem_addr: {0})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "rk" => format!(
                    "(mem_addr: {}, b: {})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
        } else if para_num == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
            fn_inputs = if asm_fmts[2].starts_with("ui") {
                format!(
                    "<const IMM{3}: {2}>(a: {0}, b: {1})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2])
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2)";
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "si12" => format!(
                    "<const IMM_S12: {2}>(a: {0}, mem_addr: {1})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2)";
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "rk" => format!(
                    "(a: {}, mem_addr: {}, b: {})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
        } else if para_num == 4 {
            fn_inputs = match (asm_fmts[2].as_str(), current_name.chars().last().unwrap()) {
                ("si8", t) => format!(
                    "<const IMM_S8: {2}, const IMM{4}: {3}>(a: {0}, mem_addr: {1})",
                    type_to_rst(in_t[0], is_store),
                    type_to_rst(in_t[1], is_store),
                    type_to_rst(in_t[2], is_store),
                    type_to_rst(in_t[3], is_store),
                    type_to_imm(t),
                ),
                (_, _) => panic!(
                    "unsupported assembly format: {} for {}",
                    asm_fmts[2], current_name
                ),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2, 3)";
        }
        format!("pub unsafe fn {current_name}{fn_inputs} {fn_output}")
    };
    let mut call_params = {
        match para_num {
            1 => format!("__{current_name}(a)"),
            2 => format!("__{current_name}(a, b)"),
            3 => format!("__{current_name}(a, b, c)"),
            4 => format!("__{current_name}(a, b, c, d)"),
            _ => panic!("unsupported parameter number"),
        }
    };
    if para_num == 1 && in_t[0] == "HI" {
        call_params = match asm_fmts[1].as_str() {
            "si10" => {
                format!("static_assert_simm_bits!(IMM_S10, 10);\n    __{current_name}(IMM_S10)")
            }
            "i13" => {
                format!("static_assert_simm_bits!(IMM_S13, 13);\n    __{current_name}(IMM_S13)")
            }
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        }
    } else if para_num == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
        call_params = if asm_fmts[2].starts_with("ui") {
            format!(
                "static_assert_uimm_bits!(IMM{0}, {0});\n    __{current_name}(a, IMM{0})",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        };
    } else if para_num == 2 && in_t[1] == "QI" {
        call_params = match asm_fmts[2].as_str() {
            "si5" => {
                format!("static_assert_simm_bits!(IMM_S5, 5);\n    __{current_name}(a, IMM_S5)")
            }
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
        call_params = if asm_fmts[2].starts_with("si") {
            format!(
                "static_assert_simm_bits!(IMM_S{0}, {0});\n    __{current_name}(mem_addr, IMM_S{0})",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        }
    } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
        call_params = match asm_fmts[2].as_str() {
            "rk" => format!("__{current_name}(mem_addr, b)"),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
        call_params = if asm_fmts[2].starts_with("ui") {
            format!(
                "static_assert_uimm_bits!(IMM{0}, {0});\n    __{current_name}(a, b, IMM{0})",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        }
    } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
        call_params = match asm_fmts[2].as_str() {
            "si12" => format!(
                "static_assert_simm_bits!(IMM_S12, 12);\n    __{current_name}(a, mem_addr, IMM_S12)"
            ),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
        call_params = match asm_fmts[2].as_str() {
            "rk" => format!("__{current_name}(a, mem_addr, b)"),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 4 {
        call_params = match (asm_fmts[2].as_str(), current_name.chars().last().unwrap()) {
            ("si8", t) => format!(
                "static_assert_simm_bits!(IMM_S8, 8);\n    static_assert_uimm_bits!(IMM{0}, {0});\n    __{current_name}(a, mem_addr, IMM_S8, IMM{0})",
                type_to_imm(t)
            ),
            (_, _) => panic!(
                "unsupported assembly format: {} for {}",
                asm_fmts[2], current_name
            ),
        }
    }
    let function = if !rustc_legacy_const_generics.is_empty() {
        format!(
            r#"
#[inline]{target_feature}
#[{rustc_legacy_const_generics}]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
{fn_decl}{{
    {call_params}
}}
"#,
            target_feature = target.to_target_feature_attr(current_name)
        )
    } else {
        format!(
            r#"
#[inline]{target_feature}
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
{fn_decl}{{
    {call_params}
}}
"#,
            target_feature = target.to_target_feature_attr(current_name)
        )
    };
    (link_function, function)
}

fn gen_test(in_file: String, ext_name: &str) -> io::Result<()> {
    let f = File::open(in_file.clone()).unwrap_or_else(|_| panic!("Failed to open {in_file}"));
    let f = BufReader::new(f);

    let target: TargetFeature = TargetFeature::new(ext_name);
    let mut para_num;
    let mut current_name: Option<String> = None;
    let mut asm_fmts: Vec<String> = Vec::new();
    let mut impl_function_str = String::new();
    let mut call_function_str = String::new();
    let mut out = String::new();

    out.push_str(&format!(
        r#"/*
 * This code is automatically generated. DO NOT MODIFY.
 *
 * Instead, modify `{in_file}` and run the following command to re-generate this file:
 *
 * ```
 * OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- {in_file} test
 * ```
 */

#include <stdio.h>
#include <stdint.h>
#include <lsxintrin.h>
#include <lasxintrin.h>

union v16qi
{{
    __m128i v;
    int64_t i64[2];
    int8_t i8[16];
}};

union v32qi
{{
    __m256i v;
    int64_t i64[4];
    int8_t i8[32];
}};

union v8hi
{{
    __m128i v;
    int64_t i64[2];
    int16_t i16[8];
}};

union v16hi
{{
    __m256i v;
    int64_t i64[4];
    int16_t i16[16];
}};

union v4si
{{
    __m128i v;
    int64_t i64[2];
    int32_t i32[4];
}};

union v8si
{{
    __m256i v;
    int64_t i64[4];
    int32_t i32[8];
}};

union v2di
{{
    __m128i v;
    int64_t i64[2];
}};

union v4di
{{
    __m256i v;
    int64_t i64[4];
}};

union uv16qi
{{
    __m128i v;
    uint64_t i64[2];
    uint8_t i8[16];
}};

union uv32qi
{{
    __m256i v;
    uint64_t i64[4];
    uint8_t i8[32];
}};

union uv8hi
{{
    __m128i v;
    uint64_t i64[2];
    uint16_t i16[8];
}};

union uv16hi
{{
    __m256i v;
    uint64_t i64[4];
    uint16_t i16[16];
}};

union uv4si
{{
    __m128i v;
    uint64_t i64[2];
    uint32_t i32[4];
}};

union uv8si
{{
    __m256i v;
    uint64_t i64[4];
    uint32_t i32[8];
}};

union uv2di
{{
    __m128i v;
    uint64_t i64[2];
}};

union uv4di
{{
    __m256i v;
    uint64_t i64[4];
}};

union v4sf
{{
    __m128 v;
    int64_t i64[2];
    uint32_t i32[2];
    float f32[4];
}};

union v8sf
{{
    __m256 v;
    int64_t i64[4];
    uint32_t i32[4];
    float f32[8];
}};

union v2df
{{
    __m128d v;
    uint64_t i64[2];
    double f64[2];
}};

union v4df
{{
    __m256d v;
    uint64_t i64[4];
    double f64[4];
}};
"#
    ));

    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix("name = ") {
            current_name = Some(String::from(name));
        } else if line.starts_with("asm-fmts = ") {
            asm_fmts = line[10..]
                .split(',')
                .map(|v| v.trim().to_string())
                .collect();
        } else if line.starts_with("data-types = ") {
            let current_name = current_name.clone().unwrap();
            let data_types: Vec<&str> = line
                .get(12..)
                .unwrap()
                .split(',')
                .map(|e| e.trim())
                .collect();
            let in_t;
            let out_t;
            if data_types.len() == 2 {
                in_t = [data_types[1], "NULL", "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 1;
            } else if data_types.len() == 3 {
                in_t = [data_types[1], data_types[2], "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 2;
            } else if data_types.len() == 4 {
                in_t = [data_types[1], data_types[2], data_types[3], "NULL"];
                out_t = data_types[0];
                para_num = 3;
            } else if data_types.len() == 5 {
                in_t = [data_types[1], data_types[2], data_types[3], data_types[4]];
                out_t = data_types[0];
                para_num = 4;
            } else {
                panic!("DEBUG: line: {0} len: {1}", line, data_types.len());
            }

            let (link_function, function) =
                gen_test_body(&current_name, &asm_fmts, &in_t, out_t, para_num, target);
            impl_function_str.push_str(&link_function);
            call_function_str.push_str(&function);
        }
    }
    out.push_str(&impl_function_str);
    out.push('\n');
    out.push_str("int main(int argc, char *argv[])\n");
    out.push_str("{\n");
    out.push_str("    printf(\"// This code is automatically generated. DO NOT MODIFY.\\n\");\n");
    out.push_str("    printf(\"// See crates/stdarch-gen-loongarch/README.md\\n\\n\");\n");
    out.push_str("    printf(\"use crate::{\\n\");\n");
    out.push_str("    printf(\"    core_arch::{loongarch64::*, simd::*},\\n\");\n");
    out.push_str("    printf(\"    mem::transmute,\\n\");\n");
    out.push_str("    printf(\"};\\n\");\n");
    out.push_str("    printf(\"use stdarch_test::simd_test;\\n\");\n");
    out.push_str(&call_function_str);
    out.push_str("    return 0;\n");
    out.push('}');

    let out_dir_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::create_dir_all(&out_dir_path)?;
    let mut f = File::create(out_dir_path.join(format!("{ext_name}.c")))?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

fn gen_test_body(
    current_name: &str,
    asm_fmts: &[String],
    in_t: &[&str; 4],
    out_t: &str,
    para_num: i32,
    target: TargetFeature,
) -> (String, String) {
    let rand_i32 = |bits: u8| -> i32 {
        let val = rand::random::<i32>();
        let bits = 32 - bits;
        (val << bits) >> bits
    };
    let rand_u32 = |bits: u8| -> u32 {
        let val = rand::random::<u32>();
        let bits = 32 - bits;
        (val << bits) >> bits
    };
    let rand_i64 = || -> i64 { rand::random::<i64>() };
    let rand_u64 = || -> u64 { rand::random::<u64>() };
    let rand_f32 = || -> f32 { rand::random::<f32>() };
    let rand_f64 = || -> f64 { rand::random::<f64>() };
    let type_to_ct = |t: &str| -> &str {
        match t {
            "V16QI" => "union v16qi",
            "V32QI" => "union v32qi",
            "V8HI" => "union v8hi",
            "V16HI" => "union v16hi",
            "V4SI" => "union v4si",
            "V8SI" => "union v8si",
            "V2DI" => "union v2di",
            "V4DI" => "union v4di",
            "UV16QI" => "union uv16qi",
            "UV32QI" => "union uv32qi",
            "UV8HI" => "union uv8hi",
            "UV16HI" => "union uv16hi",
            "UV4SI" => "union uv4si",
            "UV8SI" => "union uv8si",
            "UV2DI" => "union uv2di",
            "UV4DI" => "union uv4di",
            "SI" => "int32_t",
            "DI" => "int64_t",
            "USI" => "uint32_t",
            "UDI" => "uint64_t",
            "V4SF" => "union v4sf",
            "V8SF" => "union v8sf",
            "V2DF" => "union v2df",
            "V4DF" => "union v4df",
            "UQI" => "uint32_t",
            "QI" => "int32_t",
            "CVPOINTER" => "void*",
            "HI" => "int32_t",
            _ => panic!("unknown type: {t}"),
        }
    };
    let type_to_va = |v: &str, t: &str| -> String {
        let n = if v.starts_with('_') {
            v.get(1..).unwrap()
        } else {
            v
        };
        let mut out = String::new();
        match t {
            "A16QI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n}: [i8; 16] = [%d"));
                for _ in 1..16 {
                    out.push_str(", %d");
                }
                out.push_str(&format!("];\\n\",\n    {v}.i8[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "AM16QI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let mut {n}: [i8; 16] = [%d"));
                for _ in 1..16 {
                    out.push_str(", %d");
                }
                out.push_str(&format!("];\\n\",\n    {v}.i8[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "V16QI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i8x16::new(%d"));
                for _ in 1..16 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i8[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "V32QI" => {
                for i in 0..32 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i8x32::new(%d"));
                for _ in 1..32 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i8[0]"));
                for i in 1..32 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "A32QI" => {
                for i in 0..32 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n}: [i8; 32] = [%d"));
                for _ in 1..32 {
                    out.push_str(", %d");
                }
                out.push_str(&format!("];\\n\",\n    {v}.i8[0]"));
                for i in 1..32 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "AM32QI" => {
                for i in 0..32 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_i32(8)));
                }
                out.push_str(&format!("    printf(\"    let mut {n}: [i8; 32] = [%d"));
                for _ in 1..32 {
                    out.push_str(", %d");
                }
                out.push_str(&format!("];\\n\",\n    {v}.i8[0]"));
                for i in 1..32 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "V8HI" => {
                for i in 0..8 {
                    out.push_str(&format!("    {v}.i16[{i}] = {};\n", rand_i32(16)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i16x8::new(%d"));
                for _ in 1..8 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i16[0]"));
                for i in 1..8 {
                    out.push_str(&format!(", {v}.i16[{i}]"));
                }
            }
            "V16HI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i16[{i}] = {};\n", rand_i32(16)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i16x16::new(%d"));
                for _ in 1..16 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i16[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i16[{i}]"));
                }
            }
            "V4SI" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.i32[{i}] = {};\n", rand_i32(32)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i32x4::new(%d"));
                for _ in 1..4 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "V8SI" => {
                for i in 0..8 {
                    out.push_str(&format!("    {v}.i32[{i}] = {};\n", rand_i32(32)));
                }
                out.push_str(&format!("    printf(\"    let {n} = i32x8::new(%d"));
                for _ in 1..8 {
                    out.push_str(", %d");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..8 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "V2DI" => {
                for i in 0..2 {
                    out.push_str(&format!("    {v}.i64[{i}] = {}L;\n", rand_i64()));
                }
                out.push_str(&format!("    printf(\"    let {n} = i64x2::new(%ld"));
                for _ in 1..2 {
                    out.push_str(", %ld");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..2 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "V4DI" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.i64[{i}] = {}L;\n", rand_i64()));
                }
                out.push_str(&format!("    printf(\"    let {n} = i64x4::new(%ld"));
                for _ in 1..4 {
                    out.push_str(", %ld");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "UV16QI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_u32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u8x16::new(%u"));
                for _ in 1..16 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i8[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "UV32QI" => {
                for i in 0..32 {
                    out.push_str(&format!("    {v}.i8[{i}] = {};\n", rand_u32(8)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u8x32::new(%u"));
                for _ in 1..32 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i8[0]"));
                for i in 1..32 {
                    out.push_str(&format!(", {v}.i8[{i}]"));
                }
            }
            "UV8HI" => {
                for i in 0..8 {
                    out.push_str(&format!("    {v}.i16[{i}] = {};\n", rand_u32(16)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u16x8::new(%u"));
                for _ in 1..8 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i16[0]"));
                for i in 1..8 {
                    out.push_str(&format!(", {v}.i16[{i}]"));
                }
            }
            "UV16HI" => {
                for i in 0..16 {
                    out.push_str(&format!("    {v}.i16[{i}] = {};\n", rand_u32(16)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u16x16::new(%u"));
                for _ in 1..16 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i16[0]"));
                for i in 1..16 {
                    out.push_str(&format!(", {v}.i16[{i}]"));
                }
            }
            "UV4SI" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.i32[{i}] = {};\n", rand_u32(32)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u32x4::new(%u"));
                for _ in 1..4 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "UV8SI" => {
                for i in 0..8 {
                    out.push_str(&format!("    {v}.i32[{i}] = {};\n", rand_u32(32)));
                }
                out.push_str(&format!("    printf(\"    let {n} = u32x8::new(%u"));
                for _ in 1..8 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..8 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "UV2DI" => {
                for i in 0..2 {
                    out.push_str(&format!("    {v}.i64[{i}] = {}UL;\n", rand_u64()));
                }
                out.push_str(&format!("    printf(\"    let {n} = u64x2::new(%lu"));
                for _ in 1..2 {
                    out.push_str(", %lu");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..2 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "UV4DI" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.i64[{i}] = {}UL;\n", rand_u64()));
                }
                out.push_str(&format!("    printf(\"    let {n} = u64x4::new(%lu"));
                for _ in 1..4 {
                    out.push_str(", %lu");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "V4SF" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.f32[{i}] = {};\n", rand_f32()));
                }
                out.push_str(&format!("    printf(\"    let {n} = u32x4::new(%u"));
                for _ in 1..4 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "V8SF" => {
                for i in 0..8 {
                    out.push_str(&format!("    {v}.f32[{i}] = {};\n", rand_f32()));
                }
                out.push_str(&format!("    printf(\"    let {n} = u32x8::new(%u"));
                for _ in 1..8 {
                    out.push_str(", %u");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i32[0]"));
                for i in 1..8 {
                    out.push_str(&format!(", {v}.i32[{i}]"));
                }
            }
            "V2DF" => {
                for i in 0..2 {
                    out.push_str(&format!("    {v}.f64[{i}] = {};\n", rand_f64()));
                }
                out.push_str(&format!("    printf(\"    let {n} = u64x2::new(%lu"));
                for _ in 1..2 {
                    out.push_str(", %lu");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..2 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "V4DF" => {
                for i in 0..4 {
                    out.push_str(&format!("    {v}.f64[{i}] = {};\n", rand_f64()));
                }
                out.push_str(&format!("    printf(\"    let {n} =  u64x4::new(%lu"));
                for _ in 1..4 {
                    out.push_str(", %lu");
                }
                out.push_str(&format!(");\\n\",\n    {v}.i64[0]"));
                for i in 1..4 {
                    out.push_str(&format!(", {v}.i64[{i}]"));
                }
            }
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "CVPOINTER" | "HI" => (),
            _ => panic!("unknown type: {t}"),
        }
        if !out.is_empty() {
            out.push_str(");");
        }
        out
    };
    let type_to_rp = |t: &str| -> &str {
        match t {
            "SI" => "    printf(\"    let r: i32 = %d;\\n\", o);",
            "DI" => "    printf(\"    let r: i64 = %ld;\\n\", o);",
            "USI" => "    printf(\"    let r: u32 = %u;\\n\", o);",
            "UDI" => "    printf(\"    let r: u64 = %lu;\\n\", o);",
            "UQI" => "    printf(\"    let r: u32 = %u;\\n\", o);",
            "QI" => "    printf(\"    let r: i32 = %d;\\n\", o);",
            "HI" => "    printf(\"    let r: i32 = %d;\\n\", o);",
            "V32QI" | "V16HI" | "V8SI" | "V4DI" | "UV32QI" | "UV16HI" | "UV8SI" | "UV4DI"
            | "V8SF" | "V4DF" => {
                "    printf(\"    let r = i64x4::new(%ld, %ld, %ld, %ld);\\n\", o.i64[0], o.i64[1], o.i64[2], o.i64[3]);"
            }
            _ => "    printf(\"    let r = i64x2::new(%ld, %ld);\\n\", o.i64[0], o.i64[1]);",
        }
    };
    let type_to_rx = |t: &str| -> &str {
        match t {
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "HI" => "o",
            _ => "o.v",
        }
    };
    let type_to_imm = |t| -> i8 {
        match t {
            'b' => 4,
            'h' => 3,
            'w' => 2,
            'd' => 1,
            _ => panic!("unsupported type"),
        }
    };

    let impl_function = {
        let fn_output = if out_t.to_lowercase() == "void" {
            String::new()
        } else {
            format!("    {} o;", type_to_ct(out_t))
        };
        let mut fn_inputs = match para_num {
            1 => format!(
                "    {} a;\n{}",
                type_to_ct(in_t[0]),
                type_to_va("a", in_t[0])
            ),
            2 => format!(
                "    {} a;\n{}\n    {} b;\n{}",
                type_to_ct(in_t[0]),
                type_to_va("a", in_t[0]),
                type_to_ct(in_t[1]),
                type_to_va("b", in_t[1])
            ),
            3 => format!(
                "    {} a;\n{}\n    {} b;\n{}\n    {} c;\n{}",
                type_to_ct(in_t[0]),
                type_to_va("a", in_t[0]),
                type_to_ct(in_t[1]),
                type_to_va("b", in_t[1]),
                type_to_ct(in_t[2]),
                type_to_va("c", in_t[2])
            ),
            4 => format!(
                "    {} a;\n{}\n    {} b;\n{}\n    {} c;\n{}\n    {} d;\n{}",
                type_to_ct(in_t[0]),
                type_to_va("a", in_t[0]),
                type_to_ct(in_t[1]),
                type_to_va("b", in_t[1]),
                type_to_ct(in_t[2]),
                type_to_va("c", in_t[2]),
                type_to_ct(in_t[3]),
                type_to_va("d", in_t[3])
            ),
            _ => panic!("unsupported parameter number"),
        };
        let mut fn_params = match para_num {
            1 => "(a.v)".to_string(),
            2 => "(a.v, b.v)".to_string(),
            3 => "(a.v, b.v, c.v)".to_string(),
            4 => "(a.v, b.v, c.v, d.v)".to_string(),
            _ => "unsupported parameter number".to_string(),
        };
        let mut as_params = match para_num {
            1 => "(transmute(a))".to_string(),
            2 => "(transmute(a), transmute(b))".to_string(),
            3 => "(transmute(a), transmute(b), transmute(c))".to_string(),
            4 => "(transmute(a), transmute(b), transmute(c), transmute(d))".to_string(),
            _ => panic!("unsupported parameter number"),
        };
        let mut as_args = String::new();
        if para_num == 1 && in_t[0] == "HI" {
            fn_inputs = "".to_string();
            match asm_fmts[1].as_str() {
                "si13" => {
                    let val = rand_i32(13);
                    fn_params = format!("({val})");
                    as_params = format!("::<{val}>()");
                }
                "i13" => {
                    let val = rand_u32(12);
                    fn_params = format!("({val})");
                    as_params = format!("::<{val}>()");
                }
                "si10" => {
                    let val = rand_i32(10);
                    fn_params = format!("({val})");
                    as_params = format!("::<{val}>()");
                }
                _ => panic!("unsupported assembly format: {}", asm_fmts[1]),
            }
        } else if para_num == 1
            && (in_t[0] == "SI" || in_t[0] == "DI")
            && asm_fmts[1].starts_with("rj")
        {
            fn_params = "(a)".to_string();
            if in_t[0] == "SI" {
                as_params = "(%d)".to_string();
            } else {
                as_params = "(%ld)".to_string();
            }
            as_args = ", a".to_string();
        } else if para_num == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
            if asm_fmts[2].starts_with("ui") {
                fn_inputs = format!(
                    "    {} a;\n{}",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0])
                );
                let val = rand_u32(asm_fmts[2].get(2..).unwrap().parse::<u8>().unwrap());
                fn_params = format!("(a.v, {val})");
                as_params = format!("::<{val}>(transmute(a))");
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 2 && in_t[1] == "QI" {
            if asm_fmts[2].starts_with("si") {
                fn_inputs = format!(
                    "    {} a;\n{}",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0])
                );
                let val = rand_i32(asm_fmts[2].get(2..).unwrap().parse::<u8>().unwrap());
                fn_params = format!("(a.v, {val})");
                as_params = format!("::<{val}>(transmute(a))");
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 2 && in_t[1] == "SI" && asm_fmts[2].starts_with("rk") {
            fn_params = "(a.v, b)".to_string();
            as_params = "(transmute(a), %d)".to_string();
            as_args = ", b".to_string();
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
            if asm_fmts[2].starts_with("si") {
                fn_inputs = format!(
                    "    union v{}qi _a;\n{}\n    {} a = &_a;",
                    target.bytes(),
                    type_to_va(
                        "_a",
                        if target == TargetFeature::Lsx {
                            "A16QI"
                        } else {
                            "A32QI"
                        }
                    ),
                    type_to_ct(in_t[0])
                );
                fn_params = "(a, 0)".to_string();
                as_params = "::<0>(a.as_ptr())".to_string();
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
            if asm_fmts[2].as_str() == "rk" {
                fn_inputs = format!(
                    "    union v{}qi _a;\n{}\n    {} a = &_a;",
                    target.bytes(),
                    type_to_va(
                        "_a",
                        if target == TargetFeature::Lsx {
                            "A16QI"
                        } else {
                            "A32QI"
                        }
                    ),
                    type_to_ct(in_t[0])
                );
                fn_params = "(a, 0)".to_string();
                as_params = "(a.as_ptr(), 0)".to_string();
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 3 && in_t[2] == "UQI" && asm_fmts[1].starts_with("rj") {
            if asm_fmts[2].starts_with("ui") {
                fn_inputs = format!(
                    "    {} a;\n{}",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0])
                );
                let ival = rand_i32(32);
                let uval = rand_u32(asm_fmts[2].get(2..).unwrap().parse::<u8>().unwrap());
                fn_params = format!("(a.v, {ival}, {uval})");
                as_params = format!("::<{uval}>(transmute(a), {ival})");
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
            if asm_fmts[2].starts_with("ui") {
                fn_inputs = format!(
                    "    {} a;\n{}\n    {} b;\n{}",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0]),
                    type_to_ct(in_t[1]),
                    type_to_va("b", in_t[1]),
                );
                let val = rand_u32(asm_fmts[2].get(2..).unwrap().parse::<u8>().unwrap());
                fn_params = format!("(a.v, b.v, {val})");
                as_params = format!("::<{val}>(transmute(a), transmute(b))");
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
            if asm_fmts[2].as_str() == "si12" {
                fn_inputs = format!(
                    "    {} a;\n{}\n    union v{}qi o;\n{}\n    {} b = &o;",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0]),
                    target.bytes(),
                    type_to_va(
                        "o",
                        if target == TargetFeature::Lsx {
                            "AM16QI"
                        } else {
                            "AM32QI"
                        }
                    ),
                    type_to_ct(in_t[1])
                );
                fn_params = "(a.v, b, 0)".to_string();
                as_params = "::<0>(transmute(a), o.as_mut_ptr())".to_string();
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
            if asm_fmts[2].as_str() == "rk" {
                fn_inputs = format!(
                    "    {} a;\n{}\n    union v{}qi o;\n{}\n    {} b = &o;",
                    type_to_ct(in_t[0]),
                    type_to_va("a", in_t[0]),
                    target.bytes(),
                    type_to_va(
                        "o",
                        if target == TargetFeature::Lsx {
                            "AM16QI"
                        } else {
                            "AM32QI"
                        }
                    ),
                    type_to_ct(in_t[1])
                );
                fn_params = "(a.v, b, 0)".to_string();
                as_params = "(transmute(a), o.as_mut_ptr(), 0)".to_string();
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            }
        } else if para_num == 4 {
            match (asm_fmts[2].as_str(), current_name.chars().last().unwrap()) {
                ("si8", t) => {
                    fn_inputs = format!(
                        "    {} a;\n{}\n    union v{}qi o;\n{}\n    {} b = &o;",
                        type_to_ct(in_t[0]),
                        type_to_va("a", in_t[0]),
                        target.bytes(),
                        type_to_va(
                            "o",
                            if target == TargetFeature::Lsx {
                                "AM16QI"
                            } else {
                                "AM32QI"
                            }
                        ),
                        type_to_ct(in_t[1])
                    );
                    let val = rand_u32(type_to_imm(t).try_into().unwrap());
                    fn_params = format!("(a.v, b, 0, {val})");
                    as_params = format!("::<0, {val}>(transmute(a), o.as_mut_ptr())");
                }
                (_, _) => panic!(
                    "unsupported assembly format: {} for {}",
                    asm_fmts[2], current_name
                ),
            };
        }
        let fn_docall = if out_t.to_lowercase() == "void" {
            format!("    __{current_name}{fn_params};")
        } else {
            format!("    {} = __{current_name}{fn_params};", type_to_rx(out_t))
        };
        let fn_result = if out_t.to_lowercase() == "void" {
            if target == TargetFeature::Lsx {
                type_to_rp("V16QI")
            } else {
                type_to_rp("V32QI")
            }
        } else {
            type_to_rp(out_t)
        };
        let fn_assert = {
            if out_t.to_lowercase() == "void" {
                format!(
                    "    printf(\"\\n    {current_name}{as_params};\\n    assert_eq!(r, transmute(o));\\n\"{as_args});"
                )
            } else {
                format!(
                    "    printf(\"\\n    assert_eq!(r, transmute({current_name}{as_params}));\\n\"{as_args});"
                )
            }
        };
        format!(
            r#"
static void {current_name}(void)
{{
    printf("\n#[simd_test(enable = \"{}\")]\n");
    printf("unsafe fn test_{current_name}() {{\n");
{fn_inputs}
{fn_output}
{fn_docall}
{fn_result}
{fn_assert}
    printf("}}\n");
}}
"#,
            target.as_target_feature_arg(current_name)
        )
    };
    let call_function = format!("    {current_name}();\n");
    (impl_function, call_function)
}

pub fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let in_file = args.get(1).cloned().expect("Input file missing!");
    let in_file_path = PathBuf::from(&in_file);
    let in_file_name = in_file_path
        .file_name()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();

    let ext_name = if in_file_name.starts_with("lasx") {
        "lasx"
    } else {
        "lsx"
    };

    if in_file_name.ends_with(".h") {
        gen_spec(in_file, ext_name)
    } else if args.get(2).is_some() {
        gen_test(in_file, ext_name)
    } else {
        gen_bind(in_file, ext_name)
    }
}
