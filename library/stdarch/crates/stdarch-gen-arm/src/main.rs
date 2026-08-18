#![feature(pattern)]

mod assert_instr;
mod big_endian;
mod context;
mod expression;
mod fn_suffix;
mod input;
mod intrinsic;
mod load_store_tests;
mod matching;
mod predicate_forms;
mod typekinds;
mod wildcards;
mod wildstring;

use intrinsic::Test;
use itertools::Itertools;
use quote::quote;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use stdarch_gen_common::{Mode, run_generator};
use walkdir::WalkDir;

fn main() -> Result<(), String> {
    let (in_path, out_base) = parse_args();
    let mode = Mode::from_env();

    for filepath in WalkDir::new(&in_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|f| f.file_type().is_file())
        .filter(|f| f.file_name().to_string_lossy().ends_with(".yml"))
        .map(|f| f.into_path())
    {
        // Directory the harness checks/blesses against. The committed output
        // location for this spec file (`<out_base>/<arch>/<feature>/`).
        let committed = make_output_filepath(&filepath, &out_base)
            .parent()
            .expect("generated output path must have a parent directory")
            .to_path_buf();

        run_generator(&committed, mode, |scratch: &Path| {
            generate_spec(&filepath, scratch)
        })
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Generate the output files for a single spec file into `out_dir`.
fn generate_spec(filepath: &Path, out_dir: &Path) -> Result<(), String> {
    let file = File::open(filepath).map_err(|e| format!("could not read input file: {e}"))?;
    let input: input::GeneratorInput =
        serde_yaml::from_reader(file).map_err(|e| format!("could not parse input file: {e}"))?;

    let intrinsics = input
        .intrinsics
        .into_iter()
        .map(|intrinsic| intrinsic.generate_variants(&input.ctx))
        .try_collect()
        .map(|mut vv: Vec<_>| {
            vv.sort_by_cached_key(|variants| {
                variants.first().map_or_else(String::default, |variant| {
                    variant.signature.fn_name().to_string()
                })
            });
            vv.into_iter().flatten().collect_vec()
        })?;

    if input.ctx.generate_load_store_tests {
        let loads = intrinsics
            .iter()
            .filter_map(|i| match i.test {
                Test::Load(..) => Some(i.clone()),
                _ => None,
            })
            .collect();
        let stores = intrinsics
            .iter()
            .filter_map(|i| match i.test {
                Test::Store(..) => Some(i.clone()),
                _ => None,
            })
            .collect();
        // Reuse make_tests_filepath to derive the correct leaf name
        // (`ld_st_tests_<arch>.rs`), then write it into out_dir.
        let tests_name = make_tests_filepath(filepath, Path::new(""))
            .file_name()
            .expect("load/store test path must have a file name")
            .to_owned();
        let tests_path = out_dir.join(tests_name);
        load_store_tests::generate_load_store_tests(loads, stores, Some(&tests_path))?;
    }

    let generated = input::GeneratorInput {
        intrinsics,
        ctx: input.ctx,
    };
    let out_file = File::create(out_dir.join("generated.rs"))
        .map_err(|e| format!("could not create output file: {e}"))?;
    generate_file(generated, Box::new(out_file) as Box<dyn Write>)
        .map_err(|e| format!("could not generate output file: {e}"))
}

fn parse_args() -> (PathBuf, PathBuf) {
    let mut args_it = std::env::args().skip(1);
    assert!(
        1 <= args_it.len() && args_it.len() <= 2,
        "Usage: cargo run -p stdarch-gen-arm -- INPUT_DIR [OUTPUT_DIR]\n\
        where:\n\
        - INPUT_DIR contains a tree like: INPUT_DIR/<feature>/<arch>.spec.yml\n\
        - OUTPUT_DIR is a directory like: crates/core_arch/src/"
    );

    let in_path = Path::new(args_it.next().unwrap().as_str()).to_path_buf();
    assert!(
        in_path.exists() && in_path.is_dir(),
        "invalid path {in_path:#?} given"
    );

    let out_base = if let Some(dir) = args_it.next() {
        let out_path = Path::new(dir.as_str()).to_path_buf();
        assert!(
            out_path.exists() && out_path.is_dir(),
            "invalid path {out_path:#?} given"
        );
        out_path
    } else {
        std::env::current_exe()
            .ok()
            .map(|mut f| {
                f.pop();
                f.push("../../crates/core_arch/src/");
                f
            })
            .filter(|f| f.exists())
            .expect("could not locate crates/core_arch/src; pass OUTPUT_DIR command-line argument explicitly")
    };

    (in_path, out_base)
}

fn generate_file(
    generated_input: input::GeneratorInput,
    mut out: Box<dyn Write>,
) -> std::io::Result<()> {
    write!(
        out,
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-arm/spec/` and run the following command to re-generate this file:
//
// ```
// cargo run --bin=stdarch-gen-arm -- crates/stdarch-gen-arm/spec
// ```
#![allow(improper_ctypes)]

#[cfg(test)]
use stdarch_test::assert_instr;

use super::*;{uses_neon}

"#,
        uses_neon = if generated_input.ctx.uses_neon_types {
            "\nuse crate::core_arch::arch::aarch64::*;\nuse super::{AsSigned, AsUnsigned};"
        } else {
            ""
        },
    )?;
    let intrinsics = generated_input.intrinsics;
    format_code(out, quote! { #(#intrinsics)* })?;
    Ok(())
}

pub fn format_code(
    mut output: impl std::io::Write,
    input: impl std::fmt::Display,
) -> std::io::Result<()> {
    let proc = Command::new("rustfmt")
        .arg("--config")
        .arg("newline_style=Unix")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    write!(proc.stdin.as_ref().unwrap(), "{input}")?;
    output.write_all(proc.wait_with_output()?.stdout.as_slice())
}

/// Derive an output file path from an input file path and an output directory.
///
/// `in_filepath` is expected to have a structure like:
///     .../<feature>/<arch>.spec.yml
///
/// The resulting output path will have a structure like:
///     <out_dirpath>/<arch>/<feature>/generated.rs
///
/// Panics if the resulting name is empty, or if file_name() is not UTF-8.
fn make_output_filepath(in_filepath: &Path, out_dirpath: &Path) -> PathBuf {
    make_filepath(in_filepath, out_dirpath, |_name: &str| {
        "generated.rs".to_owned()
    })
}

fn make_tests_filepath(in_filepath: &Path, out_dirpath: &Path) -> PathBuf {
    make_filepath(in_filepath, out_dirpath, |name: &str| {
        format!("ld_st_tests_{name}.rs")
    })
}

fn make_filepath<F: FnOnce(&str) -> String>(
    in_filepath: &Path,
    out_dirpath: &Path,
    name_formatter: F,
) -> PathBuf {
    let mut parts = in_filepath.components().rev().map(|f| {
        f.as_os_str()
            .to_str()
            .expect("Inputs must have valid, UTF-8 file_name()")
    });
    let yml = parts.next().expect("Not enough input path elements.");
    let feature = parts.next().expect("Not enough input path elements.");

    let arch = yml
        .strip_suffix(".yml")
        .expect("Expected .yml file input.")
        .strip_suffix(".spec")
        .expect("Expected .spec.yml file input.");
    if arch.is_empty() {
        panic!("Extended ARCH.spec.yml file input.");
    }

    let mut output = out_dirpath.to_path_buf();
    output.push(arch);
    output.push(feature);
    output.push(name_formatter(arch));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_output_file() {
        macro_rules! t {
            ($src:expr, $outdir:expr, $dst:expr, $ldst:expr) => {
                let src: PathBuf = $src.iter().collect();
                let outdir: PathBuf = $outdir.iter().collect();
                let dst: PathBuf = $dst.iter().collect();
                let ldst: PathBuf = $ldst.iter().collect();
                assert_eq!(make_output_filepath(&src, &outdir), dst);
                assert_eq!(make_tests_filepath(&src, &outdir), ldst);
            };
        }
        // Documented usage.
        t!(
            ["FEAT", "ARCH.spec.yml"],
            [""],
            ["ARCH", "FEAT", "generated.rs"],
            ["ARCH", "FEAT", "ld_st_tests_ARCH.rs"]
        );
        t!(
            ["x", "y", "FEAT", "ARCH.spec.yml"],
            ["out"],
            ["out", "ARCH", "FEAT", "generated.rs"],
            ["out", "ARCH", "FEAT", "ld_st_tests_ARCH.rs"]
        );
        t!(
            ["p", "q", "FEAT", "ARCH.spec.yml"],
            ["a", "b"],
            ["a", "b", "ARCH", "FEAT", "generated.rs"],
            ["a", "b", "ARCH", "FEAT", "ld_st_tests_ARCH.rs"]
        );
        // Extra extensions get treated as part of the stem.
        t!(
            ["FEAT", "ARCH.variant.spec.yml"],
            ["out"],
            ["out", "ARCH.variant", "FEAT", "generated.rs"],
            ["out", "ARCH.variant", "FEAT", "ld_st_tests_ARCH.variant.rs"]
        );
    }

    #[test]
    #[should_panic]
    fn infer_output_file_no_stem() {
        let src = PathBuf::from("FEAT/.spec.yml");
        make_output_filepath(&src, Path::new(""));
    }

    #[test]
    #[should_panic]
    fn infer_output_file_no_feat() {
        let src = PathBuf::from("ARCH.spec.yml");
        make_output_filepath(&src, Path::new(""));
    }

    #[test]
    #[should_panic]
    fn infer_output_file_ldst_no_stem() {
        let src = PathBuf::from("FEAT/.spec.yml");
        make_tests_filepath(&src, Path::new(""));
    }

    #[test]
    #[should_panic]
    fn infer_output_file_ldst_no_feat() {
        let src = PathBuf::from("ARCH.spec.yml");
        make_tests_filepath(&src, Path::new(""));
    }
}
