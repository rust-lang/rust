#![feature(pattern)]

mod assert_instr;
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
use walkdir::WalkDir;

fn main() -> Result<(), String> {
    parse_args()
        .into_iter()
        .map(|(filepath, out)| {
            File::open(&filepath)
                .map(|f| (f, filepath, out))
                .map_err(|e| format!("could not read input file: {e}"))
        })
        .map(|res| {
            let (file, filepath, out) = res?;
            serde_yaml::from_reader(file)
                .map(|input: input::GeneratorInput| (input, filepath, out))
                .map_err(|e| format!("could not parse input file: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|(input, filepath, out)| {
            let intrinsics = input.intrinsics.into_iter()
                .map(|intrinsic| {
                    intrinsic.generate_variants(&input.ctx)
                })
                .try_collect()
                .map(|mut vv: Vec<_>| {
                    vv.sort_by_cached_key(|variants| {
                        variants.first().map_or_else(String::default, |variant| {
                            variant.signature.fn_name().to_string()
                        })
                    });
                    vv.into_iter().flatten().collect_vec()
                })?;

            if filepath.ends_with("sve.spec.yml") || filepath.ends_with("sve2.spec.yml") {
                let loads = intrinsics.iter()
                    .filter_map(|i| {
                        if matches!(i.test, Test::Load(..)) {
                            Some(i.clone())
                        } else {
                            None
                        }
                    }).collect();
                let stores = intrinsics.iter()
                    .filter_map(|i| {
                        if matches!(i.test, Test::Store(..)) {
                            Some(i.clone())
                        } else {
                            None
                        }
                    }).collect();
                load_store_tests::generate_load_store_tests(loads, stores, out.as_ref().map(|o| make_tests_filepath(&filepath, o)).as_ref())?;
            }

            Ok((
                input::GeneratorInput {
                    intrinsics,
                    ctx: input.ctx,
                },
                filepath,
                out,
            ))
        })
        .try_for_each(
            |result: context::Result<(input::GeneratorInput, PathBuf, Option<PathBuf>)>| -> context::Result {
                let (generated, filepath, out) = result?;

                let w = match out {
                    Some(out) => Box::new(
                        File::create(make_output_filepath(&filepath, &out))
                            .map_err(|e| format!("could not create output file: {e}"))?,
                    ) as Box<dyn Write>,
                    None => Box::new(std::io::stdout()) as Box<dyn Write>,
                };

                generate_file(generated, w)
                    .map_err(|e| format!("could not generate output file: {e}"))
            },
        )
}

fn parse_args() -> Vec<(PathBuf, Option<PathBuf>)> {
    let mut args_it = std::env::args().skip(1);
    assert!(
        1 <= args_it.len() && args_it.len() <= 2,
        "Usage: cargo run -p stdarch-gen2 -- INPUT_DIR [OUTPUT_DIR]"
    );

    let in_path = Path::new(args_it.next().unwrap().as_str()).to_path_buf();
    assert!(
        in_path.exists() && in_path.is_dir(),
        "invalid path {in_path:#?} given"
    );

    let out_dir = if let Some(dir) = args_it.next() {
        let out_path = Path::new(dir.as_str()).to_path_buf();
        assert!(
            out_path.exists() && out_path.is_dir(),
            "invalid path {out_path:#?} given"
        );
        Some(out_path)
    } else {
        std::env::current_exe()
            .map(|mut f| {
                f.pop();
                f.push("../../crates/core_arch/src/aarch64/");
                f.exists().then_some(f)
            })
            .ok()
            .flatten()
    };

    WalkDir::new(in_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|f| f.file_type().is_file())
        .map(|f| (f.into_path(), out_dir.clone()))
        .collect()
}

fn generate_file(
    generated_input: input::GeneratorInput,
    mut out: Box<dyn Write>,
) -> std::io::Result<()> {
    write!(
        out,
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen2/spec/` and run the following command to re-generate this file:
//
// ```
// cargo run --bin=stdarch-gen2 -- crates/stdarch-gen2/spec
// ```
#![allow(improper_ctypes)]

#[cfg(test)]
use stdarch_test::assert_instr;

use super::*;{uses_neon}

"#,
        uses_neon = generated_input
            .ctx
            .uses_neon_types
            .then_some("\nuse crate::core_arch::arch::aarch64::*;")
            .unwrap_or_default(),
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    write!(proc.stdin.as_ref().unwrap(), "{input}")?;
    output.write_all(proc.wait_with_output()?.stdout.as_slice())
}

/// Derive an output file name from an input file and an output directory.
///
/// The name is formed by:
///
///   - ... taking in_filepath.file_name() (dropping all directory components),
///   - ... dropping a .yml or .yaml extension (if present),
///   - ... then dropping a .spec extension (if present).
///
/// Panics if the resulting name is empty, or if file_name() is not UTF-8.
fn make_output_filepath(in_filepath: &Path, out_dirpath: &Path) -> PathBuf {
    make_filepath(in_filepath, out_dirpath, |name: &str| format!("{name}.rs"))
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
    let mut parts = in_filepath.iter();
    let name = parts
        .next_back()
        .and_then(|f| f.to_str())
        .expect("Inputs must have valid, UTF-8 file_name()");
    let dir = parts.next_back().unwrap();

    let name = name
        .trim_end_matches(".yml")
        .trim_end_matches(".yaml")
        .trim_end_matches(".spec");
    assert!(!name.is_empty());

    let mut output = out_dirpath.to_path_buf();
    output.push(dir);
    output.push(name_formatter(name));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_output_file() {
        macro_rules! t {
            ($src:expr, $outdir:expr, $dst:expr) => {
                let src: PathBuf = $src.iter().collect();
                let outdir: PathBuf = $outdir.iter().collect();
                let dst: PathBuf = $dst.iter().collect();
                assert_eq!(make_output_filepath(&src, &outdir), dst);
            };
        }
        // Documented usage.
        t!(["x", "NAME.spec.yml"], [""], ["x", "NAME.rs"]);
        t!(
            ["x", "NAME.spec.yml"],
            ["a", "b"],
            ["a", "b", "x", "NAME.rs"]
        );
        t!(
            ["x", "y", "NAME.spec.yml"],
            ["out"],
            ["out", "y", "NAME.rs"]
        );
        t!(["x", "NAME.spec.yaml"], ["out"], ["out", "x", "NAME.rs"]);
        t!(["x", "NAME.spec"], ["out"], ["out", "x", "NAME.rs"]);
        t!(["x", "NAME.yml"], ["out"], ["out", "x", "NAME.rs"]);
        t!(["x", "NAME.yaml"], ["out"], ["out", "x", "NAME.rs"]);
        // Unrecognised extensions get treated as part of the stem.
        t!(
            ["x", "NAME.spac.yml"],
            ["out"],
            ["out", "x", "NAME.spac.rs"]
        );
        t!(["x", "NAME.txt"], ["out"], ["out", "x", "NAME.txt.rs"]);
        // Always take the top-level directory from the input path
        t!(
            ["x", "y", "z", "NAME.spec.yml"],
            ["out"],
            ["out", "z", "NAME.rs"]
        );
    }

    #[test]
    #[should_panic]
    fn infer_output_file_no_stem() {
        make_output_filepath(Path::new(".spec.yml"), Path::new(""));
    }
}
