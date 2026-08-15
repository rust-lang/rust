use std::process::Command;

use itertools::Itertools;

use super::intrinsic_helpers::TypeDefinition;
use crate::common::cli::{CcArgStyle, ProcessedCli};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::values::{test_values_array_name, test_values_array_static};
use crate::common::{PASSES, PREDICATE_LOCAL, SupportedArchitecture};

/// Rust definitions that are included verbatim in the generated source. In particular, defines
/// a wrapper around float types that defines `NaN`s to be equal reflexively to enable
/// comparison of results that use floats types.
const COMMON_RUST_DEFINITIONS: &str = r#"
macro_rules! wrap_partialeq {
    ($($wrapper:ident ($inner:ty)),*) => {$(
        #[derive(Debug, Copy, Clone)]
        #[repr(transparent)]
        pub struct $wrapper($inner);

        impl PartialEq for $wrapper {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
            }
        }

        impl Eq for $wrapper {}
    )*}
}

wrap_partialeq!(NanEqF16(f16), NanEqF32(f32), NanEqF64(f64));
"#;

/// Run rustfmt on the generated source code
pub fn run_rustfmt(source_path: &str) {
    let output = Command::new("rustfmt")
        .args([source_path])
        .output()
        .expect("failed to run rustfmt on generated sources");

    if !output.status.success() {
        panic!(
            "failed to run rustfmt on generated sources:\nstdout:{stdout}\nstderr:{stderr}",
            stdout = String::from_utf8_lossy(&output.stdout),
            stderr = String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Writes a `Cargo.toml` containing a workspace with `module_count` members to `w`.
///
/// e.g.
/// ```toml
/// [workspace]
/// members = [ "mod_0", "mod_1" ]
/// ```
pub fn write_bin_cargo_toml(
    w: &mut impl std::io::Write,
    module_count: usize,
) -> std::io::Result<()> {
    write!(
        w,
        r#"
[workspace]
members = [{members}]
"#,
        members = (0..module_count).format_with(",", |i, fmt| fmt(&format_args!("\"mod_{i}\"")))
    )
}

/// Writes a `Cargo.toml` for a crate with name `name` to `w` that will contain a single Rust source
/// file with a subset of the testing being generated.
pub fn write_lib_cargo_toml(w: &mut impl std::io::Write, name: &str) -> std::io::Result<()> {
    write!(
        w,
        r#"
[package]
name = "{name}"
version = "{version}"
authors = [{authors}]
license = "{license}"
edition = "2018"

[dependencies]
core_arch = {{ path = "../../crates/core_arch" }}

[build-dependencies]
cc = "1"
"#,
        version = env!("CARGO_PKG_VERSION"),
        authors = env!("CARGO_PKG_AUTHORS")
            .split(":")
            .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
        license = env!("CARGO_PKG_LICENSE"),
    )
}

/// Writes a Rust source file into `w` with common definitions, static arrays with test values,
/// declarations of C wrapper functions for FFI and Rust test functions.
pub fn write_lib_rs<A: SupportedArchitecture>(
    w: &mut impl std::io::Write,
    i: usize,
    intrinsics: &[Intrinsic<A>],
) -> std::io::Result<()> {
    writeln!(
        w,
        r#"
{notice}
#![feature(simd_ffi)]
#![feature(f16)]
#![allow(unused)]

// Cargo will spam the logs if these warnings are not silenced.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

{prelude}
{COMMON_RUST_DEFINITIONS}
"#,
        notice = A::NOTICE,
        prelude = A::RUST_PRELUDE,
    )?;

    let mut seen = std::collections::HashSet::new();

    for intrinsic in intrinsics {
        for arg in &intrinsic.arguments.args {
            // Skip arguments with constraints as these correspond to generic instantiatons, and
            // predicates for scalable intrinsics as the same predicate is used for all intrinsics
            // under test.
            if !arg.has_constraint() && !arg.is_predicate {
                let name = test_values_array_name(&arg.ty);

                if seen.insert(name) {
                    test_values_array_static(w, &arg.ty)?;
                }
            }
        }
    }

    write_bindings_rust(w, i, intrinsics)?;

    for intrinsic in intrinsics {
        create_rust_test(w, intrinsic)?;
    }

    Ok(())
}

/// Writes the body of an intrinsic test to `w` for `intrinsic`.
///
/// Each specialisation of the intrinsic (i.e. specific instantiations of the immediate arguments
/// of the intrinsic) is added to an array of specialisations. Each specialisation is tested
/// (first loop) `PASSES` number of times (second loop). For a given iteration of a given
/// specialisation, test values are loaded for each argument and passed to the Rust intrinsic
/// and the C wrapper function, and the results are compared.
fn generate_rust_test_loop<A: SupportedArchitecture>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<A>,
) -> std::io::Result<()> {
    let intrinsic_name = &intrinsic.name;

    // Each function (and each specialization) has its own type. Erase that type with a cast.
    let mut coerce = String::from("fn(");
    let mut c_coerce = String::from("fn(_, ");
    for _ in intrinsic.arguments.iter().filter(|a| !a.has_constraint()) {
        coerce += "_, ";
        c_coerce += "_, ";
    }
    coerce += ") -> _";
    c_coerce += ")";

    write!(
        w,
        r#"
let specializations = [{specializations}];
for (id, rust, c) in specializations {{
    for i in 0..{PASSES} {{
        unsafe {{
            {predicate}
            {loaded_args}
            let __rust_return_value = rust({rust_args});

            let mut __c_return_value = std::mem::MaybeUninit::uninit();
            c(__c_return_value.as_mut_ptr(){c_args});
            let __c_return_value = __c_return_value.assume_init();

            {comparison}
        }}
    }}
}}
"#,
        specializations = intrinsic
            .specializations()
            .format_with(",", |imm_values, fmt| {
                if imm_values.is_empty() {
                    fmt(&format_args!(
                        "(\"\", {intrinsic_name}, {intrinsic_name}_wrapper)"
                    ))
                } else {
                    let constraint_args = intrinsic.arguments.iter().filter(|a| a.has_constraint());
                    fmt(&format_args!(
                        r#"
                        (
                            "{const_args}",
                            {intrinsic_name}::<{const_args}> as unsafe {coerce},
                            {intrinsic_name}_wrapper_{c_const_args} as unsafe extern "C" {c_coerce}
                        )
                        "#,
                        const_args = imm_values
                            .iter()
                            .zip(constraint_args)
                            .map(|(imm_val, arg)| {
                                match arg.ty.kind() {
                                    TypeKind::SvPattern | TypeKind::SvPrefetchOp => {
                                        format!("{{ {}_from_i32({imm_val}) }}", arg.ty.kind())
                                    }
                                    _ => imm_val.to_string(),
                                }
                            })
                            .join(","),
                        c_const_args = imm_values.iter().join("_"),
                    ))
                }
            }),
        loaded_args = intrinsic.arguments.load_values_rust(),
        rust_args = intrinsic.arguments.as_call_param_rust(),
        c_args = intrinsic.arguments.as_c_call_param_rust(),
        predicate = if intrinsic.has_scalable_argument_or_result() {
            format!(
                "let {PREDICATE_LOCAL} = {pred};",
                pred = A::predicate_function(intrinsic.results.inner_size()),
            )
        } else {
            "".to_string()
        },
        comparison = intrinsic.results.comparison_function(),
    )
}

/// Writes a test function for an given intrinsic to `w`, with a body generated by
/// `generate_rust_test_loop`.
fn create_rust_test<A: SupportedArchitecture>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<A>,
) -> std::io::Result<()> {
    trace!("generating `{}`", intrinsic.name);

    write!(
        w,
        r#"
#[test]
fn test_{intrinsic_name}() {{
"#,
        intrinsic_name = intrinsic.name,
    )?;

    generate_rust_test_loop(w, intrinsic)?;

    writeln!(w, "}}")?;

    Ok(())
}

/// Writes an `extern "C"` block with function declarations for each of the C wrapper functions into
/// `w`.
pub fn write_bindings_rust<A: SupportedArchitecture>(
    w: &mut impl std::io::Write,
    i: usize,
    intrinsics: &[Intrinsic<A>],
) -> std::io::Result<()> {
    write!(
        w,
        r#"
#[allow(improper_ctypes)]
#[link(name = "wrapper_{i}")]
unsafe extern "C" {{
    {definitions}
}}
"#,
        definitions = intrinsics.iter().format_with("", |intrinsic, fmt| {
            fmt(&intrinsic
                .specializations()
                .format_with("\n", |imm_values, fmt| {
                    fmt(&format_args!(
                        "fn {name}_wrapper{imm_arglist}(__dst: *mut {return_ty}{arglist});",
                        return_ty = intrinsic.results.rust_type(),
                        name = intrinsic.name,
                        imm_arglist = imm_values
                            .iter()
                            .format_with("", |i, fmt| fmt(&format_args!("_{i}"))),
                        arglist = intrinsic.arguments.as_non_imm_arglist_rust(),
                    ))
                }))
        })
    )
}

/// Writes a `build.rs` into `w` for each test crate that compiles the corresponding C source code
/// with wrapper functions.
pub fn write_build_rs(
    w: &mut impl std::io::Write,
    i: usize,
    arch_flags: &[&str],
    cli_options: &ProcessedCli,
) -> std::io::Result<()> {
    const COMMON_FLAGS: &[&str] = &["-ffp-contract=off", "-Wno-narrowing"];
    const CLANG_FLAGS: &[&str] = &["-ffp-model=strict"];
    const GCC_FLAGS: &[&str] = &[
        "-flax-vector-conversions",
        "-fno-fast-math",
        "-frounding-math",
        "-fexcess-precision=standard",
        "-ftrapping-math",
        "-fsignaling-nans",
    ];

    write!(
        w,
        r#"
fn main() {{
    cc::Build::new()
        .file("../../c_programs/wrapper_{i}.c")
        .opt_level(2)
        .flags(&[{flags}])
        .compile("wrapper_{i}");
}}
"#,
        flags = COMMON_FLAGS
            .iter()
            .chain(match cli_options.cc_arg_style {
                CcArgStyle::Gcc => GCC_FLAGS,
                CcArgStyle::Clang => CLANG_FLAGS,
            })
            .chain(arch_flags)
            .format_with(",", |flag, fmt| fmt(&format_args!("\"{flag}\""))),
    )
}
