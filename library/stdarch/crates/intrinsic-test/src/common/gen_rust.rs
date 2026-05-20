use std::process::Command;

use itertools::Itertools;

use super::intrinsic_helpers::IntrinsicTypeDefinition;
use crate::common::argument::ArgumentList;
use crate::common::cli::{CcArgStyle, ProcessedCli};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::values::test_values_array_name;

// The number of times each intrinsic will be called - influences the generation of the
// test arrays to minimise repeated testing of the same test values.
pub(crate) const PASSES: u32 = 20;

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

macro_rules! concatln {
    ($($lines:expr),* $(,)?) => {
        concat!($( $lines, "\n" ),*)
    };
}

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
    write!(w, concatln!("[workspace]", "members = ["))?;
    for i in 0..module_count {
        writeln!(w, "    \"mod_{i}\",")?;
    }
    writeln!(w, "]")
}

/// Writes a `Cargo.toml` for a crate with name `name` to `w` that will contain a single Rust source
/// file with a subset of the testing being generated.
pub fn write_lib_cargo_toml(w: &mut impl std::io::Write, name: &str) -> std::io::Result<()> {
    write!(
        w,
        concatln!(
            "[package]",
            "name = \"{name}\"",
            "version = \"{version}\"",
            "authors = [{authors}]",
            "license = \"{license}\"",
            "edition = \"2018\"",
            "",
            "[dependencies]",
            "core_arch = {{ path = \"../../crates/core_arch\" }}",
            "",
            "[build-dependencies]",
            "cc = \"1\""
        ),
        name = name,
        version = env!("CARGO_PKG_VERSION"),
        authors = env!("CARGO_PKG_AUTHORS")
            .split(":")
            .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
        license = env!("CARGO_PKG_LICENSE"),
    )
}

/// Writes a Rust source file into `w` with common definitions, static arrays with test values,
/// declarations of C wrapper functions for FFI and Rust test functions.
pub fn write_lib_rs<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    notice: &str,
    cfg: &str,
    definitions: &str,
    i: usize,
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(w, "{notice}")?;

    writeln!(w, "#![feature(simd_ffi)]")?;
    writeln!(w, "#![feature(f16)]")?;
    writeln!(w, "#![allow(unused)]")?;

    // Cargo will spam the logs if these warnings are not silenced.
    writeln!(w, "#![allow(non_upper_case_globals)]")?;
    writeln!(w, "#![allow(non_camel_case_types)]")?;
    writeln!(w, "#![allow(non_snake_case)]")?;

    writeln!(w, "{cfg}")?;

    writeln!(w, "{}", COMMON_RUST_DEFINITIONS)?;

    writeln!(w, "{definitions}")?;

    let mut seen = std::collections::HashSet::new();

    for intrinsic in intrinsics {
        for arg in &intrinsic.arguments.args {
            if !arg.has_constraint() {
                let name = test_values_array_name(&arg.ty, PASSES);

                if seen.insert(name) {
                    ArgumentList::gen_arg_rust(arg, w, PASSES)?;
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
fn generate_rust_test_loop<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
    passes: u32,
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

    if intrinsic
        .arguments
        .iter()
        .filter(|arg| arg.has_constraint())
        .count()
        == 0
    {
        writeln!(
            w,
            "    let specializations = [(\"\", {intrinsic_name}, {intrinsic_name}_wrapper)];"
        )?;
    } else {
        writeln!(w, "    let specializations = [")?;

        intrinsic.iter_specializations(|imm_values| {
            writeln!(
                w,
                "        (\"{const_args}\", {intrinsic_name}::<{const_args}> as unsafe {coerce}, {intrinsic_name}_wrapper_{c_const_args} as unsafe extern \"C\" {c_coerce}),",
                const_args = imm_values.iter().join(","),
                c_const_args = imm_values.iter().join("_"),
            )
        })?;

        writeln!(w, "    ];")?;
    }

    let (cast_prefix, cast_suffix) = if intrinsic.results.is_simd() {
        (
            format!(
                "std::mem::transmute::<_, [{}; {}]>(",
                intrinsic.results.rust_scalar_type().replace("f", "NanEqF"),
                intrinsic.results.num_lanes() * intrinsic.results.num_vectors()
            ),
            ")",
        )
    } else if intrinsic.results.kind == TypeKind::Float {
        (
            match intrinsic.results.inner_size() {
                16 => format!("NanEqF16("),
                32 => format!("NanEqF32("),
                64 => format!("NanEqF64("),
                _ => unimplemented!(),
            },
            ")",
        )
    } else {
        ("".to_string(), "")
    };

    write!(
        w,
        concatln!(
            "    for (id, rust, c) in specializations {{",
            "        for i in 0..{passes} {{",
            "            unsafe {{",
            "{loaded_args}",
            "                let __rust_return_value = rust({rust_args});",
            "",
            "                let mut __c_return_value = std::mem::MaybeUninit::uninit();",
            "                c(__c_return_value.as_mut_ptr(){c_args});",
            "                let __c_return_value = __c_return_value.assume_init();",
            "",
            "                assert_eq!({cast_prefix}__rust_return_value{cast_suffix}, {cast_prefix}__c_return_value{cast_suffix}, \"{{id}}\");",
            "            }}",
            "        }}",
            "    }}",
        ),
        loaded_args = intrinsic.arguments.load_values_rust(passes),
        rust_args = intrinsic.arguments.as_call_param_rust(),
        c_args = intrinsic.arguments.as_c_call_param_rust(),
        passes = passes,
        cast_prefix = cast_prefix,
        cast_suffix = cast_suffix,
    )
}

/// Writes a test function for an given intrinsic to `w`, with a body generated by
/// `generate_rust_test_loop`.
fn create_rust_test<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
) -> std::io::Result<()> {
    trace!("generating `{}`", intrinsic.name);

    write!(
        w,
        concatln!("#[test]", "fn test_{intrinsic_name}() {{"),
        intrinsic_name = intrinsic.name,
    )?;

    generate_rust_test_loop(w, intrinsic, PASSES)?;

    writeln!(w, "}}")?;

    Ok(())
}

/// Writes an `extern "C"` block with function declarations for each of the C wrapper functions into
/// `w`.
pub fn write_bindings_rust<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    i: usize,
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(
        w,
        concatln!(
            "#[allow(improper_ctypes)]",
            "#[link(name = \"wrapper_{i}\")]",
            "unsafe extern \"C\" {{"
        ),
        i = i
    )?;

    for intrinsic in intrinsics {
        intrinsic.iter_specializations(|imm_values| {
            writeln!(
                w,
                "    fn {name}_wrapper{imm_arglist}(__dst: *mut {return_ty}{arglist});",
                return_ty = intrinsic.results.rust_type(),
                name = intrinsic.name,
                imm_arglist = imm_values
                    .iter()
                    .format_with("", |i, fmt| fmt(&format_args!("_{i}"))),
                arglist = intrinsic.arguments.as_non_imm_arglist_rust(),
            )
        })?;
    }

    writeln!(w, "}}")
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
        concatln!(
            "fn main() {{",
            "    cc::Build::new()",
            "    .file(\"../../c_programs/wrapper_{i}.c\")",
            "    .opt_level(2)",
            "    .flags(&[",
        ),
        i = i
    )?;

    let compiler_specific_flags = match cli_options.cc_arg_style {
        CcArgStyle::Gcc => GCC_FLAGS,
        CcArgStyle::Clang => CLANG_FLAGS,
    };

    for flag in COMMON_FLAGS
        .iter()
        .chain(compiler_specific_flags)
        .chain(arch_flags)
    {
        writeln!(w, "\"{flag}\",")?;
    }

    write!(
        w,
        concatln!("    ])", "    .compile(\"wrapper_{i}\");", "}}"),
        i = i
    )
}
