use std::collections::HashSet;
use std::env;

use gccjit::Context;
#[cfg(feature = "master")]
use gccjit::Version;
use rustc_codegen_ssa::target_features;
use rustc_data_structures::smallvec::{SmallVec, smallvec};
use rustc_session::Session;
use rustc_target::spec::{Arch, RelocModel};

fn gcc_features_by_flags(sess: &Session, features: &mut Vec<String>) {
    target_features::retpoline_features_by_flags(sess, features);
    // FIXME: LLVM also sets +reserve-x18 here under some conditions.
}

/// The list of GCC features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
pub(crate) fn global_gcc_features(sess: &Session) -> Vec<String> {
    // Features that come earlier are overridden by conflicting features later in the string.
    // Typically we'll want more explicit settings to override the implicit ones, so:
    //
    // * Features from -Ctarget-cpu=*; are overridden by [^1]
    // * Features implied by --target; are overridden by
    // * Features from -Ctarget-feature; are overridden by
    // * function specific features.
    //
    // [^1]: target-cpu=native is handled here, other target-cpu values are handled implicitly
    // through GCC march implementation.
    //
    // FIXME(nagisa): it isn't clear what's the best interaction between features implied by
    // `-Ctarget-cpu` and `--target` are. On one hand, you'd expect CLI arguments to always
    // override anything that's implicit, so e.g. when there's no `--target` flag, features implied
    // the host target are overridden by `-Ctarget-cpu=*`. On the other hand, what about when both
    // `--target` and `-Ctarget-cpu=*` are specified? Both then imply some target features and both
    // flags are specified by the user on the CLI. It isn't as clear-cut which order of precedence
    // should be taken in cases like these.
    let mut features = vec![];

    let mut extend_backend_features = |feature: &str, enable: bool| {
        // We run through `to_gcc_features` when
        // passing requests down to GCC. This means that all in-language
        // features also work on the command line instead of having two
        // different names when the GCC name and the Rust name differ.
        features.extend(
            to_gcc_features(sess, feature)
                .iter()
                .flat_map(|feat| to_gcc_features(sess, feat).into_iter())
                .map(|feature| if !enable { format!("-{}", feature) } else { feature.to_string() }),
        );
    };

    // Features implied by an implicit or explicit `--target`.
    target_features::target_spec_to_backend_features(sess, &mut extend_backend_features);

    // -Ctarget-features
    target_features::flag_to_backend_features(sess, extend_backend_features);

    gcc_features_by_flags(sess, &mut features);

    features
}

// To find a list of GCC's names, check https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
pub fn to_gcc_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    // cSpell:disable
    match (&sess.target.arch, s) {
        // FIXME: seems like x87 does not exist?
        (&Arch::X86 | &Arch::X86_64, "x87") => smallvec![],
        (&Arch::X86 | &Arch::X86_64, "sse4.2") => smallvec!["sse4.2", "crc32"],
        (&Arch::X86 | &Arch::X86_64, "pclmulqdq") => smallvec!["pclmul"],
        (&Arch::X86 | &Arch::X86_64, "rdrand") => smallvec!["rdrnd"],
        (&Arch::X86 | &Arch::X86_64, "bmi1") => smallvec!["bmi"],
        (&Arch::X86 | &Arch::X86_64, "cmpxchg16b") => smallvec!["cx16"],
        (&Arch::X86 | &Arch::X86_64, "lahfsahf") => smallvec!["sahf"],
        (&Arch::X86 | &Arch::X86_64, "avx512vaes") => smallvec!["vaes"],
        (&Arch::X86 | &Arch::X86_64, "avx512gfni") => smallvec!["gfni"],
        (&Arch::X86 | &Arch::X86_64, "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512vbmi2'.
        (&Arch::X86 | &Arch::X86_64, "avx512vbmi2") => {
            smallvec!["avx512vbmi2", "avx512bw"]
        }
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512bitalg'.
        (&Arch::X86 | &Arch::X86_64, "avx512bitalg") => {
            smallvec!["avx512bitalg", "avx512bw"]
        }
        (&Arch::AArch64, "rcpc2") => smallvec!["rcpc-immo"],
        (&Arch::AArch64, "dpb") => smallvec!["ccpp"],
        (&Arch::AArch64, "dpb2") => smallvec!["ccdp"],
        (&Arch::AArch64, "frintts") => smallvec!["fptoint"],
        (&Arch::AArch64, "fcma") => smallvec!["complxnum"],
        (&Arch::AArch64, "pmuv3") => smallvec!["perfmon"],
        (&Arch::AArch64, "paca") => smallvec!["pauth"],
        (&Arch::AArch64, "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In GCC neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        (&Arch::AArch64, "f32mm") => smallvec!["f32mm", "neon"],
        (&Arch::AArch64, "f64mm") => smallvec!["f64mm", "neon"],
        (&Arch::AArch64, "fhm") => smallvec!["fp16fml", "neon"],
        (&Arch::AArch64, "fp16") => smallvec!["fullfp16", "neon"],
        (&Arch::AArch64, "jsconv") => smallvec!["jsconv", "neon"],
        (&Arch::AArch64, "sve") => smallvec!["sve", "neon"],
        (&Arch::AArch64, "sve2") => smallvec!["sve2", "neon"],
        (&Arch::AArch64, "sve2-aes") => smallvec!["sve2-aes", "neon"],
        (&Arch::AArch64, "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        (&Arch::AArch64, "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        (&Arch::AArch64, "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
        (_, s) => smallvec![s],
    }
    // cSpell:enable
}

fn arch_to_gcc(name: &str) -> &str {
    match name {
        "M68000" => "68000",
        "M68020" => "68020",
        _ => name,
    }
}

fn handle_native(name: &str) -> &str {
    if name != "native" {
        return arch_to_gcc(name);
    }

    #[cfg(feature = "master")]
    {
        // Get the native arch.
        let context = Context::default();
        context.get_target_info().arch().unwrap().to_str().unwrap()
    }
    #[cfg(not(feature = "master"))]
    unimplemented!();
}

pub fn target_cpu(sess: &Session) -> &str {
    match sess.opts.cg.target_cpu {
        Some(ref name) => handle_native(name),
        None => handle_native(sess.target.cpu.as_ref()),
    }
}

pub fn new_context<'gcc>(sess: &Session) -> Context<'gcc> {
    let context = Context::default();
    if matches!(sess.target.arch, Arch::X86 | Arch::X86_64) {
        context.add_command_line_option("-masm=intel");
    }
    #[cfg(feature = "master")]
    {
        context.set_special_chars_allowed_in_func_names("$.*");
        let version = Version::get();
        let version = format!("{}.{}.{}", version.major, version.minor, version.patch);
        context.set_output_ident(&format!(
            "rustc version {} with libgccjit {}",
            rustc_interface::util::rustc_version_str().unwrap_or("unknown version"),
            version,
        ));
    }
    // FIXME(antoyo): check if this should only be added when using -Cforce-unwind-tables=n.
    context.add_command_line_option("-fno-asynchronous-unwind-tables");

    if sess.panic_strategy().unwinds() {
        context.add_command_line_option("-fexceptions");
        context.add_driver_option("-fexceptions");
    }

    let disabled_features: HashSet<_> = sess
        .opts
        .cg
        .target_feature
        .split(',')
        .filter(|feature| feature.starts_with('-'))
        .map(|string| &string[1..])
        .collect();

    if !disabled_features.contains("avx") && sess.target.arch == Arch::X86_64 {
        // NOTE: we always enable AVX because the equivalent of llvm.x86.sse2.cmp.pd in GCC for
        // SSE2 is multiple builtins, so we use the AVX __builtin_ia32_cmppd instead.
        // FIXME(antoyo): use the proper builtins for llvm.x86.sse2.cmp.pd and similar.
        context.add_command_line_option("-mavx");
    }

    for arg in &sess.opts.cg.llvm_args {
        context.add_command_line_option(arg);
    }
    // NOTE: This is needed to compile the file src/intrinsic/archs.rs during a bootstrap of rustc.
    context.add_command_line_option("-fno-var-tracking-assignments");
    // NOTE: an optimization (https://github.com/rust-lang/rustc_codegen_gcc/issues/53).
    context.add_command_line_option("-fno-semantic-interposition");
    // NOTE: Rust relies on LLVM not doing TBAA (https://github.com/rust-lang/unsafe-code-guidelines/issues/292).
    context.add_command_line_option("-fno-strict-aliasing");
    // NOTE: Rust relies on LLVM doing wrapping on overflow.
    context.add_command_line_option("-fwrapv");

    if let Some(model) = sess.code_model() {
        use rustc_target::spec::CodeModel;

        context.add_command_line_option(match model {
            CodeModel::Tiny => "-mcmodel=tiny",
            CodeModel::Small => "-mcmodel=small",
            CodeModel::Kernel => "-mcmodel=kernel",
            CodeModel::Medium => "-mcmodel=medium",
            CodeModel::Large => "-mcmodel=large",
        });
    }

    add_pic_option(&context, sess.relocation_model());

    let target_cpu = target_cpu(sess);
    if target_cpu != "generic" {
        context.add_command_line_option(format!("-march={}", target_cpu));
    }

    if sess.opts.unstable_opts.function_sections.unwrap_or(sess.target.function_sections) {
        context.add_command_line_option("-ffunction-sections");
        context.add_command_line_option("-fdata-sections");
    }

    if env::var("CG_GCCJIT_DUMP_RTL").as_deref() == Ok("1") {
        context.add_command_line_option("-fdump-rtl-vregs");
    }
    if env::var("CG_GCCJIT_DUMP_RTL_ALL").as_deref() == Ok("1") {
        context.add_command_line_option("-fdump-rtl-all");
    }
    if env::var("CG_GCCJIT_DUMP_TREE_ALL").as_deref() == Ok("1") {
        context.add_command_line_option("-fdump-tree-all-eh");
    }
    if env::var("CG_GCCJIT_DUMP_IPA_ALL").as_deref() == Ok("1") {
        context.add_command_line_option("-fdump-ipa-all-eh");
    }
    if env::var("CG_GCCJIT_DUMP_CODE").as_deref() == Ok("1") {
        context.set_dump_code_on_compile(true);
    }
    if env::var("CG_GCCJIT_DUMP_GIMPLE").as_deref() == Ok("1") {
        context.set_dump_initial_gimple(true);
    }
    if env::var("CG_GCCJIT_DUMP_EVERYTHING").as_deref() == Ok("1") {
        context.set_dump_everything(true);
    }
    if env::var("CG_GCCJIT_KEEP_INTERMEDIATES").as_deref() == Ok("1") {
        context.set_keep_intermediates(true);
    }
    if env::var("CG_GCCJIT_VERBOSE").as_deref() == Ok("1") {
        context.add_driver_option("-v");
    }

    context
}

pub fn add_pic_option<'gcc>(context: &Context<'gcc>, relocation_model: RelocModel) {
    match relocation_model {
        rustc_target::spec::RelocModel::Static => {
            context.add_command_line_option("-fno-pie");
            context.add_driver_option("-fno-pie");
        }
        rustc_target::spec::RelocModel::Pic => {
            context.add_command_line_option("-fPIC");
            // NOTE: we use both add_command_line_option and add_driver_option because the usage in
            // base (compile_codegen_unit) requires add_command_line_option while the usage
            // in the back::write module (codegen) requires add_driver_option.
            context.add_driver_option("-fPIC");
        }
        rustc_target::spec::RelocModel::Pie => {
            context.add_command_line_option("-fPIE");
            context.add_driver_option("-fPIE");
        }
        model => eprintln!("Unsupported relocation model: {:?}", model),
    }
}
