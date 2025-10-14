#[cfg(feature = "master")]
use gccjit::Context;
use rustc_codegen_ssa::target_features;
use rustc_session::Session;
use rustc_target::spec::Architecture;
use smallvec::{SmallVec, smallvec};

fn gcc_features_by_flags(sess: &Session, features: &mut Vec<String>) {
    target_features::retpoline_features_by_flags(sess, features);
    // FIXME: LLVM also sets +reserve-x18 here under some conditions.
}

/// The list of GCC features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
pub(crate) fn global_gcc_features(sess: &Session, diagnostics: bool) -> Vec<String> {
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

    // Features implied by an implicit or explicit `--target`.
    features.extend(sess.target.features.split(',').filter(|v| !v.is_empty()).map(String::from));

    // -Ctarget-features
    target_features::flag_to_backend_features(
        sess,
        diagnostics,
        |feature| to_gcc_features(sess, feature),
        |feature, enable| {
            // We run through `to_gcc_features` when
            // passing requests down to GCC. This means that all in-language
            // features also work on the command line instead of having two
            // different names when the GCC name and the Rust name differ.
            features.extend(
                to_gcc_features(sess, feature)
                    .iter()
                    .flat_map(|feat| to_gcc_features(sess, feat).into_iter())
                    .map(
                        |feature| {
                            if !enable { format!("-{}", feature) } else { feature.to_string() }
                        },
                    ),
            );
        },
    );

    gcc_features_by_flags(sess, &mut features);

    features
}

// To find a list of GCC's names, check https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
pub fn to_gcc_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    // cSpell:disable
    match (sess.target.arch, s) {
        // FIXME: seems like x87 does not exist?
        (Architecture::X86 | Architecture::X86_64, "x87") => smallvec![],
        (Architecture::X86 | Architecture::X86_64, "sse4.2") => smallvec!["sse4.2", "crc32"],
        (Architecture::X86 | Architecture::X86_64, "pclmulqdq") => smallvec!["pclmul"],
        (Architecture::X86 | Architecture::X86_64, "rdrand") => smallvec!["rdrnd"],
        (Architecture::X86 | Architecture::X86_64, "bmi1") => smallvec!["bmi"],
        (Architecture::X86 | Architecture::X86_64, "cmpxchg16b") => smallvec!["cx16"],
        (Architecture::X86 | Architecture::X86_64, "avx512vaes") => smallvec!["vaes"],
        (Architecture::X86 | Architecture::X86_64, "avx512gfni") => smallvec!["gfni"],
        (Architecture::X86 | Architecture::X86_64, "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512vbmi2'.
        (Architecture::X86 | Architecture::X86_64, "avx512vbmi2") => {
            smallvec!["avx512vbmi2", "avx512bw"]
        }
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512bitalg'.
        (Architecture::X86 | Architecture::X86_64, "avx512bitalg") => {
            smallvec!["avx512bitalg", "avx512bw"]
        }
        (Architecture::AArch64, "rcpc2") => smallvec!["rcpc-immo"],
        (Architecture::AArch64, "dpb") => smallvec!["ccpp"],
        (Architecture::AArch64, "dpb2") => smallvec!["ccdp"],
        (Architecture::AArch64, "frintts") => smallvec!["fptoint"],
        (Architecture::AArch64, "fcma") => smallvec!["complxnum"],
        (Architecture::AArch64, "pmuv3") => smallvec!["perfmon"],
        (Architecture::AArch64, "paca") => smallvec!["pauth"],
        (Architecture::AArch64, "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In GCC neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        (Architecture::AArch64, "f32mm") => smallvec!["f32mm", "neon"],
        (Architecture::AArch64, "f64mm") => smallvec!["f64mm", "neon"],
        (Architecture::AArch64, "fhm") => smallvec!["fp16fml", "neon"],
        (Architecture::AArch64, "fp16") => smallvec!["fullfp16", "neon"],
        (Architecture::AArch64, "jsconv") => smallvec!["jsconv", "neon"],
        (Architecture::AArch64, "sve") => smallvec!["sve", "neon"],
        (Architecture::AArch64, "sve2") => smallvec!["sve2", "neon"],
        (Architecture::AArch64, "sve2-aes") => smallvec!["sve2-aes", "neon"],
        (Architecture::AArch64, "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        (Architecture::AArch64, "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        (Architecture::AArch64, "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
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
