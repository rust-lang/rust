#[cfg(feature = "master")]
use gccjit::Context;
use rustc_codegen_ssa::codegen_attrs::check_tied_features;
use rustc_codegen_ssa::errors::TargetFeatureDisableOrEnable;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::bug;
use rustc_session::Session;
use rustc_target::target_features::{RUSTC_SPECIFIC_FEATURES, Stability};
use smallvec::{SmallVec, smallvec};

use crate::errors::{
    ForbiddenCTargetFeature, PossibleFeature, UnknownCTargetFeature, UnknownCTargetFeaturePrefix,
    UnstableCTargetFeature,
};

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
    features.extend(
        sess.target
            .features
            .split(',')
            .filter(|v| !v.is_empty() && backend_feature_name(v).is_some())
            .map(String::from),
    );

    // -Ctarget-features
    let known_features = sess.target.rust_target_features();
    let mut featsmap = FxHashMap::default();
    let feats = sess
        .opts
        .cg
        .target_feature
        .split(',')
        .filter_map(|s| {
            let enable_disable = match s.chars().next() {
                None => return None,
                Some(c @ ('+' | '-')) => c,
                Some(_) => {
                    if diagnostics {
                        sess.dcx().emit_warn(UnknownCTargetFeaturePrefix { feature: s });
                    }
                    return None;
                }
            };

            // Get the backend feature name, if any.
            // This excludes rustc-specific features, that do not get passed down to GCC.
            let feature = backend_feature_name(s)?;
            // Warn against use of GCC specific feature names on the CLI.
            if diagnostics {
                let feature_state = known_features.iter().find(|&&(v, _, _)| v == feature);
                match feature_state {
                    None => {
                        let rust_feature =
                            known_features.iter().find_map(|&(rust_feature, _, _)| {
                                let gcc_features = to_gcc_features(sess, rust_feature);
                                if gcc_features.contains(&feature)
                                    && !gcc_features.contains(&rust_feature)
                                {
                                    Some(rust_feature)
                                } else {
                                    None
                                }
                            });
                        let unknown_feature = if let Some(rust_feature) = rust_feature {
                            UnknownCTargetFeature {
                                feature,
                                rust_feature: PossibleFeature::Some { rust_feature },
                            }
                        } else {
                            UnknownCTargetFeature { feature, rust_feature: PossibleFeature::None }
                        };
                        sess.dcx().emit_warn(unknown_feature);
                    }
                    Some((_, Stability::Stable, _)) => {}
                    Some((_, Stability::Unstable(_), _)) => {
                        // An unstable feature. Warn about using it.
                        sess.dcx().emit_warn(UnstableCTargetFeature { feature });
                    }
                    Some((_, Stability::Forbidden { .. }, _)) => {
                        sess.dcx().emit_err(ForbiddenCTargetFeature { feature });
                    }
                }

                // FIXME(nagisa): figure out how to not allocate a full hashset here.
                featsmap.insert(feature, enable_disable == '+');
            }

            // ... otherwise though we run through `to_gcc_features` when
            // passing requests down to GCC. This means that all in-language
            // features also work on the command line instead of having two
            // different names when the GCC name and the Rust name differ.
            Some(
                to_gcc_features(sess, feature)
                    .iter()
                    .flat_map(|feat| to_gcc_features(sess, feat).into_iter())
                    .map(|feature| {
                        if enable_disable == '-' {
                            format!("-{}", feature)
                        } else {
                            feature.to_string()
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flatten();
    features.extend(feats);

    if diagnostics {
        if let Some(f) = check_tied_features(sess, &featsmap) {
            sess.dcx().emit_err(TargetFeatureDisableOrEnable {
                features: f,
                span: None,
                missing_features: None,
            });
        }
    }

    features
}

/// Returns a feature name for the given `+feature` or `-feature` string.
///
/// Only allows features that are backend specific (i.e. not [`RUSTC_SPECIFIC_FEATURES`].)
fn backend_feature_name(s: &str) -> Option<&str> {
    // features must start with a `+` or `-`.
    let feature = s.strip_prefix(&['+', '-'][..]).unwrap_or_else(|| {
        bug!("target feature `{}` must begin with a `+` or `-`", s);
    });
    // Rustc-specific feature requests like `+crt-static` or `-crt-static`
    // are not passed down to GCC.
    if RUSTC_SPECIFIC_FEATURES.contains(&feature) {
        return None;
    }
    Some(feature)
}

// To find a list of GCC's names, check https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
pub fn to_gcc_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "sse4.2") => smallvec!["sse4.2", "crc32"],
        ("x86", "pclmulqdq") => smallvec!["pclmul"],
        ("x86", "rdrand") => smallvec!["rdrnd"],
        ("x86", "bmi1") => smallvec!["bmi"],
        ("x86", "cmpxchg16b") => smallvec!["cx16"],
        ("x86", "avx512vaes") => smallvec!["vaes"],
        ("x86", "avx512gfni") => smallvec!["gfni"],
        ("x86", "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512vbmi2'.
        ("x86", "avx512vbmi2") => smallvec!["avx512vbmi2", "avx512bw"],
        // NOTE: seems like GCC requires 'avx512bw' for 'avx512bitalg'.
        ("x86", "avx512bitalg") => smallvec!["avx512bitalg", "avx512bw"],
        ("aarch64", "rcpc2") => smallvec!["rcpc-immo"],
        ("aarch64", "dpb") => smallvec!["ccpp"],
        ("aarch64", "dpb2") => smallvec!["ccdp"],
        ("aarch64", "frintts") => smallvec!["fptoint"],
        ("aarch64", "fcma") => smallvec!["complxnum"],
        ("aarch64", "pmuv3") => smallvec!["perfmon"],
        ("aarch64", "paca") => smallvec!["pauth"],
        ("aarch64", "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In GCC neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        ("aarch64", "f32mm") => smallvec!["f32mm", "neon"],
        ("aarch64", "f64mm") => smallvec!["f64mm", "neon"],
        ("aarch64", "fhm") => smallvec!["fp16fml", "neon"],
        ("aarch64", "fp16") => smallvec!["fullfp16", "neon"],
        ("aarch64", "jsconv") => smallvec!["jsconv", "neon"],
        ("aarch64", "sve") => smallvec!["sve", "neon"],
        ("aarch64", "sve2") => smallvec!["sve2", "neon"],
        ("aarch64", "sve2-aes") => smallvec!["sve2-aes", "neon"],
        ("aarch64", "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        ("aarch64", "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        ("aarch64", "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
        (_, s) => smallvec![s],
    }
}

fn arch_to_gcc(name: &str) -> &str {
    match name {
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
