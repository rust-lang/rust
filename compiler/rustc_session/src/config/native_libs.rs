//! Parser for the `-l` command-line option, which links the generated crate to
//! a native library.
//!
//! (There is also a similar but separate syntax for `#[link]` attributes,
//! which have their own parser in `rustc_metadata`.)

use rustc_feature::UnstableFeatures;
use rustc_hir::attrs::NativeLibKind;

use crate::EarlyDiagCtxt;
use crate::config::UnstableOptions;
use crate::utils::NativeLib;

#[cfg(test)]
mod tests;

/// Parses all `-l` options.
pub(crate) fn parse_native_libs(
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    unstable_features: UnstableFeatures,
    matches: &getopts::Matches,
) -> Vec<NativeLib> {
    let cx = ParseNativeLibCx {
        early_dcx,
        unstable_options_enabled: unstable_opts.unstable_options,
        is_nightly: unstable_features.is_nightly_build(),
    };
    matches.opt_strs("l").into_iter().map(|value| parse_native_lib(&cx, &value)).collect()
}

struct ParseNativeLibCx<'a> {
    early_dcx: &'a EarlyDiagCtxt,
    unstable_options_enabled: bool,
    is_nightly: bool,
}

impl ParseNativeLibCx<'_> {
    /// If unstable values are not permitted, exits with a fatal error made by
    /// combining the given strings.
    fn on_unstable_value(&self, message: &str, if_nightly: &str, if_stable: &str) {
        if self.unstable_options_enabled {
            return;
        }

        let suffix = if self.is_nightly { if_nightly } else { if_stable };
        self.early_dcx.early_fatal(format!("{message}{suffix}"));
    }
}

/// Parses the value of a single `-l` option.
fn parse_native_lib(cx: &ParseNativeLibCx<'_>, value: &str) -> NativeLib {
    let NativeLibParts { kind, modifiers, name, new_name } = split_native_lib_value(value);

    let kind = kind.map_or(NativeLibKind::Unspecified, |kind| match kind {
        "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
        "dylib" => NativeLibKind::Dylib { as_needed: None },
        "framework" => NativeLibKind::Framework { as_needed: None },
        "link-arg" => {
            cx.on_unstable_value(
                "library kind `link-arg` is unstable",
                ", the `-Z unstable-options` flag must also be passed to use it",
                " and only accepted on the nightly compiler",
            );
            NativeLibKind::LinkArg
        }
        _ => cx.early_dcx.early_fatal(format!(
            "unknown library kind `{kind}`, expected one of: static, dylib, framework, link-arg"
        )),
    });

    // Provisionally create the result, so that modifiers can modify it.
    let mut native_lib = NativeLib {
        name: name.to_owned(),
        new_name: new_name.map(str::to_owned),
        kind,
        verbatim: None,
    };

    if let Some(modifiers) = modifiers {
        // If multiple modifiers are present, they are separated by commas.
        for modifier in modifiers.split(',') {
            parse_and_apply_modifier(cx, modifier, &mut native_lib);
        }
    }

    if native_lib.name.is_empty() {
        cx.early_dcx.early_fatal("library name must not be empty");
    }

    native_lib
}

/// Parses one of the comma-separated modifiers (prefixed by `+` or `-`), and
/// modifies `native_lib` appropriately.
///
/// Exits with a fatal error if a malformed/unknown/inappropriate modifier is
/// found.
fn parse_and_apply_modifier(cx: &ParseNativeLibCx<'_>, modifier: &str, native_lib: &mut NativeLib) {
    let early_dcx = cx.early_dcx;

    // Split off the leading `+` or `-` into a boolean value.
    let (modifier, value) = match modifier.split_at_checked(1) {
        Some(("+", m)) => (m, true),
        Some(("-", m)) => (m, false),
        _ => cx.early_dcx.early_fatal(
            "invalid linking modifier syntax, expected '+' or '-' prefix \
             before one of: bundle, verbatim, whole-archive, as-needed",
        ),
    };

    // Assigns the value (from `+` or `-`) to an empty `Option<bool>`, or emits
    // a fatal error if the option has already been set.
    let assign_modifier = |opt_bool: &mut Option<bool>| {
        if opt_bool.is_some() {
            let msg = format!("multiple `{modifier}` modifiers in a single `-l` option");
            early_dcx.early_fatal(msg)
        }
        *opt_bool = Some(value);
    };

    // Check that the modifier is applicable to the native lib kind, and apply it.
    match (modifier, &mut native_lib.kind) {
        ("bundle", NativeLibKind::Static { bundle, .. }) => assign_modifier(bundle),
        ("bundle", _) => early_dcx
            .early_fatal("linking modifier `bundle` is only compatible with `static` linking kind"),

        ("verbatim", _) => assign_modifier(&mut native_lib.verbatim),

        ("whole-archive", NativeLibKind::Static { whole_archive, .. }) => {
            assign_modifier(whole_archive)
        }
        ("whole-archive", _) => early_dcx.early_fatal(
            "linking modifier `whole-archive` is only compatible with `static` linking kind",
        ),

        ("as-needed", NativeLibKind::Dylib { as_needed })
        | ("as-needed", NativeLibKind::Framework { as_needed })
        | ("as-needed", NativeLibKind::RawDylib { as_needed }) => {
            cx.on_unstable_value(
                "linking modifier `as-needed` is unstable",
                ", the `-Z unstable-options` flag must also be passed to use it",
                " and only accepted on the nightly compiler",
            );
            assign_modifier(as_needed)
        }
        ("as-needed", _) => early_dcx.early_fatal(
            "linking modifier `as-needed` is only compatible with \
             `dylib` and `framework` linking kinds",
        ),

        _ => early_dcx.early_fatal(format!(
            "unknown linking modifier `{modifier}`, expected one \
             of: bundle, verbatim, whole-archive, as-needed"
        )),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct NativeLibParts<'a> {
    kind: Option<&'a str>,
    modifiers: Option<&'a str>,
    name: &'a str,
    new_name: Option<&'a str>,
}

/// Splits a string of the form `[KIND[:MODIFIERS]=]NAME[:NEW_NAME]` into those
/// individual parts. This cannot fail, but the resulting strings require
/// further validation.
fn split_native_lib_value(value: &str) -> NativeLibParts<'_> {
    // Split the initial value into `[KIND=]NAME`.
    let name = value;
    let (kind, name) = match name.split_once('=') {
        Some((prefix, name)) => (Some(prefix), name),
        None => (None, name),
    };

    // Split the kind part, if present, into `KIND[:MODIFIERS]`.
    let (kind, modifiers) = match kind {
        Some(kind) => match kind.split_once(':') {
            Some((kind, modifiers)) => (Some(kind), Some(modifiers)),
            None => (Some(kind), None),
        },
        None => (None, None),
    };

    // Split the name part into `NAME[:NEW_NAME]`.
    let (name, new_name) = match name.split_once(':') {
        Some((name, new_name)) => (name, Some(new_name)),
        None => (name, None),
    };

    NativeLibParts { kind, modifiers, name, new_name }
}
