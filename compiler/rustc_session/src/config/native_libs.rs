use crate::EarlyDiagCtxt;
use crate::config::nightly_options;
use crate::utils::{NativeLib, NativeLibKind};

pub(crate) fn parse_libs(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> Vec<NativeLib> {
    matches
        .opt_strs("l")
        .into_iter()
        .map(|s| {
            // Parse string of the form "[KIND[:MODIFIERS]=]lib[:new_name]",
            // where KIND is one of "dylib", "framework", "static", "link-arg" and
            // where MODIFIERS are a comma separated list of supported modifiers
            // (bundle, verbatim, whole-archive, as-needed). Each modifier is prefixed
            // with either + or - to indicate whether it is enabled or disabled.
            // The last value specified for a given modifier wins.
            let (name, kind, verbatim) = match s.split_once('=') {
                None => (s, NativeLibKind::Unspecified, None),
                Some((kind, name)) => {
                    let (kind, verbatim) = parse_native_lib_kind(early_dcx, matches, kind);
                    (name.to_string(), kind, verbatim)
                }
            };

            let (name, new_name) = match name.split_once(':') {
                None => (name, None),
                Some((name, new_name)) => (name.to_string(), Some(new_name.to_owned())),
            };
            if name.is_empty() {
                early_dcx.early_fatal("library name must not be empty");
            }
            NativeLib { name, new_name, kind, verbatim }
        })
        .collect()
}

fn parse_native_lib_kind(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    kind: &str,
) -> (NativeLibKind, Option<bool>) {
    let (kind, modifiers) = match kind.split_once(':') {
        None => (kind, None),
        Some((kind, modifiers)) => (kind, Some(modifiers)),
    };

    let kind = match kind {
        "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
        "dylib" => NativeLibKind::Dylib { as_needed: None },
        "framework" => NativeLibKind::Framework { as_needed: None },
        "link-arg" => {
            if !nightly_options::is_unstable_enabled(matches) {
                let why = if nightly_options::match_is_nightly_build(matches) {
                    " and only accepted on the nightly compiler"
                } else {
                    ", the `-Z unstable-options` flag must also be passed to use it"
                };
                early_dcx.early_fatal(format!("library kind `link-arg` is unstable{why}"))
            }
            NativeLibKind::LinkArg
        }
        _ => early_dcx.early_fatal(format!(
            "unknown library kind `{kind}`, expected one of: static, dylib, framework, link-arg"
        )),
    };
    match modifiers {
        None => (kind, None),
        Some(modifiers) => parse_native_lib_modifiers(early_dcx, kind, modifiers, matches),
    }
}

fn parse_native_lib_modifiers(
    early_dcx: &EarlyDiagCtxt,
    mut kind: NativeLibKind,
    modifiers: &str,
    matches: &getopts::Matches,
) -> (NativeLibKind, Option<bool>) {
    let mut verbatim = None;
    for modifier in modifiers.split(',') {
        let (modifier, value) = match modifier.strip_prefix(['+', '-']) {
            Some(m) => (m, modifier.starts_with('+')),
            None => early_dcx.early_fatal(
                "invalid linking modifier syntax, expected '+' or '-' prefix \
                 before one of: bundle, verbatim, whole-archive, as-needed",
            ),
        };

        let report_unstable_modifier = || {
            if !nightly_options::is_unstable_enabled(matches) {
                let why = if nightly_options::match_is_nightly_build(matches) {
                    " and only accepted on the nightly compiler"
                } else {
                    ", the `-Z unstable-options` flag must also be passed to use it"
                };
                early_dcx.early_fatal(format!("linking modifier `{modifier}` is unstable{why}"))
            }
        };
        let assign_modifier = |dst: &mut Option<bool>| {
            if dst.is_some() {
                let msg = format!("multiple `{modifier}` modifiers in a single `-l` option");
                early_dcx.early_fatal(msg)
            } else {
                *dst = Some(value);
            }
        };
        match (modifier, &mut kind) {
            ("bundle", NativeLibKind::Static { bundle, .. }) => assign_modifier(bundle),
            ("bundle", _) => early_dcx.early_fatal(
                "linking modifier `bundle` is only compatible with `static` linking kind",
            ),

            ("verbatim", _) => assign_modifier(&mut verbatim),

            ("whole-archive", NativeLibKind::Static { whole_archive, .. }) => {
                assign_modifier(whole_archive)
            }
            ("whole-archive", _) => early_dcx.early_fatal(
                "linking modifier `whole-archive` is only compatible with `static` linking kind",
            ),

            ("as-needed", NativeLibKind::Dylib { as_needed })
            | ("as-needed", NativeLibKind::Framework { as_needed }) => {
                report_unstable_modifier();
                assign_modifier(as_needed)
            }
            ("as-needed", _) => early_dcx.early_fatal(
                "linking modifier `as-needed` is only compatible with \
                 `dylib` and `framework` linking kinds",
            ),

            // Note: this error also excludes the case with empty modifier
            // string, like `modifiers = ""`.
            _ => early_dcx.early_fatal(format!(
                "unknown linking modifier `{modifier}`, expected one \
                     of: bundle, verbatim, whole-archive, as-needed"
            )),
        }
    }

    (kind, verbatim)
}
