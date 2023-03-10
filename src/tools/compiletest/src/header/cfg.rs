use crate::common::{Config, CompareMode, Debugger};
use std::collections::HashSet;

/// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
/// or `normalize-stderr-32bit`.
pub(super) fn parse_cfg_name_directive<'a>(
    config: &Config,
    line: &'a str,
    prefix: &str,
) -> ParsedNameDirective<'a> {
    if !line.as_bytes().starts_with(prefix.as_bytes()) {
        return ParsedNameDirective::invalid();
    }
    if line.as_bytes().get(prefix.len()) != Some(&b'-') {
        return ParsedNameDirective::invalid();
    }
    let line = &line[prefix.len() + 1..];

    let (name, comment) =
        line.split_once(&[':', ' ']).map(|(l, c)| (l, Some(c))).unwrap_or((line, None));

    // Some of the matchers might be "" depending on what the target information is. To avoid
    // problems we outright reject empty directives.
    if name == "" {
        return ParsedNameDirective::invalid();
    }

    let mut outcome = MatchOutcome::Invalid;
    let mut message = None;

    macro_rules! maybe_condition {
        (
            name: $name:expr,
            $(allowed_names: $allowed_names:expr,)?
            $(condition: $condition:expr,)?
            message: $($message:tt)*
        ) => {{
            // This is not inlined to avoid problems with macro repetitions.
            let format_message = || format!($($message)*);

            if outcome != MatchOutcome::Invalid {
                // Ignore all other matches if we already found one
            } else if $name.as_ref().map(|n| n == &name).unwrap_or(false) {
                message = Some(format_message());
                if true $(&& $condition)? {
                    outcome = MatchOutcome::Match;
                } else {
                    outcome = MatchOutcome::NoMatch;
                }
            }
            $(else if $allowed_names.contains(name) {
                message = Some(format_message());
                outcome = MatchOutcome::NoMatch;
            })?
        }};
    }
    macro_rules! condition {
        (
            name: $name:expr,
            $(allowed_names: $allowed_names:expr,)?
            $(condition: $condition:expr,)?
            message: $($message:tt)*
        ) => {
            maybe_condition! {
                name: Some($name),
                $(allowed_names: $allowed_names,)*
                $(condition: $condition,)*
                message: $($message)*
            }
        };
    }
    macro_rules! hashset {
        ($($value:expr),* $(,)?) => {{
            let mut set = HashSet::new();
            $(set.insert($value);)*
            set
        }}
    }

    let target_cfgs = config.target_cfgs();
    let target_cfg = config.target_cfg();

    condition! {
        name: "test",
        message: "always"
    }
    condition! {
        name: &config.target,
        allowed_names: &target_cfgs.all_targets,
        message: "when the target is {name}"
    }
    condition! {
        name: &target_cfg.os,
        allowed_names: &target_cfgs.all_oses,
        message: "when the operative system is {name}"
    }
    condition! {
        name: &target_cfg.env,
        allowed_names: &target_cfgs.all_envs,
        message: "when the target environment is {name}"
    }
    condition! {
        name: &target_cfg.abi,
        allowed_names: &target_cfgs.all_abis,
        message: "when the ABI is {name}"
    }
    condition! {
        name: &target_cfg.arch,
        allowed_names: &target_cfgs.all_archs,
        message: "when the architecture is {name}"
    }
    condition! {
        name: format!("{}bit", target_cfg.pointer_width),
        allowed_names: &target_cfgs.all_pointer_widths,
        message: "when the pointer width is {name}"
    }
    for family in &target_cfg.families {
        condition! {
            name: family,
            allowed_names: &target_cfgs.all_families,
            message: "when the target family is {name}"
        }
    }

    // If something is ignored for emscripten, it likely also needs to be
    // ignored for wasm32-unknown-unknown.
    // `wasm32-bare` is an alias to refer to just wasm32-unknown-unknown
    // (in contrast to `wasm32` which also matches non-bare targets like
    // asmjs-unknown-emscripten).
    condition! {
        name: "emscripten",
        condition: config.target == "wasm32-unknown-unknown",
        message: "when the target is WASM",
    }
    condition! {
        name: "wasm32-bare",
        condition: config.target == "wasm32-unknown-unknown",
        message: "when the target is WASM"
    }

    condition! {
        name: &config.channel,
        allowed_names: hashset!["stable", "beta", "nightly"],
        message: "when the release channel is {name}",
    }
    condition! {
        name: "cross-compile",
        condition: config.target != config.host,
        message: "when cross-compiling"
    }
    condition! {
        name: "endian-big",
        condition: config.is_big_endian(),
        message: "on big-endian targets",
    }
    condition! {
        name: config.stage_id.split('-').next().unwrap(),
        allowed_names: hashset!["stable", "beta", "nightly"],
        message: "when the bootstrapping stage is {name}",
    }
    condition! {
        name: "remote",
        condition: config.remote_test_client.is_some(),
        message: "when running tests remotely",
    }
    condition! {
        name: "debug",
        condition: cfg!(debug_assertions),
        message: "when building with debug assertions",
    }
    maybe_condition! {
        name: config.debugger.as_ref().map(|d| d.to_str()),
        allowed_names: Debugger::VARIANTS
            .iter()
            .map(|v| v.to_str())
            .collect::<HashSet<_>>(),
        message: "when the debugger is {name}",
    }
    maybe_condition! {
        name: config.compare_mode
            .as_ref()
            .map(|d| format!("compare-mode-{}", d.to_str())),
        allowed_names: CompareMode::VARIANTS
            .iter()
            .map(|cm| format!("compare-mode-{}", cm.to_str()))
            .collect::<HashSet<_>>(),
        message: "when comparing with {name}",
    }

    // Don't error out for ignore-tidy-* diretives, as those are not handled by compiletest.
    if prefix == "ignore" && name.starts_with("tidy-") && outcome == MatchOutcome::Invalid {
        outcome = MatchOutcome::External;
    }

    // Don't error out for ignore-pass, as that is handled elsewhere.
    if prefix == "ignore" && name == "pass" && outcome == MatchOutcome::Invalid {
        outcome = MatchOutcome::External;
    }

    ParsedNameDirective {
        name: Some(name),
        comment: comment.map(|c| c.trim().trim_start_matches('-').trim()),
        outcome,
        pretty_reason: message,
    }
}

/// The result of parse_cfg_name_directive.
#[derive(Clone, PartialEq, Debug)]
pub(super) struct ParsedNameDirective<'a> {
    pub(super) name: Option<&'a str>,
    pub(super) pretty_reason: Option<String>,
    pub(super) comment: Option<&'a str>,
    pub(super) outcome: MatchOutcome,
}

impl ParsedNameDirective<'_> {
    fn invalid() -> Self {
        Self { name: None, pretty_reason: None, comment: None, outcome: MatchOutcome::NoMatch }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum MatchOutcome {
    /// No match.
    NoMatch,
    /// Match.
    Match,
    /// The directive was invalid.
    Invalid,
    /// The directive is handled by other parts of our tooling.
    External,
}
