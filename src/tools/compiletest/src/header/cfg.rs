use crate::common::{CompareMode, Config, Debugger};
use crate::header::IgnoreDecision;
use std::collections::HashSet;

const EXTRA_ARCHS: &[&str] = &["spirv"];

pub(super) fn handle_ignore(config: &Config, line: &str) -> IgnoreDecision {
    let parsed = parse_cfg_name_directive(config, line, "ignore");
    match parsed.outcome {
        MatchOutcome::NoMatch => IgnoreDecision::Continue,
        MatchOutcome::Match => IgnoreDecision::Ignore {
            reason: match parsed.comment {
                Some(comment) => format!("ignored {} ({comment})", parsed.pretty_reason.unwrap()),
                None => format!("ignored {}", parsed.pretty_reason.unwrap()),
            },
        },
        MatchOutcome::Invalid => IgnoreDecision::Error { message: format!("invalid line: {line}") },
        MatchOutcome::External => IgnoreDecision::Continue,
        MatchOutcome::NotADirective => IgnoreDecision::Continue,
    }
}

pub(super) fn handle_only(config: &Config, line: &str) -> IgnoreDecision {
    let parsed = parse_cfg_name_directive(config, line, "only");
    match parsed.outcome {
        MatchOutcome::Match => IgnoreDecision::Continue,
        MatchOutcome::NoMatch => IgnoreDecision::Ignore {
            reason: match parsed.comment {
                Some(comment) => {
                    format!("only executed {} ({comment})", parsed.pretty_reason.unwrap())
                }
                None => format!("only executed {}", parsed.pretty_reason.unwrap()),
            },
        },
        MatchOutcome::Invalid => IgnoreDecision::Error { message: format!("invalid line: {line}") },
        MatchOutcome::External => IgnoreDecision::Continue,
        MatchOutcome::NotADirective => IgnoreDecision::Continue,
    }
}

/// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
/// or `normalize-stderr-32bit`.
pub(super) fn parse_cfg_name_directive<'a>(
    config: &Config,
    line: &'a str,
    prefix: &str,
) -> ParsedNameDirective<'a> {
    if !line.as_bytes().starts_with(prefix.as_bytes()) {
        return ParsedNameDirective::not_a_directive();
    }
    if line.as_bytes().get(prefix.len()) != Some(&b'-') {
        return ParsedNameDirective::not_a_directive();
    }
    let line = &line[prefix.len() + 1..];

    let (name, comment) =
        line.split_once(&[':', ' ']).map(|(l, c)| (l, Some(c))).unwrap_or((line, None));

    // Some of the matchers might be "" depending on what the target information is. To avoid
    // problems we outright reject empty directives.
    if name == "" {
        return ParsedNameDirective::not_a_directive();
    }

    let mut outcome = MatchOutcome::Invalid;
    let mut message = None;

    macro_rules! condition {
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
            } else if $name.custom_matches(name) {
                message = Some(format_message());
                if true $(&& $condition)? {
                    outcome = MatchOutcome::Match;
                } else {
                    outcome = MatchOutcome::NoMatch;
                }
            }
            $(else if $allowed_names.custom_contains(name) {
                message = Some(format_message());
                outcome = MatchOutcome::NoMatch;
            })?
        }};
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
        name: &[
            Some(&*target_cfg.os),
            // If something is ignored for emscripten, it likely also needs to be
            // ignored for wasm32-unknown-unknown.
            (config.target == "wasm32-unknown-unknown").then_some("emscripten"),
        ],
        allowed_names: &target_cfgs.all_oses,
        message: "when the operative system is {name}"
    }
    condition! {
        name: &target_cfg.env,
        allowed_names: &target_cfgs.all_envs,
        message: "when the target environment is {name}"
    }
    condition! {
        name: &target_cfg.os_and_env(),
        allowed_names: &target_cfgs.all_oses_and_envs,
        message: "when the operative system and target environment are {name}"
    }
    condition! {
        name: &target_cfg.abi,
        allowed_names: &target_cfgs.all_abis,
        message: "when the ABI is {name}"
    }
    condition! {
        name: &target_cfg.arch,
        allowed_names: ContainsEither { a: &target_cfgs.all_archs, b: &EXTRA_ARCHS },
        message: "when the architecture is {name}"
    }
    condition! {
        name: format!("{}bit", target_cfg.pointer_width),
        allowed_names: &target_cfgs.all_pointer_widths,
        message: "when the pointer width is {name}"
    }
    condition! {
        name: &*target_cfg.families,
        allowed_names: &target_cfgs.all_families,
        message: "when the target family is {name}"
    }

    // `wasm32-bare` is an alias to refer to just wasm32-unknown-unknown
    // (in contrast to `wasm32` which also matches non-bare targets like
    // asmjs-unknown-emscripten).
    condition! {
        name: "wasm32-bare",
        condition: config.target == "wasm32-unknown-unknown",
        message: "when the target is WASM"
    }

    condition! {
        name: "asmjs",
        condition: config.target.starts_with("asmjs"),
        message: "when the architecture is asm.js",
    }
    condition! {
        name: "thumb",
        condition: config.target.starts_with("thumb"),
        message: "when the architecture is part of the Thumb family"
    }

    // Technically the locally built compiler uses the "dev" channel rather than the "nightly"
    // channel, even though most people don't know or won't care about it. To avoid confusion, we
    // treat the "dev" channel as the "nightly" channel when processing the directive.
    condition! {
        name: if config.channel == "dev" { "nightly" } else { &config.channel },
        allowed_names: &["stable", "beta", "nightly"],
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
        allowed_names: &["stage0", "stage1", "stage2"],
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
    condition! {
        name: config.debugger.as_ref().map(|d| d.to_str()),
        allowed_names: &Debugger::STR_VARIANTS,
        message: "when the debugger is {name}",
    }
    condition! {
        name: config.compare_mode
            .as_ref()
            .map(|d| format!("compare-mode-{}", d.to_str())),
        allowed_names: ContainsPrefixed {
            prefix: "compare-mode-",
            inner: CompareMode::STR_VARIANTS,
        },
        message: "when comparing with {name}",
    }

    if prefix == "ignore" && outcome == MatchOutcome::Invalid {
        // Don't error out for ignore-tidy-* diretives, as those are not handled by compiletest.
        if name.starts_with("tidy-") {
            outcome = MatchOutcome::External;
        }

        // Don't error out for ignore-pass, as that is handled elsewhere.
        if name == "pass" {
            outcome = MatchOutcome::External;
        }

        // Don't error out for ignore-llvm-version, that has a custom syntax and is handled
        // elsewhere.
        if name == "llvm-version" {
            outcome = MatchOutcome::External;
        }

        // Don't error out for ignore-llvm-version, that has a custom syntax and is handled
        // elsewhere.
        if name == "gdb-version" {
            outcome = MatchOutcome::External;
        }
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
    fn not_a_directive() -> Self {
        Self {
            name: None,
            pretty_reason: None,
            comment: None,
            outcome: MatchOutcome::NotADirective,
        }
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
    /// The line is not actually a directive.
    NotADirective,
}

trait CustomContains {
    fn custom_contains(&self, item: &str) -> bool;
}

impl CustomContains for HashSet<String> {
    fn custom_contains(&self, item: &str) -> bool {
        self.contains(item)
    }
}

impl CustomContains for &[&str] {
    fn custom_contains(&self, item: &str) -> bool {
        self.contains(&item)
    }
}

impl<const N: usize> CustomContains for [&str; N] {
    fn custom_contains(&self, item: &str) -> bool {
        self.contains(&item)
    }
}

struct ContainsPrefixed<T: CustomContains> {
    prefix: &'static str,
    inner: T,
}

impl<T: CustomContains> CustomContains for ContainsPrefixed<T> {
    fn custom_contains(&self, item: &str) -> bool {
        match item.strip_prefix(self.prefix) {
            Some(stripped) => self.inner.custom_contains(stripped),
            None => false,
        }
    }
}

struct ContainsEither<'a, A: CustomContains, B: CustomContains> {
    a: &'a A,
    b: &'a B,
}

impl<A: CustomContains, B: CustomContains> CustomContains for ContainsEither<'_, A, B> {
    fn custom_contains(&self, item: &str) -> bool {
        self.a.custom_contains(item) || self.b.custom_contains(item)
    }
}

trait CustomMatches {
    fn custom_matches(&self, name: &str) -> bool;
}

impl CustomMatches for &str {
    fn custom_matches(&self, name: &str) -> bool {
        name == *self
    }
}

impl CustomMatches for String {
    fn custom_matches(&self, name: &str) -> bool {
        name == self
    }
}

impl<T: CustomMatches> CustomMatches for &[T] {
    fn custom_matches(&self, name: &str) -> bool {
        self.iter().any(|m| m.custom_matches(name))
    }
}

impl<const N: usize, T: CustomMatches> CustomMatches for [T; N] {
    fn custom_matches(&self, name: &str) -> bool {
        self.iter().any(|m| m.custom_matches(name))
    }
}

impl<T: CustomMatches> CustomMatches for Option<T> {
    fn custom_matches(&self, name: &str) -> bool {
        match self {
            Some(inner) => inner.custom_matches(name),
            None => false,
        }
    }
}
