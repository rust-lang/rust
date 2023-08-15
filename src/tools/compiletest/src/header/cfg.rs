use test_common::{CommentKind, TestComment};

use crate::common::{CompareMode, Config, Debugger};
use crate::header::IgnoreDecision;
use std::collections::HashSet;

const EXTRA_ARCHS: &[&str] = &["spirv"];

pub(super) fn handle_ignore(config: &Config, comment: TestComment<'_>) -> IgnoreDecision {
    match parse_cfg_name_directive(config, &comment, "ignore") {
        MatchOutcome::Match { message, comment } => IgnoreDecision::Ignore {
            reason: match comment {
                Some(comment) => format!("ignored {} ({comment})", message),
                None => format!("ignored {}", message),
            },
        },
        MatchOutcome::NoMatch { .. } => IgnoreDecision::Continue,
        MatchOutcome::Invalid => {
            IgnoreDecision::Error { message: format!("invalid line: {}", comment.comment_str()) }
        }
        MatchOutcome::External => IgnoreDecision::Continue,
        MatchOutcome::NotADirective => IgnoreDecision::Continue,
    }
}

pub(super) fn handle_only(config: &Config, comment: TestComment<'_>) -> IgnoreDecision {
    match parse_cfg_name_directive(config, &comment, "only") {
        MatchOutcome::Match { .. } => IgnoreDecision::Continue,
        MatchOutcome::NoMatch { message, comment } => IgnoreDecision::Ignore {
            reason: match comment {
                Some(comment) => format!("only executed {} ({comment})", message),
                None => format!("only executed {}", message),
            },
        },
        MatchOutcome::Invalid => {
            IgnoreDecision::Error { message: format!("invalid line: {}", comment.comment_str()) }
        }
        MatchOutcome::External => IgnoreDecision::Continue,
        MatchOutcome::NotADirective => IgnoreDecision::Continue,
    }
}

/// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
/// or `normalize-stderr-32bit`.
pub(super) fn parse_cfg_name_directive(
    config: &Config,
    comment: &TestComment<'_>,
    prefix: &str,
) -> MatchOutcome {
    let comment_kind = comment.comment();
    let Some((name, comment)) = comment_kind.parse_name_comment().and_then(|(name, comment)| {
        name.strip_prefix(format!("{}-", prefix).as_str()).map(|stripped| (stripped, comment))
    }) else {
        return MatchOutcome::NotADirective;
    };
    let comment = comment.map(|c| c.trim().trim_start_matches('-').trim().to_owned());

    match comment_kind {
        CommentKind::Compiletest(_) => {
            parse_cfg_name_directive_compiletest(config, name, comment, prefix)
        }
        CommentKind::UiTest(_) => parse_cfg_name_directive_ui_test(config, name, comment),
    }
}

fn parse_cfg_name_directive_ui_test(
    config: &Config,
    name: &str,
    comment: Option<String>,
) -> MatchOutcome {
    let target_cfg = config.target_cfg();

    // Parsing copied from ui_test: https://github.com/oli-obk/ui_test/blob/a18ef37bf3dcccf5a1a631eddd55759fe0b89617/src/parser.rs#L187
    if name == "test" {
        MatchOutcome::Match { message: String::from("always"), comment }
    } else if name == "on-host" {
        unimplemented!("idk what to do about this yet")
    } else if let Some(bits) = name.strip_suffix("bit") {
        let Ok(bits) = bits.parse::<u32>() else {
            // "invalid ignore/only filter ending in 'bit': {bits:?} is not a valid bitwdith"
            return MatchOutcome::Invalid;
        };

        let message = format!("when the pointer width is {}", target_cfg.pointer_width);
        if bits == target_cfg.pointer_width {
            MatchOutcome::Match { message, comment }
        } else {
            MatchOutcome::NoMatch { message, comment }
        }
    } else if let Some(triple_substr) = name.strip_prefix("target-") {
        let message = format!("when the target is {}", config.target);
        if config.target.contains(triple_substr) {
            MatchOutcome::Match { message, comment }
        } else {
            MatchOutcome::NoMatch { message, comment }
        }
    } else if let Some(triple_substr) = name.strip_prefix("host-") {
        let message = format!("when the host is {}", config.host);
        if config.host.contains(triple_substr) {
            MatchOutcome::Match { message, comment }
        } else {
            MatchOutcome::NoMatch { message, comment }
        }
    } else {
        panic!(
            "`{name}` is not a valid condition, expected `on-host`, /[0-9]+bit/, /host-.*/, or /target-.*/"
        )
    }
}

fn parse_cfg_name_directive_compiletest(
    config: &Config,
    name: &str,
    comment: Option<String>,
    prefix: &str,
) -> MatchOutcome {
    macro_rules! condition {
        (
            name: $name:expr,
            $(allowed_names: $allowed_names:expr,)?
            $(condition: $condition:expr,)?
            message: $($message:tt)*
        ) => {{
            // This is not inlined to avoid problems with macro repetitions.
            let format_message = || format!($($message)*);

            if $name.custom_matches(name) {
                if true $(&& $condition)? {
                    return MatchOutcome::Match {
                        message: format_message(),
                        comment,
                    };
                } else {
                    return MatchOutcome::NoMatch{
                        message: format_message(),
                        comment,
                    };
                }
            }
            $(else if $allowed_names.custom_contains(name) {
                return MatchOutcome::NoMatch {
                    message: format_message(),
                    comment,
                };
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
        message: "when the operating system is {name}"
    }
    condition! {
        name: &target_cfg.env,
        allowed_names: &target_cfgs.all_envs,
        message: "when the target environment is {name}"
    }
    condition! {
        name: &target_cfg.os_and_env(),
        allowed_names: &target_cfgs.all_oses_and_envs,
        message: "when the operating system and target environment are {name}"
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

    if prefix == "ignore" {
        // Don't error out for ignore-tidy-* diretives, as those are not handled by compiletest.
        if name.starts_with("tidy-") {
            return MatchOutcome::External;
        }

        // Don't error out for ignore-pass, as that is handled elsewhere.
        if name == "pass" {
            return MatchOutcome::External;
        }

        // Don't error out for ignore-llvm-version, that has a custom syntax and is handled
        // elsewhere.
        if name == "llvm-version" {
            return MatchOutcome::External;
        }

        // Don't error out for ignore-llvm-version, that has a custom syntax and is handled
        // elsewhere.
        if name == "gdb-version" {
            return MatchOutcome::External;
        }
    }

    // Did not match any known condition, emit an error.
    MatchOutcome::Invalid
}

#[derive(Clone, PartialEq, Debug)]
pub(super) enum MatchOutcome {
    /// No match.
    NoMatch { message: String, comment: Option<String> },
    /// Match.
    Match { message: String, comment: Option<String> },
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
