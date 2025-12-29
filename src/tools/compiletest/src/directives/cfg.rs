use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use crate::common::{CompareMode, Config, Debugger};
use crate::directives::{DirectiveLine, IgnoreDecision};

const EXTRA_ARCHS: &[&str] = &["spirv"];

const EXTERNAL_IGNORES_LIST: &[&str] = &[
    // tidy-alphabetical-start
    "ignore-backends",
    "ignore-gdb-version",
    "ignore-llvm-version",
    "ignore-pass",
    // tidy-alphabetical-end
];

/// Directive names that begin with `ignore-`, but are disregarded by this
/// module because they are handled elsewhere.
pub(crate) static EXTERNAL_IGNORES_SET: LazyLock<HashSet<&str>> =
    LazyLock::new(|| EXTERNAL_IGNORES_LIST.iter().copied().collect());

pub(super) fn handle_ignore(
    conditions: &PreparedConditions,
    line: &DirectiveLine<'_>,
) -> IgnoreDecision {
    let parsed = parse_cfg_name_directive(conditions, line, "ignore-");
    let line = line.display();

    match parsed.outcome {
        MatchOutcome::NoMatch => IgnoreDecision::Continue,
        MatchOutcome::Match => IgnoreDecision::Ignore {
            reason: match parsed.comment {
                Some(comment) => format!("ignored {} ({comment})", parsed.pretty_reason.unwrap()),
                None => format!("ignored {}", parsed.pretty_reason.unwrap()),
            },
        },
        MatchOutcome::Invalid => IgnoreDecision::Error { message: format!("invalid line: {line}") },
        MatchOutcome::NotHandledHere => IgnoreDecision::Continue,
    }
}

pub(super) fn handle_only(
    conditions: &PreparedConditions,
    line: &DirectiveLine<'_>,
) -> IgnoreDecision {
    let parsed = parse_cfg_name_directive(conditions, line, "only-");
    let line = line.display();

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
        MatchOutcome::NotHandledHere => IgnoreDecision::Continue,
    }
}

/// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
/// or `only-windows`.
fn parse_cfg_name_directive<'a>(
    conditions: &PreparedConditions,
    line: &'a DirectiveLine<'a>,
    prefix: &str,
) -> ParsedNameDirective<'a> {
    let Some(name) = line.name.strip_prefix(prefix) else {
        return ParsedNameDirective::not_handled_here();
    };

    if prefix == "ignore-" && EXTERNAL_IGNORES_SET.contains(line.name) {
        return ParsedNameDirective::not_handled_here();
    }

    // FIXME(Zalathar): This currently allows either a space or a colon, and
    // treats any "value" after a colon as though it were a remark.
    // We should instead forbid the colon syntax for these directives.
    let comment = line
        .remark_after_space()
        .or_else(|| line.value_after_colon())
        .map(|c| c.trim().trim_start_matches('-').trim());

    if let Some(cond) = conditions.conds.get(name) {
        ParsedNameDirective {
            pretty_reason: Some(Arc::clone(&cond.message_when_ignored)),
            comment,
            outcome: if cond.value { MatchOutcome::Match } else { MatchOutcome::NoMatch },
        }
    } else {
        ParsedNameDirective { pretty_reason: None, comment, outcome: MatchOutcome::Invalid }
    }
}

/// Uses information about the current target (and all targets) to pre-compute
/// a value (true or false) for a number of "conditions". Those conditions can
/// then be used by `ignore-*` and `only-*` directives.
pub(crate) fn prepare_conditions(config: &Config) -> PreparedConditions {
    let cfgs = config.target_cfgs();
    let current = &cfgs.current;

    let mut builder = ConditionsBuilder::new();

    // Some condition names overlap (e.g. "macabi" is both an env and an ABI),
    // so the order in which conditions are added is significant.
    // Whichever condition registers that name _first_ will take precedence.
    // (See `ConditionsBuilder::build`.)

    builder.cond("test", true, "always");
    builder.cond("auxiliary", true, "used by another main test file");

    for target in &cfgs.all_targets {
        builder.cond(target, *target == config.target, &format!("when the target is {target}"));
    }
    for os in &cfgs.all_oses {
        builder.cond(os, *os == current.os, &format!("when the operating system is {os}"));
    }
    for env in &cfgs.all_envs {
        builder.cond(env, *env == current.env, &format!("when the target environment is {env}"));
    }
    for os_and_env in &cfgs.all_oses_and_envs {
        builder.cond(
            os_and_env,
            *os_and_env == current.os_and_env(),
            &format!("when the operating system and target environment are {os_and_env}"),
        );
    }
    for abi in &cfgs.all_abis {
        builder.cond(abi, *abi == current.abi, &format!("when the ABI is {abi}"));
    }
    for arch in cfgs.all_archs.iter().map(String::as_str).chain(EXTRA_ARCHS.iter().copied()) {
        builder.cond(arch, *arch == current.arch, &format!("when the architecture is {arch}"));
    }
    for n_bit in &cfgs.all_pointer_widths {
        builder.cond(
            n_bit,
            *n_bit == format!("{}bit", current.pointer_width),
            &format!("when the pointer width is {n_bit}"),
        );
    }
    for family in &cfgs.all_families {
        builder.cond(
            family,
            current.families.contains(family),
            &format!("when the target family is {family}"),
        )
    }

    builder.cond(
        "thumb",
        config.target.starts_with("thumb"),
        "when the architecture is part of the Thumb family",
    );

    // The "arch" of `i586-` targets is "x86", so for more specific matching
    // we have to resort to a string-prefix check.
    builder.cond("i586", config.matches_arch("i586"), "when the subarchitecture is i586");
    // FIXME(Zalathar): Use proper target vendor information instead?
    builder.cond("apple", config.target.contains("apple"), "when the target vendor is Apple");
    // FIXME(Zalathar): Support all known binary formats, not just ELF?
    builder.cond("elf", current.binary_format == "elf", "when the target binary format is ELF");
    builder.cond("enzyme", config.has_enzyme, "when rustc is built with LLVM Enzyme");

    // Technically the locally built compiler uses the "dev" channel rather than the "nightly"
    // channel, even though most people don't know or won't care about it. To avoid confusion, we
    // treat the "dev" channel as the "nightly" channel when processing the directive.
    for channel in ["stable", "beta", "nightly"] {
        let curr_channel = match config.channel.as_str() {
            "dev" => "nightly",
            ch => ch,
        };
        builder.cond(
            channel,
            channel == curr_channel,
            &format!("when the release channel is {channel}"),
        );
    }

    builder.cond("cross-compile", config.target != config.host, "when cross-compiling");
    builder.cond("endian-big", config.is_big_endian(), "on big-endian targets");

    for stage in ["stage0", "stage1", "stage2"] {
        builder.cond(
            stage,
            stage == format!("stage{}", config.stage),
            &format!("when the bootstrapping stage is {stage}"),
        );
    }

    builder.cond("remote", config.remote_test_client.is_some(), "when running tests remotely");
    builder.cond(
        "rustc-debug-assertions",
        config.with_rustc_debug_assertions,
        "when rustc is built with debug assertions",
    );
    builder.cond(
        "std-debug-assertions",
        config.with_std_debug_assertions,
        "when std is built with debug assertions",
    );
    builder.cond(
        "std-remap-debuginfo",
        config.with_std_remap_debuginfo,
        "when std is built with remapping of debuginfo",
    );

    for &debugger in Debugger::STR_VARIANTS {
        builder.cond(
            debugger,
            Some(debugger) == config.debugger.as_ref().map(Debugger::to_str),
            &format!("when the debugger is {debugger}"),
        );
    }

    for &compare_mode in CompareMode::STR_VARIANTS {
        builder.cond(
            &format!("compare-mode-{compare_mode}"),
            Some(compare_mode) == config.compare_mode.as_ref().map(CompareMode::to_str),
            &format!("when comparing with compare-mode-{compare_mode}"),
        );
    }

    // Coverage tests run the same test file in multiple modes.
    // If a particular test should not be run in one of the modes, ignore it
    // with "ignore-coverage-map" or "ignore-coverage-run".
    for test_mode in ["coverage-map", "coverage-run"] {
        builder.cond(
            test_mode,
            test_mode == config.mode.to_str(),
            &format!("when the test mode is {test_mode}"),
        );
    }

    for rustc_abi in &cfgs.all_rustc_abis {
        builder.cond(
            &format!("rustc_abi-{rustc_abi}"),
            Some(rustc_abi) == current.rustc_abi.as_ref(),
            &format!("when the target `rustc_abi` is rustc_abi-{rustc_abi}"),
        );
    }

    // FIXME(Zalathar): Ideally this should be configured by a command-line
    // flag, not an environment variable.
    builder.cond(
        "dist",
        std::env::var("COMPILETEST_ENABLE_DIST_TESTS").as_deref() == Ok("1"),
        "when performing tests on dist toolchain",
    );

    builder.build()
}

/// The result of parse_cfg_name_directive.
#[derive(Clone, PartialEq, Debug)]
pub(super) struct ParsedNameDirective<'a> {
    pub(super) pretty_reason: Option<Arc<str>>,
    pub(super) comment: Option<&'a str>,
    pub(super) outcome: MatchOutcome,
}

impl ParsedNameDirective<'_> {
    fn not_handled_here() -> Self {
        Self { pretty_reason: None, comment: None, outcome: MatchOutcome::NotHandledHere }
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
    /// The directive should be ignored by this module, because it is handled elsewhere.
    NotHandledHere,
}

#[derive(Debug)]
pub(crate) struct PreparedConditions {
    /// Maps the "bare" name of each condition to a structure indicating
    /// whether the condition is true or false for the target being tested.
    conds: HashMap<Arc<str>, Cond>,
}

#[derive(Debug)]
struct Cond {
    /// Bare condition name without an ignore/only prefix, e.g. `aarch64` or `windows`.
    bare_name: Arc<str>,

    /// Is this condition true or false for the target being tested, based on
    /// the config that was used to prepare these conditions?
    ///
    /// For example, the condition `windows` is true on Windows targets.
    value: bool,

    /// Message fragment to show when a test is ignored based on this condition
    /// being true or false, e.g. "when the architecture is aarch64".
    message_when_ignored: Arc<str>,
}

struct ConditionsBuilder {
    conds: Vec<Cond>,
}

impl ConditionsBuilder {
    fn new() -> Self {
        Self { conds: vec![] }
    }

    fn cond(&mut self, bare_name: &str, value: bool, message_when_ignored: &str) {
        self.conds.push(Cond {
            bare_name: Arc::<str>::from(bare_name),
            value,
            message_when_ignored: Arc::<str>::from(message_when_ignored),
        });
    }

    fn build(self) -> PreparedConditions {
        let conds = self
            .conds
            .into_iter()
            // Build the map in reverse order, so that conditions declared
            // earlier have priority over ones declared later.
            .rev()
            .map(|cond| (Arc::clone(&cond.bare_name), cond))
            .collect::<HashMap<_, _>>();
        PreparedConditions { conds }
    }
}
