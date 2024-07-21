//! Turn a list of `Items`

use std::{collections::BTreeMap, iter, path::PathBuf, rc::Rc, str::FromStr, sync::LazyLock};

use regex::Regex;

use crate::common::{FailMode, PassMode};

use super::{
    itemlist::{Item, ItemVal},
    CommentTy, Error, ErrorExt, LineCol, Result,
};

const REV_NAME_PAT: &str = r"^[\w-,]+$";
static REV_NAME_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(REV_NAME_PAT).unwrap());

type BoxStr = Box<str>;
type RcStr = Rc<str>;

pub fn entrypoint(gcfg: GlobalConfig, items: &[Item]) -> Result<TestProps> {
    let mut pcx = PassCtx::default();

    for pass in ALL_PASSES {
        pass(items, &mut pcx)?;
    }

    Ok(pcx.into_test_props())
}

/// Used for `ignore-*` type directives
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Platform {
    // TODO we can do better here
    X86,
    Aarch64,
}

pub struct TestProps {
    /// Configuration that is always the same for all tests
    pub global_cfg: Config,
    /// Configuration that only applies to a specific revision
    pub revisions: BTreeMap<RcStr, Config>,
}

/// Configuration for tests that is not configured by the test file.
#[derive(Debug)]
pub struct GlobalConfig {
    /// Directory (if any) to use for incremental compilation.  This is
    /// not set by end-users; rather it is set by the incremental
    /// testing harness and used when generating compilation
    /// arguments. (In particular, it propagates to the aux-builds.)
    pub incremental_dir: Option<PathBuf>,
}

/// Config that can be used for one platform or one
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<RcStr>,
    /// Regexes that should be expected, in order, on standard out
    pub regex_error_patterns: Vec<RcStr>,
    /// Extra flags to pass to the compiler
    pub compile_flags: Vec<RcStr>,
    /// Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Vec<RcStr>,

    pub ui_directives: Vec<UiDirective>,
}

// impl Default for Config {
//     fn default() -> Self {
//         Config {}
//     }
// }

/// Context shared among all passes.
#[derive(Debug, Default)]
struct PassCtx {
    // We use `Option` here so we can't accidentally misinterpret an empty vector as a
    // completed but empty pass.
    config: Option<Config>,
    revisions: Option<Vec<RcStr>>,
    rev_cfg: Option<BTreeMap<RcStr, Config>>,
    ui_directives: Option<Vec<UiDirective>>,
    check_prefixes: Option<Vec<BoxStr>>,
}

impl PassCtx {
    fn into_test_props(self) -> TestProps {
        todo!()
    }
}

const ALL_PASSES: &[fn(&[Item], &mut PassCtx) -> Result<()>] = &[
    pass_initialize,
    pass_check_ordering,
    pass_validate_repetition,
    pass_extract_revisions,
    pass_build_default_config,
    pass_ui_directives,
    pass_build_revision_config,
    pass_filecheck_directives,
    pass_finalize,
];

/// Check that we haven't used this before and messed up our `used` tracking.
fn pass_initialize(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    for item in items {
        assert!(!item.used.get(), "found already used item {item:?}");
    }
    Ok(())
}

/// Verify that header directives come before others (UI directives, filecheck, etc).
fn pass_check_ordering(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    let mut last_header = None;
    let mut first_nonheader = None;

    for item in items {
        if item.val.is_header_directive() {
            last_header = Some(item)
        } else if first_nonheader.is_none() {
            first_nonheader = Some(item)
        }
    }

    match (last_header, first_nonheader) {
        (Some(lh), Some(fnh)) if fnh.pos.line <= lh.pos.line => Err(format!(
            "Header directives should be at the top of the file, before any other \
            directives. Last header {lh:?}, first nonheader {fnh:?}"
        )
        .into()),
        (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) | (None, None) => Ok(()),
    }
}

/// Some keys may only be specified once, or once within a group. Check this early.
fn pass_validate_repetition(items: &[Item], _pcx: &mut PassCtx) -> Result<()> {
    // TODO
    Ok(())
}

/// Find where revisions are listed and extract that. This is its own pass just so we have this
/// information early.
fn pass_extract_revisions(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    let mut iter = items.iter().filter_map(|v| matches!(v.val, ItemVal::Revisions(_)).then_some(v));
    let Some(first) = iter.next() else {
        // Revisions not specified
        pcx.revisions = Some(Vec::new());
        return Ok(());
    };

    assert!(iter.count() == 0, "duplicates should have been checked already");
    first.used.set(true);

    let ItemVal::Revisions(all_revs) = first.val else {
        unreachable!("filtered above");
    };

    let revs: Vec<_> = all_revs.split_whitespace().map(Rc::from).collect();

    for idx in 0..revs.len() {
        let name = &revs[idx];
        if !REV_NAME_RE.is_match(&name) {
            Err(format!("revision '{name}' is not valid. Expected: `{REV_NAME_PAT}`"))?;
        }

        if revs[..idx].contains(&name) {
            Err(format!("revision '{name}' is listed twice"))?;
        }
    }

    pcx.revisions = Some(revs);
    Ok(())
}

/// Construct the config that is used everywhere by default.
fn pass_build_default_config(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    let mut cfg = Config::default();

    for item in items {
        visit_default_config(&item.val, &mut cfg).pos(item.pos)?;
    }

    pcx.config = Some(cfg);
    Ok(())
}

/// Locate config that can apply to revisions.
fn pass_build_revision_config(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    let default_cfg = pcx.config.as_ref().unwrap();
    let all_revs = pcx.revisions.as_ref().unwrap();

    let mut map: BTreeMap<Rc<str>, Config> =
        all_revs.iter().map(|r| (Rc::clone(r), default_cfg.clone())).collect();

    let mut iter = items
        .iter()
        .filter_map(|v| matches!(v.val, ItemVal::RevisionSpecificExpanded { .. }).then_some(v));

    for item in iter {
        let ItemVal::RevisionSpecificExpanded { revs, ref content } = item.val else {
            unreachable!("filtered above");
        };

        for rev in split_validate_revisions(revs, &all_revs) {
            let rev = rev.pos(item.pos)?;
            let mut cfg = map.get_mut(rev).unwrap();
            visit_revision_config(content, cfg).pos(item.pos)?;
        }

        item.used.set(true);
    }

    pcx.rev_cfg = Some(map);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum UiLevel {
    Error,
    Warn,
    Help,
    Note,
}

impl UiLevel {
    const ALL: &'static [&'static str] =
        &["ERROR", "WARN", "WARNING", "SUGGESTION", "HELP", "NOTE"];

    /// Extract self
    fn prep_directive(s: &str) -> Result<(Self, RcStr)> {
        let (dir, rest) = s.split_once(|ch: char| ch.is_whitespace()).unwrap_or((s, ""));
        dir.parse().map(|v| (v, rest.trim().into()))
    }
}

impl FromStr for UiLevel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let ret = match s {
            "ERROR" => Self::Error,
            "WARN" | "WARNING" => Self::Warn,
            "SUGGESTION" | "HELP" => Self::Help,
            "NOTE" => Self::Note,
            _ => Err(format!("unknown revision '{s}. Must be one of '{:?}'", UiLevel::ALL))?,
        };
        Ok(ret)
    }
}

#[derive(Clone, Debug)]
pub struct UiDirective {
    line: usize,
    requires_preceding: bool,
    level: UiLevel,
    content: RcStr,
}

impl UiDirective {
    fn new_to_cfg(
        items: &[Item],
        base_line: usize,
        level: UiLevel,
        content: &RcStr,
        adjust: Option<&str>,
        cfg: &mut Config,
    ) -> Result<()> {
        let (offset, requires_preceding) = if let Some(adj) = adjust {
            if adj == "|" {
                (1, true)
            } else if adj.bytes().all(|b| b == b'^') {
                (adj.bytes().len(), false)
            } else {
                Err(format!("invalid adjuster `{adj}`"))?
            }
        } else {
            (0, false)
        };

        let line = match base_line.checked_sub(offset) {
            Some(0) | None => Err(format!("an offset of {offset} points outside of the file"))?,
            Some(v) => v,
        };

        let val = Self { line, level, requires_preceding, content: Rc::clone(content) };
        cfg.ui_directives.push(val);

        Ok(())
    }
}

/// Extract UI error directives, apply them to revisions, and validate their position adjusters.
fn pass_ui_directives(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    let all_revs = pcx.revisions.as_ref().unwrap();
    let mut iter = items
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| matches!(v.val, ItemVal::UiDirective { .. }).then_some((idx, v)));

    for (idx, item) in iter {
        let ItemVal::UiDirective { revisions, adjust, content } = item.val else {
            unreachable!("filtered above");
        };
        let (level, content) = UiLevel::prep_directive(content)?;

        if let Some(revs_str) = revisions {
            for rev in split_validate_revisions(revs_str, &all_revs) {
                let rev = rev.pos(item.pos)?;
                let cfg = pcx.rev_cfg.as_mut().unwrap().get_mut(rev).unwrap();

                UiDirective::new_to_cfg(items, item.pos.line, level, &content, adjust, cfg);
            }
        } else {
            // If the revision is unspecified, applly to all revisions including default
            for cfg in pcx
                .rev_cfg
                .as_mut()
                .unwrap()
                .values_mut()
                .chain(iter::once(pcx.config.as_mut().unwrap()))
            {
                UiDirective::new_to_cfg(items, item.pos.line, level, &content, adjust, cfg);
            }
        }

        item.used.set(true);
    }

    Ok(())
}

/// Extract filecheck directives and validate they match revisions.
fn pass_filecheck_directives(items: &[Item], pcx: &mut PassCtx) -> Result<()> {
    Ok(())
}

/// Just check that no items that haven't been consumed in some way.
fn pass_finalize(items: &[Item], vctx: &mut PassCtx) -> Result<()> {
    for item in items {
        assert!(item.used.get(), "found unused item {item:?}");
        item.used.set(false);
    }
    Ok(())
}

/// Handle a single item and update `cfg` to match.
fn visit_default_config(item: &ItemVal, cfg: &mut Config) -> Result<()> {
    match item {
        ItemVal::BuildPass => todo!(),
        ItemVal::BuildFail => todo!(),
        ItemVal::CheckPass => todo!(),
        ItemVal::CheckFail => todo!(),
        ItemVal::RunPass => todo!(),
        ItemVal::RunFail => todo!(),
        ItemVal::NoPreferDynamic => todo!(),
        ItemVal::NoAutoCheckCfg => todo!(),
        ItemVal::ShouldIce(_) => todo!(),
        ItemVal::ShouldFail(_) => todo!(),
        ItemVal::BuildAuxDocs(_) => todo!(),
        ItemVal::ForceHost(_) => todo!(),
        ItemVal::CheckStdout(_) => todo!(),
        ItemVal::CheckRunResults(_) => todo!(),
        ItemVal::DontCheckCompilerStdout(_) => todo!(),
        ItemVal::DontCheckCompilerStderr(_) => todo!(),
        ItemVal::PrettyExpanded(_) => todo!(),
        ItemVal::PrettyCompareOnly(_) => todo!(),
        ItemVal::CheckTestLineNumbersMatch(_) => todo!(),
        ItemVal::StderrPerBitwidth(_) => todo!(),
        ItemVal::Incremental(_) => todo!(),
        ItemVal::DontCheckFailureStatus(_) => todo!(),
        ItemVal::RunRustfix(_) => todo!(),
        ItemVal::RustfixOnlyMachineApplicable(_) => todo!(),
        ItemVal::CompareOutputLinesBySubset(_) => todo!(),
        ItemVal::KnownBug(_) => todo!(),
        ItemVal::RemapSrcBase(_) => todo!(),
        ItemVal::ErrorPattern(_) => todo!(),
        ItemVal::RegexErrorPattern(_) => todo!(),
        ItemVal::CompileFlags(_) => todo!(),
        ItemVal::Edition(_) => todo!(),
        ItemVal::RunFlags(_) => todo!(),
        ItemVal::PrettyMode(_) => todo!(),
        ItemVal::AuxBin(_) => todo!(),
        ItemVal::AuxBuild(_) => todo!(),
        ItemVal::AuxCrate(_) => todo!(),
        ItemVal::AuxCodegenBackend(_) => todo!(),
        ItemVal::ExecEnv(_) => todo!(),
        ItemVal::UnsetExecEnv(_) => todo!(),
        ItemVal::RustcEnv(_) => todo!(),
        ItemVal::UnsetRustcEnv(_) => todo!(),
        ItemVal::ForbidOutput(_) => todo!(),
        ItemVal::FailureStatus(_) => todo!(),
        ItemVal::AssemblyOutput(_) => todo!(),
        ItemVal::TestMirPass(_) => todo!(),
        ItemVal::LlvmCovFlags(_) => todo!(),
        ItemVal::FilecheckFlags(_) => todo!(),
        ItemVal::Revisions(_) => todo!(),
        ItemVal::Ignore { what } => todo!(),
        ItemVal::Needs { what } => todo!(),
        ItemVal::Only { what } => todo!(),
        ItemVal::Normalize { what } => todo!(),
        ItemVal::RevisionSpecificItems { .. } => unreachable!("should have been expanded"),
        ItemVal::RevisionSpecificExpanded { revs, content } => todo!(),
        ItemVal::UiDirective { revisions, adjust, content } => todo!(),
        ItemVal::FileCheckDirective { .. } => todo!(),
    }

    Ok(())
}

/// Handle a single item that applies only to a revision.
fn visit_revision_config(val: &ItemVal, cfg: &mut Config) -> Result<()> {
    match &val {
        // Some directives are not allowed per-revision
        ItemVal::BuildPass
        | ItemVal::BuildFail
        | ItemVal::CheckPass
        | ItemVal::CheckFail
        | ItemVal::RunPass
        | ItemVal::RunFail
        | ItemVal::NoPreferDynamic
        | ItemVal::NoAutoCheckCfg => Err("TODO")?,
        // Most directives can forward to the default config
        ItemVal::ShouldIce(_)
        | ItemVal::ShouldFail(_)
        | ItemVal::BuildAuxDocs(_)
        | ItemVal::ForceHost(_)
        | ItemVal::CheckStdout(_)
        | ItemVal::CheckRunResults(_)
        | ItemVal::DontCheckCompilerStdout(_)
        | ItemVal::DontCheckCompilerStderr(_)
        | ItemVal::PrettyExpanded(_)
        | ItemVal::PrettyCompareOnly(_)
        | ItemVal::CheckTestLineNumbersMatch(_)
        | ItemVal::StderrPerBitwidth(_)
        | ItemVal::Incremental(_)
        | ItemVal::DontCheckFailureStatus(_)
        | ItemVal::RunRustfix(_)
        | ItemVal::RustfixOnlyMachineApplicable(_)
        | ItemVal::CompareOutputLinesBySubset(_)
        | ItemVal::KnownBug(_)
        | ItemVal::RemapSrcBase(_)
        | ItemVal::ErrorPattern(_)
        | ItemVal::RegexErrorPattern(_)
        | ItemVal::CompileFlags(_)
        | ItemVal::Edition(_)
        | ItemVal::RunFlags(_)
        | ItemVal::PrettyMode(_)
        | ItemVal::AuxBin(_)
        | ItemVal::AuxBuild(_)
        | ItemVal::AuxCrate(_)
        | ItemVal::AuxCodegenBackend(_)
        | ItemVal::ExecEnv(_)
        | ItemVal::UnsetExecEnv(_)
        | ItemVal::RustcEnv(_)
        | ItemVal::UnsetRustcEnv(_)
        | ItemVal::ForbidOutput(_)
        | ItemVal::FailureStatus(_)
        | ItemVal::AssemblyOutput(_)
        | ItemVal::TestMirPass(_)
        | ItemVal::LlvmCovFlags(_)
        | ItemVal::FilecheckFlags(_)
        | ItemVal::Revisions(_)
        | ItemVal::Ignore { .. }
        | ItemVal::Needs { .. }
        | ItemVal::Only { .. }
        | ItemVal::Normalize { .. } => visit_default_config(val, cfg)?,
        ItemVal::RevisionSpecificExpanded { revs, content } => Err("don't to that TODO")?,
        ItemVal::RevisionSpecificItems { .. } => unreachable!("should have been expanded"),
        ItemVal::UiDirective { .. } | ItemVal::FileCheckDirective { .. } => {
            unreachable!("global directives shouldn't happen here")
        }
    }

    Ok(())
}

/// Split revisions by comma and ensure they exist in a list
fn split_validate_revisions<'a>(
    revs_str: &'a str,
    all_revs: &'a [RcStr],
) -> impl Iterator<Item = Result<&'a RcStr>> + 'a {
    revs_str.split(',').map(str::trim).map(move |s| {
        all_revs
            .iter()
            .find(|r| ***r == *s)
            .ok_or_else(|| format!("revision 's' was not found. Available: {all_revs:?}").into())
    })
}

#[cfg(test)]
#[path = "test_prepare.rs"]
mod tests;
