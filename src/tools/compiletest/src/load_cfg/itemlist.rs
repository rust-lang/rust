//! Extract metadata from a test file to a basic structure
//!
//! This module handles only the most basic collection of directives and other metadata;
//! [`ItemTest`] just acts as a non-recursive `TokenTree`. We try to keep everything in this file
//! as minimal as possible. That is, this just parses the approximate shape of data but leaves it
//! up to another module to validate the data's content (which helps to raise errors on typos that
//! would otherwise just not get picked up).

use std::{
    cell::{Cell, LazyCell},
    ops::ControlFlow,
    rc::Rc,
    sync::LazyLock,
};

use super::{CommentTy, Error, ErrorExt, LineCol, Result};
use regex::Regex;

const STRSIM_CONFIDENCE: f64 = 0.7;

pub fn parse(s: &str, cty: CommentTy) -> Result<Vec<Item>, Vec<Error>> {
    let mut items = Vec::new();
    let mut errors = Vec::new();

    for (idx, line) in s.lines().enumerate() {
        let line_no = idx + 1;
        match try_match_line(line, cty).line(line_no) {
            Ok(Some(item)) => items.push(Item::new(item, line_no, 1)),
            Ok(None) => (),
            Err(e) => errors.push(e),
        }
    }

    Item::visit_expand_nested(&mut items).map_err(|e| errors.push(e));

    if errors.is_empty() { Ok(items) } else { Err(errors) }
}

/// A type for mapping `CommentTy`s to an associated regex.
type CommentRePats = [(CommentTy, Regex); CommentTy::all().len()];

#[derive(Clone, Debug, PartialEq)]
pub struct Item<'src> {
    pub val: ItemVal<'src>,
    pub pos: LineCol,
    /// Purely a debugging tool; we set this to `true` when we consume the `Item` in some way
    /// when setting the config (other module), which allows us to assert that nothing
    /// accidentally goes unused.
    pub(super) used: Cell<bool>,
}

impl<'src> Item<'src> {
    fn new(item: ItemVal<'src>, line: usize, col: usize) -> Self {
        Self { val: item, pos: LineCol { line, col }, used: Cell::new(false) }
    }

    /// Expand any items that may have nested items.
    fn visit_expand_nested(list: &mut Vec<Item>) -> Result<()> {
        for item in list.iter_mut() {
            match item.val {
                // Revision-specific items need to have their content parsed. E.g.
                // `//@ [abc,def] compile-flags: -O`
                ItemVal::RevisionSpecificItems { revs, content } => {
                    let make_err = || {
                        Err("unable to parse revision directive '{content}'".into()).pos(item.pos)
                    };
                    let content = match_any(
                        |matcher, pass| matcher.try_match_directive(content, pass),
                        make_err,
                    )?
                    .ok_or_else(|| make_err().unwrap_err())?;

                    *item = Item {
                        val: ItemVal::RevisionSpecificExpanded { revs, content: Box::new(content) },
                        pos: item.pos,
                        used: Cell::new(false),
                    };
                }
                ItemVal::RevisionSpecificExpanded { .. } => unreachable!(
                    "if this visit has never happened before, there should be \
                    no expanded nodes."
                ),
                _ => (),
            }
        }

        Ok(())
    }
}

// TODO: parse revisions into `ItemVal`s.

#[derive(Clone, Debug, PartialEq)]
pub enum ItemVal<'src> {
    /* Flags that can only be set to true */
    BuildPass,
    BuildFail,
    CheckPass,
    CheckFail,
    RunPass,
    RunFail,
    NoPreferDynamic,
    NoAutoCheckCfg,

    /* Boolean flags that can be set with `foo-bar` or `no-foo-bar` */
    ShouldIce(bool),
    ShouldFail(bool),
    BuildAuxDocs(bool),
    ForceHost(bool),
    CheckStdout(bool),
    CheckRunResults(bool),
    DontCheckCompilerStdout(bool),
    DontCheckCompilerStderr(bool),
    PrettyExpanded(bool),
    PrettyCompareOnly(bool),
    CheckTestLineNumbersMatch(bool),
    StderrPerBitwidth(bool),
    Incremental(bool),
    DontCheckFailureStatus(bool),
    RunRustfix(bool),
    RustfixOnlyMachineApplicable(bool),
    CompareOutputLinesBySubset(bool),
    KnownBug(bool),
    RemapSrcBase(bool),

    /* Key-value directivesthat can be set with `foo-bar: baz qux` */
    ErrorPattern(&'src str),
    RegexErrorPattern(&'src str),
    CompileFlags(&'src str),
    Edition(&'src str),
    RunFlags(&'src str),
    PrettyMode(&'src str),
    AuxBin(&'src str),
    AuxBuild(&'src str),
    AuxCrate(&'src str),
    AuxCodegenBackend(&'src str),
    ExecEnv(&'src str),
    UnsetExecEnv(&'src str),
    RustcEnv(&'src str),
    UnsetRustcEnv(&'src str),
    ForbidOutput(&'src str),
    FailureStatus(&'src str),
    AssemblyOutput(&'src str),
    TestMirPass(&'src str),
    LlvmCovFlags(&'src str),
    FilecheckFlags(&'src str),
    /// The revisions list
    Revisions(&'src str),

    /* Directives that are single keys but have variable suffixes */
    /// Ignore something specific about this test.
    ///
    /// `//@ ignore-x86`
    Ignore {
        what: &'src str,
    },
    /// Components needed by the test
    ///
    /// `//@ needs-rust-lld`
    Needs {
        what: &'src str,
    },
    /// Run only if conditions are met
    ///
    /// `//@ only-linux`
    Only {
        what: &'src str,
    },
    /// Normalizations make transformations in output files before checking
    ///
    /// `//@ normalize-stdout: foo -> bar`
    /// `//@ normalize-stderr-32bit: bar -> baz`
    /// `//@ normalize-stderr-test: "h[[:xdigit:]]{16}" -> "h[HASH]"`
    Normalize {
        what: &'src str,
    },

    /* Specialized patterns */
    /// Directives that apply to a single revision. Don't process this here; just extract the
    /// key and value. The content will be reprocessed as a new item.
    ///
    /// Note that this does not yet recurse.
    ///
    /// `//@[revname] directive`
    RevisionSpecificItems {
        revs: &'src str,
        content: &'src str,
    },

    /// Once we have parsed the entire file, we re-parse `RevisionSpecificItems` into
    /// `RevisionSpecificExpanded`.
    RevisionSpecificExpanded {
        revs: &'src str,
        content: Box<Self>,
    },

    /* Non-header things that we consume. */
    UiDirective {
        /// If specified, a list of which revisions this should check for
        revisions: Option<&'src str>,
        /// `^`, `^^`, `|`, `~`, etc
        adjust: Option<&'src str>,
        /// Directives for filecheck, including `ERROR`, `WARN`, `HELP`, etc
        content: &'src str,
    },
    /// Directives intended for filecheck.
    ///
    /// `// CHECK:`, `// CHECK-NEXT`, ...
    FileCheckDirective {
        directive: &'src str,
        content: Option<&'src str>,
    },
}

impl<'src> ItemVal<'src> {
    pub const fn to_item(self, line: usize, col: usize) -> Item<'src> {
        Item { val: self, pos: LineCol::new(line, col), used: Cell::new(false) }
    }

    /// True if this is an `//@` directive, i.e. something in a header`
    pub fn is_header_directive(&self) -> bool {
        match self {
            ItemVal::BuildPass
            | ItemVal::BuildFail
            | ItemVal::CheckPass
            | ItemVal::CheckFail
            | ItemVal::RunPass
            | ItemVal::RunFail
            | ItemVal::NoPreferDynamic
            | ItemVal::NoAutoCheckCfg
            | ItemVal::ShouldIce(_)
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
            | ItemVal::Normalize { .. }
            | ItemVal::RevisionSpecificItems { .. }
            | ItemVal::RevisionSpecificExpanded { .. } => true,
            ItemVal::UiDirective { .. } | ItemVal::FileCheckDirective { .. } => false,
        }
    }
}

static ITEM_MATCHERS: [MatchItem; 54] = [
    /* set-once values */
    MatchItem::once("build-pass", ItemVal::BuildPass),
    MatchItem::once("build-fail", ItemVal::BuildFail),
    MatchItem::once("check-pass", ItemVal::CheckPass),
    MatchItem::once("check-fail", ItemVal::CheckFail),
    MatchItem::once("run-pass", ItemVal::RunPass),
    MatchItem::once("run-fail", ItemVal::RunFail),
    MatchItem::once("no-prefer-dynamic", ItemVal::NoPreferDynamic),
    MatchItem::once("no-auto-check-cfg", ItemVal::NoAutoCheckCfg),
    /* boolean flags */
    MatchItem::flag("should-ice", |b| ItemVal::ShouldIce(b)),
    MatchItem::flag("should-fail", |b| ItemVal::ShouldFail(b)),
    MatchItem::flag("build-aux-docs", |b| ItemVal::BuildAuxDocs(b)),
    MatchItem::flag("force-host", |b| ItemVal::ForceHost(b)),
    MatchItem::flag("check-stdout", |b| ItemVal::CheckStdout(b)),
    MatchItem::flag("check-run-results", |b| ItemVal::CheckRunResults(b)),
    MatchItem::flag("dont-check-compiler-stdout", |b| ItemVal::DontCheckCompilerStdout(b)),
    MatchItem::flag("dont-check-compiler-stderr", |b| ItemVal::DontCheckCompilerStderr(b)),
    MatchItem::flag("pretty-expanded", |b| ItemVal::PrettyExpanded(b)),
    MatchItem::flag("pretty-compare-only", |b| ItemVal::PrettyCompareOnly(b)),
    MatchItem::flag("check-test-line-numbers-match", |b| ItemVal::CheckTestLineNumbersMatch(b)),
    MatchItem::flag("stderr-per-bitwidth", |b| ItemVal::StderrPerBitwidth(b)),
    MatchItem::flag("incremental", |b| ItemVal::Incremental(b)),
    MatchItem::flag("dont-check-failure-status", |b| ItemVal::DontCheckFailureStatus(b)),
    MatchItem::flag("run-rustfix", |b| ItemVal::RunRustfix(b)),
    MatchItem::flag("rustfix-only-machine-applicable", |b| {
        ItemVal::RustfixOnlyMachineApplicable(b)
    }),
    MatchItem::flag("compare-output-lines-by-subset", |b| ItemVal::CompareOutputLinesBySubset(b)),
    MatchItem::flag("known-bug", |b| ItemVal::KnownBug(b)),
    MatchItem::flag("remap-src-base", |b| ItemVal::RemapSrcBase(b)),
    /* exact matches */
    MatchItem::map("error-pattern", |s| ItemVal::ErrorPattern(s)),
    MatchItem::map("regex-error-pattern", |s| ItemVal::RegexErrorPattern(s)),
    MatchItem::map("compile-flags", |s| ItemVal::CompileFlags(s)),
    MatchItem::map("run-flags", |s| ItemVal::RunFlags(s)),
    MatchItem::map("pretty-mode", |s| ItemVal::PrettyMode(s)),
    MatchItem::map("aux-bin", |s| ItemVal::AuxBin(s)),
    MatchItem::map("aux-build", |s| ItemVal::AuxBuild(s)),
    MatchItem::map("aux-crate", |s| ItemVal::AuxCrate(s)),
    MatchItem::map("aux-codegen-backend", |s| ItemVal::AuxCodegenBackend(s)),
    MatchItem::map("exec-env", |s| ItemVal::ExecEnv(s)),
    MatchItem::map("unset-exec-env", |s| ItemVal::UnsetExecEnv(s)),
    MatchItem::map("rustc-env", |s| ItemVal::RustcEnv(s)),
    MatchItem::map("unset-rustc-env", |s| ItemVal::UnsetRustcEnv(s)),
    MatchItem::map("forbid-output", |s| ItemVal::ForbidOutput(s)),
    MatchItem::map("failure-status", |s| ItemVal::FailureStatus(s)),
    MatchItem::map("assembly-output", |s| ItemVal::AssemblyOutput(s)),
    MatchItem::map("test-mir-pass", |s| ItemVal::TestMirPass(s)),
    MatchItem::map("llvm-cov-flags", |s| ItemVal::LlvmCovFlags(s)),
    MatchItem::map("filecheck-flags", |s| ItemVal::FilecheckFlags(s)),
    MatchItem::map("revisions", |s| ItemVal::Revisions(s)),
    /* prefix-based rules */
    MatchItem::map_pfx("ignore", |what| ItemVal::Ignore { what }),
    MatchItem::map_pfx("needs", |what| ItemVal::Needs { what }),
    MatchItem::map_pfx("only", |what| ItemVal::Only { what }),
    MatchItem::map_pfx("normalize", |what| ItemVal::Normalize { what }),
    /* regex matchers */
    // Flags for revisions are within a `//@` directive
    MatchItem::re_dir(
        || Regex::new(r"\[(?P<revs>[\w\-,]+)\](?P<content>.*)").unwrap(),
        // Catch cases where an invalid revision name is specified
        Some(|| Regex::new(r"\[.*\].*").unwrap()),
        |c| ItemVal::RevisionSpecificItems {
            revs: c.name("revs").unwrap().as_str().trim(),
            content: c.name("content").unwrap().as_str().trim(),
        },
        "usage: `[rev1,rev-2] directives ...`. Revision names may contain lowercase letters, \
        numbers, and hyphens.",
    ),
    MatchItem::re_global(
        || {
            let ok_template = r"
                COMMENT                 # start with a comment
                (?:\[                   # optional revision spec
                    (?P<revs> \S+ )     # revision content. (not always valid, we check that later)
                ])?
                ~                       # sigil
                (?P<adjust> \| | \^+ )? # adjustments like `^^` or `|`
                \x20                    # trailing space required for style (hex for verbose mode)
                (?P<content>.*)         # the rest of the string
            ";
            cty_re_from_template(ok_template, "COMMENT")
        },
        Some(|| {
            let err_template = r"
                COMMENT
                \s*   # whitespace after the comment is an easy typo
                # Try a couple patterns. Note that these are very general and will also
                # capture correct directives; however, since this RE always runs after the
                # correct one, the goal is just to flag anything that _looks_ like somebody's
                # attempt to write a UI directive.
                (?:
                    ~            # match anything with a sigil
                    | (?:\[.*\]) # or something that looks like a revision
                    | [\^\|]+    # or something that looks like an adjustment
                )
                
            ";
            cty_re_from_template(err_template, "COMMENT")
        }),
        |c| ItemVal::UiDirective {
            revisions: c.name("revs").map(|m| m.as_str().trim()),
            adjust: c.name("adjust").map(|m| m.as_str()),
            content: c.name("content").unwrap().as_str().trim(),
        },
        "usage: `//~ LABEL`, `//[revision]~ LABEL`, `//[rev1,rev2]~ LABEL`, ...",
    ),
    MatchItem::re_global(
        || {
            let ok_template = r"
                COMMENT \s*
                (?P<directive> CHECK[A-Z-]* ) \s*:
                (?P<content> .+ )?
            ";
            cty_re_from_template(ok_template, "COMMENT")
        },
        // Checking for `// .*:` might be too restrictive,
        Some(|| cty_re_from_template(r"COMMENT //\s*\S*\s*:", "COMMENT")),
        |c| ItemVal::FileCheckDirective {
            directive: c.name("directive").unwrap().as_str().trim(),
            content: c.name("content").map(|m| m.as_str().trim()),
        },
        "usage: `//~ LABEL`, `//[revision]~ LABEL`, `//[rev1,rev2]~ LABEL`, ...",
    ),
];

/// Turn a single regex string into one for each comment type. Enables verbose mode for comments
/// in the regex.
fn cty_re_from_template(template: &str, placeholder: &str) -> [(CommentTy, Regex); 3] {
    CommentTy::all()
        .iter()
        .map(|cty| {
            (
                *cty,
                regex::RegexBuilder::new(&template.replace(placeholder, &cty.as_str()))
                    .ignore_whitespace(true)
                    .build()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Whether or not to check for errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Only match exact matches.
    ExactOnly,
    /// Try to locate errors.
    FindErrors,
}

/// Turn syntax into an item
enum MatchItem {
    /// Exact directives are those that are only ever an exact key+value. E.g. `foo` should
    /// match, but `foo-x86` will never match. `Once` may only be set once, not to T/F.
    DirectiveOnce { label: &'static str, v: ItemVal<'static> },

    /// Exact directives are those that are only ever an exact key+value. E.g. `foo` should
    /// match, but `foo-x86` will never match.
    DirectiveFlag { label: &'static str, f: fn(bool) -> ItemVal<'static> },

    /// Exact directives are those that are only ever an exact key+value. E.g. `foo` should
    /// match, but `foo-x86` will never match.
    DirectiveMap { label: &'static str, f: fn(&str) -> ItemVal },

    /// Prefix directives are those that always start with a string but have a specifier after.
    /// E.g. ``
    PrefixDirective { label: &'static str, f: fn(&str) -> ItemVal },

    /// Any herader directives that need to be matched by regex. This regex should NOT contain
    /// `//@`.
    ReDirective {
        /// Regex to parse.
        re: LazyLock<Regex>,
        /// Convert this regex to an item. Do only the bare minimum here! Further parsing should
        /// take place when converting from `ItemList` to `TestProps`.
        f: fn(regex::Captures) -> ItemVal,
        /// Regex of near matches that look generally right, but should raise an error or
        /// warning. This is purely for diagnostics.
        error_re: Option<LazyLock<Regex>>,
        /// Example usage for help string
        help: &'static str,
    },

    /// Any patterns that are not `//@` directives but need to be matched by regex, anywhere in
    /// the file.
    ///
    /// The regexes types for this are a bit unusual; because we need to handle different comment
    /// types (`//`, `#`, `;`, etc), we just always construct one pattern for each.
    ReGlobal {
        /// Regex to parse.
        re: LazyLock<CommentRePats>,
        /// Convert this regex to an item. Do only the bare minimum here.
        f: fn(regex::Captures) -> ItemVal,
        /// Regex of near matches that look generally right, but should raise an error or
        /// warning.
        error_re: Option<LazyLock<CommentRePats>>,
        /// Example usage for help string
        help: &'static str,
    },
}

impl MatchItem {
    /// Construct an exact directive that is a boolean flag
    const fn once(label: &'static str, v: ItemVal<'static>) -> Self {
        Self::DirectiveOnce { label, v }
    }

    /// Construct an exact directive that is a boolean flag
    const fn flag(label: &'static str, f: fn(bool) -> ItemVal<'static>) -> Self {
        Self::DirectiveFlag { label, f }
    }

    /// Construct an exact directive that has
    const fn map(label: &'static str, f: fn(&str) -> ItemVal) -> Self {
        Self::DirectiveMap { label, f }
    }

    /// Construct a prefix directive.
    const fn map_pfx(label: &'static str, f: fn(&str) -> ItemVal) -> Self {
        Self::PrefixDirective { label, f }
    }

    /// Construct a regex directive.
    const fn re_dir(
        re_fn: fn() -> Regex,
        near_fn: Option<fn() -> Regex>,
        f: fn(regex::Captures) -> ItemVal,
        help: &'static str,
    ) -> Self {
        let error_re = match near_fn {
            Some(n_f) => Some(LazyLock::new(n_f)),
            None => None,
        };

        Self::ReDirective { re: LazyLock::new(re_fn), error_re, f, help }
    }

    /// Construct a regex pattern that is not a directive.
    const fn re_global(
        re_fn: fn() -> CommentRePats,
        near_fn: Option<fn() -> CommentRePats>,
        f: fn(regex::Captures) -> ItemVal,
        help: &'static str,
    ) -> Self {
        let error_re = match near_fn {
            Some(n_f) => Some(LazyLock::new(n_f)),
            None => None,
        };

        Self::ReGlobal { re: LazyLock::new(re_fn), error_re, f, help }
    }

    /// Try to find an item that can be present in a directive.
    fn try_match_directive<'src>(&self, mut s: &'src str, pass: Pass) -> ParseResult<'src> {
        match self {
            MatchItem::DirectiveOnce { label, v } => {
                if s == *label {
                    return Ok(Some(v.clone()));
                }

                maybe_err_if_similar(pass, s, label)?;
                Ok(None)
            }

            MatchItem::DirectiveFlag { label, f } => {
                let mut val = true;

                if let Some(neg) = s.strip_prefix("no-") {
                    val = false;
                    s = neg;
                };

                // Found exact match; exit one way or another
                if let Some(rest) = s.strip_prefix(label) {
                    return if rest.is_empty() {
                        Ok(Some(f(val)))
                    } else {
                        Err(format!(
                        "'{label}' is a flag and should not have anything after it; got '{rest}'"
                    )
                    .into())
                    };
                };

                maybe_err_if_similar(pass, s, label)?;
                Ok(None)
            }

            MatchItem::DirectiveMap { label, f } => {
                if let Some((k, v)) = s.split_once(':') {
                    if k == *label && !v.trim().is_empty() {
                        return Ok(Some(f(v.trim())));
                    } else if k == *label {
                        Err(format!("'{label}' expects a key-value pair; empty value"))?;
                    }
                };

                maybe_err_if_similar(pass, s, label)?;
                Ok(None)
            }

            MatchItem::PrefixDirective { label, f } => {
                if let Some(sfx) = s.strip_prefix(label).and_then(|rest| rest.strip_prefix('-')) {
                    return Ok(Some(f(sfx)));
                };

                maybe_err_if_similar(pass, s, label)?;
                Ok(None)
            }

            MatchItem::ReDirective { re, f, error_re, help } => {
                if let Some(caps) = re.captures(s) {
                    Ok(Some(f(caps)))
                } else if pass == Pass::FindErrors
                    && error_re.as_ref().is_some_and(|re| re.is_match(s))
                {
                    Err(format!("invalid directave: {help}").into())
                } else {
                    Ok(None)
                }
            }
            // Globals cannot match directives
            MatchItem::ReGlobal { .. } => Ok(None),
        }
    }

    fn try_match_global<'src>(
        &self,
        s: &'src str,
        cty: CommentTy,
        pass: Pass,
    ) -> ParseResult<'src> {
        match self {
            // No directives can match globals
            MatchItem::DirectiveOnce { .. }
            | MatchItem::DirectiveFlag { .. }
            | MatchItem::DirectiveMap { .. }
            | MatchItem::PrefixDirective { .. }
            | MatchItem::ReDirective { .. } => Ok(None),
            MatchItem::ReGlobal { re, f, error_re, help } => {
                // Find the right regex patterns for our comment types
                let re = re.iter().find_map(|(c, r)| (*c == cty).then_some(r)).unwrap();
                let error_re = error_re.as_ref().map(|re_arr| {
                    re_arr.iter().find_map(|(c, r)| (*c == cty).then_some(r)).unwrap()
                });

                if let Some(caps) = re.captures(s) {
                    Ok(Some(f(caps)))
                } else if pass == Pass::FindErrors
                    && error_re.as_ref().is_some_and(|re| re.is_match(s))
                {
                    Err(format!("invalid directave: {help}").into())
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// Return vals:
///
/// Err(e): something looked wrong
/// Ok(None): nothing found, continue parsing
/// Ok(Some(v)): found a match to handle
type ParseResult<'src> = Result<Option<ItemVal<'src>>>;

fn match_any<'src>(
    f: impl Fn(&MatchItem, Pass) -> ParseResult<'src>,
    default: impl FnOnce() -> ParseResult<'src>,
) -> ParseResult<'src> {
    // Try to find an exact matching pattern.
    for matcher in &ITEM_MATCHERS {
        let res = f(matcher, Pass::ExactOnly);
        match res {
            Ok(Some(_)) | Err(_) => return res,
            Ok(None) => (),
        }
    }

    // Since there were no exact matches, see if anything is a close enough match to
    // cause an error.
    for matcher in &ITEM_MATCHERS {
        let res = f(matcher, Pass::FindErrors);
        match res {
            Ok(Some(ref v)) => panic!("should never return matches on error pass; got {v:?}"),
            Ok(None) => (),
            Err(_) => return res,
        };
    }

    default()
}

fn try_match_line(mut line: &str, cty: CommentTy) -> ParseResult {
    let mut col = 1;
    let end_trimmed = line.trim_end();
    line = end_trimmed.trim_start();

    // This line is a directive; try to match it
    if let Some(mut directive_line) = line.strip_prefix(cty.directive()) {
        directive_line = directive_line.trim();
        let make_err = || {
            Err(format!(
                "unmatched directive syntax; `{}` must be followed by a directive",
                cty.directive()
            )
            .into())
        };

        match_any(|matcher, pass| matcher.try_match_directive(directive_line, pass), make_err)
    } else {
        match_any(|matcher, pass| matcher.try_match_global(line, cty, pass), || Ok(None))
    }
}

/// If the line looks like it should be this directive (starts with the label or is similar),
/// construct an error.
fn maybe_err_if_similar(pass: Pass, line: &str, label: &str) -> Result<()> {
    if pass == Pass::ExactOnly {
        return Ok(());
    }

    let make_err = || Err(format!("did you mean: '{label}'?").into());

    if line.starts_with(label) {
        return make_err();
    }

    let check_str = match line.split_once(':') {
        Some((key, _value)) => key,
        None => {
            let Some(s) = line.split_whitespace().next() else {
                // Empty or only ws string
                return Ok(());
            };
            s
        }
    };

    if strsim::jaro_winkler(check_str, label) > STRSIM_CONFIDENCE { make_err() } else { Ok(()) }
}

#[cfg(test)]
#[path = "test_itemlist.rs"]
mod tests;
