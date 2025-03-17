use rustc_abi::ExternAbi;
use rustc_ast::AttrId;
use rustc_ast::attr::AttributeExt;
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::stable_hasher::{
    HashStable, StableCompare, StableHasher, ToStableHashKey,
};
use rustc_error_messages::{DiagMessage, MultiSpan};
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefPathHash;
use rustc_hir::{HashStableContext, HirId, ItemLocalId};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
pub use rustc_span::edition::Edition;
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span, Symbol, sym};
use serde::{Deserialize, Serialize};

pub use self::Level::*;

pub mod builtin;

#[macro_export]
macro_rules! pluralize {
    // Pluralize based on count (e.g., apples)
    ($x:expr) => {
        if $x == 1 { "" } else { "s" }
    };
    ("has", $x:expr) => {
        if $x == 1 { "has" } else { "have" }
    };
    ("is", $x:expr) => {
        if $x == 1 { "is" } else { "are" }
    };
    ("was", $x:expr) => {
        if $x == 1 { "was" } else { "were" }
    };
    ("this", $x:expr) => {
        if $x == 1 { "this" } else { "these" }
    };
}

/// Grammatical tool for displaying messages to end users in a nice form.
///
/// Take a list of items and a function to turn those items into a `String`, and output a display
/// friendly comma separated list of those items.
// FIXME(estebank): this needs to be changed to go through the translation machinery.
pub fn listify<T>(list: &[T], fmt: impl Fn(&T) -> String) -> Option<String> {
    Some(match list {
        [only] => fmt(&only),
        [others @ .., last] => format!(
            "{} and {}",
            others.iter().map(|i| fmt(i)).collect::<Vec<_>>().join(", "),
            fmt(&last),
        ),
        [] => return None,
    })
}

/// Indicates the confidence in the correctness of a suggestion.
///
/// All suggestions are marked with an `Applicability`. Tools use the applicability of a suggestion
/// to determine whether it should be automatically applied or if the user should be consulted
/// before applying the suggestion.
#[derive(Copy, Clone, Debug, Hash, Encodable, Decodable, Serialize, Deserialize)]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Applicability {
    /// The suggestion is definitely what the user intended, or maintains the exact meaning of the code.
    /// This suggestion should be automatically applied.
    ///
    /// In case of multiple `MachineApplicable` suggestions (whether as part of
    /// the same `multipart_suggestion` or not), all of them should be
    /// automatically applied.
    MachineApplicable,

    /// The suggestion may be what the user intended, but it is uncertain. The suggestion should
    /// result in valid Rust code if it is applied.
    MaybeIncorrect,

    /// The suggestion contains placeholders like `(...)` or `{ /* fields */ }`. The suggestion
    /// cannot be applied automatically because it will not result in valid Rust code. The user
    /// will need to fill in the placeholders.
    HasPlaceholders,

    /// The applicability of the suggestion is unknown.
    Unspecified,
}

/// Each lint expectation has a `LintExpectationId` assigned by the `LintLevelsBuilder`.
/// Expected diagnostics get the lint level `Expect` which stores the `LintExpectationId`
/// to match it with the actual expectation later on.
///
/// The `LintExpectationId` has to be stable between compilations, as diagnostic
/// instances might be loaded from cache. Lint messages can be emitted during an
/// `EarlyLintPass` operating on the AST and during a `LateLintPass` traversing the
/// HIR tree. The AST doesn't have enough information to create a stable id. The
/// `LintExpectationId` will instead store the [`AttrId`] defining the expectation.
/// These `LintExpectationId` will be updated to use the stable [`HirId`] once the
/// AST has been lowered. The transformation is done by the `LintLevelsBuilder`
///
/// Each lint inside the `expect` attribute is tracked individually, the `lint_index`
/// identifies the lint inside the attribute and ensures that the IDs are unique.
///
/// The index values have a type of `u16` to reduce the size of the `LintExpectationId`.
/// It's reasonable to assume that no user will define 2^16 attributes on one node or
/// have that amount of lints listed. `u16` values should therefore suffice.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Encodable, Decodable)]
pub enum LintExpectationId {
    /// Used for lints emitted during the `EarlyLintPass`. This id is not
    /// hash stable and should not be cached.
    Unstable { attr_id: AttrId, lint_index: Option<u16> },
    /// The [`HirId`] that the lint expectation is attached to. This id is
    /// stable and can be cached. The additional index ensures that nodes with
    /// several expectations can correctly match diagnostics to the individual
    /// expectation.
    Stable { hir_id: HirId, attr_index: u16, lint_index: Option<u16> },
}

impl LintExpectationId {
    pub fn is_stable(&self) -> bool {
        match self {
            LintExpectationId::Unstable { .. } => false,
            LintExpectationId::Stable { .. } => true,
        }
    }

    pub fn get_lint_index(&self) -> Option<u16> {
        let (LintExpectationId::Unstable { lint_index, .. }
        | LintExpectationId::Stable { lint_index, .. }) = self;

        *lint_index
    }

    pub fn set_lint_index(&mut self, new_lint_index: Option<u16>) {
        let (LintExpectationId::Unstable { lint_index, .. }
        | LintExpectationId::Stable { lint_index, .. }) = self;

        *lint_index = new_lint_index
    }
}

impl<HCX: rustc_hir::HashStableContext> HashStable<HCX> for LintExpectationId {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        match self {
            LintExpectationId::Stable { hir_id, attr_index, lint_index: Some(lint_index) } => {
                hir_id.hash_stable(hcx, hasher);
                attr_index.hash_stable(hcx, hasher);
                lint_index.hash_stable(hcx, hasher);
            }
            _ => {
                unreachable!(
                    "HashStable should only be called for filled and stable `LintExpectationId`"
                )
            }
        }
    }
}

impl<HCX: rustc_hir::HashStableContext> ToStableHashKey<HCX> for LintExpectationId {
    type KeyType = (DefPathHash, ItemLocalId, u16, u16);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HCX) -> Self::KeyType {
        match self {
            LintExpectationId::Stable { hir_id, attr_index, lint_index: Some(lint_index) } => {
                let (def_path_hash, lint_idx) = hir_id.to_stable_hash_key(hcx);
                (def_path_hash, lint_idx, *attr_index, *lint_index)
            }
            _ => {
                unreachable!("HashStable should only be called for a filled `LintExpectationId`")
            }
        }
    }
}

/// Setting for how to handle a lint.
///
/// See: <https://doc.rust-lang.org/rustc/lints/levels.html>
#[derive(
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Debug,
    Hash,
    Encodable,
    Decodable,
    HashStable_Generic
)]
pub enum Level {
    /// The `allow` level will not issue any message.
    Allow,
    /// The `expect` level will suppress the lint message but in turn produce a message
    /// if the lint wasn't issued in the expected scope. `Expect` should not be used as
    /// an initial level for a lint.
    ///
    /// Note that this still means that the lint is enabled in this position and should
    /// be emitted, this will in turn fulfill the expectation and suppress the lint.
    ///
    /// See RFC 2383.
    ///
    /// Requires a [`LintExpectationId`] to later link a lint emission to the actual
    /// expectation. It can be ignored in most cases.
    Expect,
    /// The `warn` level will produce a warning if the lint was violated, however the
    /// compiler will continue with its execution.
    Warn,
    /// This lint level is a special case of [`Warn`], that can't be overridden. This is used
    /// to ensure that a lint can't be suppressed. This lint level can currently only be set
    /// via the console and is therefore session specific.
    ///
    /// Requires a [`LintExpectationId`] to fulfill expectations marked via the
    /// `#[expect]` attribute, that will still be suppressed due to the level.
    ForceWarn,
    /// The `deny` level will produce an error and stop further execution after the lint
    /// pass is complete.
    Deny,
    /// `Forbid` is equivalent to the `deny` level but can't be overwritten like the previous
    /// levels.
    Forbid,
}

impl Level {
    /// Converts a level to a lower-case string.
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Allow => "allow",
            Level::Expect => "expect",
            Level::Warn => "warn",
            Level::ForceWarn => "force-warn",
            Level::Deny => "deny",
            Level::Forbid => "forbid",
        }
    }

    /// Converts a lower-case string to a level. This will never construct the expect
    /// level as that would require a [`LintExpectationId`].
    pub fn from_str(x: &str) -> Option<Self> {
        match x {
            "allow" => Some(Level::Allow),
            "warn" => Some(Level::Warn),
            "deny" => Some(Level::Deny),
            "forbid" => Some(Level::Forbid),
            "expect" | _ => None,
        }
    }

    /// Converts an `Attribute` to a level.
    pub fn from_attr(attr: &impl AttributeExt) -> Option<(Self, Option<LintExpectationId>)> {
        attr.name().and_then(|name| Self::from_symbol(name, || Some(attr.id())))
    }

    /// Converts a `Symbol` to a level.
    pub fn from_symbol(
        s: Symbol,
        id: impl FnOnce() -> Option<AttrId>,
    ) -> Option<(Self, Option<LintExpectationId>)> {
        match s {
            sym::allow => Some((Level::Allow, None)),
            sym::expect => {
                if let Some(attr_id) = id() {
                    Some((
                        Level::Expect,
                        Some(LintExpectationId::Unstable { attr_id, lint_index: None }),
                    ))
                } else {
                    None
                }
            }
            sym::warn => Some((Level::Warn, None)),
            sym::deny => Some((Level::Deny, None)),
            sym::forbid => Some((Level::Forbid, None)),
            _ => None,
        }
    }

    pub fn to_cmd_flag(self) -> &'static str {
        match self {
            Level::Warn => "-W",
            Level::Deny => "-D",
            Level::Forbid => "-F",
            Level::Allow => "-A",
            Level::ForceWarn => "--force-warn",
            Level::Expect => {
                unreachable!("the expect level does not have a commandline flag")
            }
        }
    }

    pub fn is_error(self) -> bool {
        match self {
            Level::Allow | Level::Expect | Level::Warn | Level::ForceWarn => false,
            Level::Deny | Level::Forbid => true,
        }
    }
}

/// Specification of a single lint.
#[derive(Copy, Clone, Debug)]
pub struct Lint {
    /// A string identifier for the lint.
    ///
    /// This identifies the lint in attributes and in command-line arguments.
    /// In those contexts it is always lowercase, but this field is compared
    /// in a way which is case-insensitive for ASCII characters. This allows
    /// `declare_lint!()` invocations to follow the convention of upper-case
    /// statics without repeating the name.
    ///
    /// The name is written with underscores, e.g., "unused_imports".
    /// On the command line, underscores become dashes.
    ///
    /// See <https://rustc-dev-guide.rust-lang.org/diagnostics.html#lint-naming>
    /// for naming guidelines.
    pub name: &'static str,

    /// Default level for the lint.
    ///
    /// See <https://rustc-dev-guide.rust-lang.org/diagnostics.html#diagnostic-levels>
    /// for guidelines on choosing a default level.
    pub default_level: Level,

    /// Description of the lint or the issue it detects.
    ///
    /// e.g., "imports that are never used"
    pub desc: &'static str,

    /// Starting at the given edition, default to the given lint level. If this is `None`, then use
    /// `default_level`.
    pub edition_lint_opts: Option<(Edition, Level)>,

    /// `true` if this lint is reported even inside expansions of external macros.
    pub report_in_external_macro: bool,

    pub future_incompatible: Option<FutureIncompatibleInfo>,

    /// `true` if this lint is being loaded by another tool (e.g. Clippy).
    pub is_externally_loaded: bool,

    /// `Some` if this lint is feature gated, otherwise `None`.
    pub feature_gate: Option<Symbol>,

    pub crate_level_only: bool,

    /// `true` if this lint should not be filtered out under any circustamces
    /// (e.g. the unknown_attributes lint)
    pub eval_always: bool,
}

/// Extra information for a future incompatibility lint.
#[derive(Copy, Clone, Debug)]
pub struct FutureIncompatibleInfo {
    /// e.g., a URL for an issue/PR/RFC or error code
    pub reference: &'static str,
    /// The reason for the lint used by diagnostics to provide
    /// the right help message
    pub reason: FutureIncompatibilityReason,
    /// Whether to explain the reason to the user.
    ///
    /// Set to false for lints that already include a more detailed
    /// explanation.
    pub explain_reason: bool,
    /// If set to `true`, this will make future incompatibility warnings show up in cargo's
    /// reports.
    ///
    /// When a future incompatibility warning is first inroduced, set this to `false`
    /// (or, rather, don't override the default). This allows crate developers an opportunity
    /// to fix the warning before blasting all dependents with a warning they can't fix
    /// (dependents have to wait for a new release of the affected crate to be published).
    ///
    /// After a lint has been in this state for a while, consider setting this to true, so it
    /// warns for everyone. It is a good signal that it is ready if you can determine that all
    /// or most affected crates on crates.io have been updated.
    pub report_in_deps: bool,
}

/// The reason for future incompatibility
///
/// Future-incompatible lints come in roughly two categories:
///
/// 1. There was a mistake in the compiler (such as a soundness issue), and
///    we're trying to fix it, but it may be a breaking change.
/// 2. A change across an Edition boundary, typically used for the
///    introduction of new language features that can't otherwise be
///    introduced in a backwards-compatible way.
///
/// See <https://rustc-dev-guide.rust-lang.org/bug-fix-procedure.html> and
/// <https://rustc-dev-guide.rust-lang.org/diagnostics.html#future-incompatible-lints>
/// for more information.
#[derive(Copy, Clone, Debug)]
pub enum FutureIncompatibilityReason {
    /// This will be an error in a future release for all editions
    ///
    /// Choose this variant when you are first introducing a "future
    /// incompatible" warning that is intended to eventually be fixed in the
    /// future.
    ///
    /// After a lint has been in this state for a while and you feel like it is ready to graduate
    /// to warning everyone, consider setting [`FutureIncompatibleInfo::report_in_deps`] to true.
    /// (see it's documentation for more guidance)
    ///
    /// After some period of time, lints with this variant can be turned into
    /// hard errors (and the lint removed). Preferably when there is some
    /// confidence that the number of impacted projects is very small (few
    /// should have a broken dependency in their dependency tree).
    FutureReleaseError,
    /// Code that changes meaning in some way in a
    /// future release.
    ///
    /// Choose this variant when the semantics of existing code is changing,
    /// (as opposed to [`FutureIncompatibilityReason::FutureReleaseError`],
    /// which is for when code is going to be rejected in the future).
    FutureReleaseSemanticsChange,
    /// Previously accepted code that will become an
    /// error in the provided edition
    ///
    /// Choose this variant for code that you want to start rejecting across
    /// an edition boundary. This will automatically include the lint in the
    /// `rust-20xx-compatibility` lint group, which is used by `cargo fix
    /// --edition` to do migrations. The lint *should* be auto-fixable with
    /// [`Applicability::MachineApplicable`].
    ///
    /// The lint can either be `Allow` or `Warn` by default. If it is `Allow`,
    /// users usually won't see this warning unless they are doing an edition
    /// migration manually or there is a problem during the migration (cargo's
    /// automatic migrations will force the level to `Warn`). If it is `Warn`
    /// by default, users on all editions will see this warning (only do this
    /// if you think it is important for everyone to be aware of the change,
    /// and to encourage people to update their code on all editions).
    ///
    /// See also [`FutureIncompatibilityReason::EditionSemanticsChange`] if
    /// you have code that is changing semantics across the edition (as
    /// opposed to being rejected).
    EditionError(Edition),
    /// Code that changes meaning in some way in
    /// the provided edition
    ///
    /// This is the same as [`FutureIncompatibilityReason::EditionError`],
    /// except for situations where the semantics change across an edition. It
    /// slightly changes the text of the diagnostic, but is otherwise the
    /// same.
    EditionSemanticsChange(Edition),
    /// This will be an error in the provided edition *and* in a future
    /// release.
    ///
    /// This variant a combination of [`FutureReleaseError`] and [`EditionError`].
    /// This is useful in rare cases when we want to have "preview" of a breaking
    /// change in an edition, but do a breaking change later on all editions anyway.
    ///
    /// [`EditionError`]: FutureIncompatibilityReason::EditionError
    /// [`FutureReleaseError`]: FutureIncompatibilityReason::FutureReleaseError
    EditionAndFutureReleaseError(Edition),
    /// This will change meaning in the provided edition *and* in a future
    /// release.
    ///
    /// This variant a combination of [`FutureReleaseSemanticsChange`]
    /// and [`EditionSemanticsChange`]. This is useful in rare cases when we
    /// want to have "preview" of a breaking change in an edition, but do a
    /// breaking change later on all editions anyway.
    ///
    /// [`EditionSemanticsChange`]: FutureIncompatibilityReason::EditionSemanticsChange
    /// [`FutureReleaseSemanticsChange`]: FutureIncompatibilityReason::FutureReleaseSemanticsChange
    EditionAndFutureReleaseSemanticsChange(Edition),
    /// A custom reason.
    ///
    /// Choose this variant if the built-in text of the diagnostic of the
    /// other variants doesn't match your situation. This is behaviorally
    /// equivalent to
    /// [`FutureIncompatibilityReason::FutureReleaseError`].
    Custom(&'static str),
}

impl FutureIncompatibilityReason {
    pub fn edition(self) -> Option<Edition> {
        match self {
            Self::EditionError(e)
            | Self::EditionSemanticsChange(e)
            | Self::EditionAndFutureReleaseError(e)
            | Self::EditionAndFutureReleaseSemanticsChange(e) => Some(e),

            FutureIncompatibilityReason::FutureReleaseError
            | FutureIncompatibilityReason::FutureReleaseSemanticsChange
            | FutureIncompatibilityReason::Custom(_) => None,
        }
    }
}

impl FutureIncompatibleInfo {
    pub const fn default_fields_for_macro() -> Self {
        FutureIncompatibleInfo {
            reference: "",
            reason: FutureIncompatibilityReason::FutureReleaseError,
            explain_reason: true,
            report_in_deps: false,
        }
    }
}

impl Lint {
    pub const fn default_fields_for_macro() -> Self {
        Lint {
            name: "",
            default_level: Level::Forbid,
            desc: "",
            edition_lint_opts: None,
            is_externally_loaded: false,
            report_in_external_macro: false,
            future_incompatible: None,
            feature_gate: None,
            crate_level_only: false,
            eval_always: false,
        }
    }

    /// Gets the lint's name, with ASCII letters converted to lowercase.
    pub fn name_lower(&self) -> String {
        self.name.to_ascii_lowercase()
    }

    pub fn default_level(&self, edition: Edition) -> Level {
        self.edition_lint_opts
            .filter(|(e, _)| *e <= edition)
            .map(|(_, l)| l)
            .unwrap_or(self.default_level)
    }
}

/// Identifies a lint known to the compiler.
#[derive(Clone, Copy, Debug)]
pub struct LintId {
    // Identity is based on pointer equality of this field.
    pub lint: &'static Lint,
}

impl PartialEq for LintId {
    fn eq(&self, other: &LintId) -> bool {
        std::ptr::eq(self.lint, other.lint)
    }
}

impl Eq for LintId {}

impl std::hash::Hash for LintId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = self.lint as *const Lint;
        ptr.hash(state);
    }
}

impl LintId {
    /// Gets the `LintId` for a `Lint`.
    pub fn of(lint: &'static Lint) -> LintId {
        LintId { lint }
    }

    pub fn lint_name_raw(&self) -> &'static str {
        self.lint.name
    }

    /// Gets the name of the lint.
    pub fn to_string(&self) -> String {
        self.lint.name_lower()
    }
}

impl<HCX> HashStable<HCX> for LintId {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.lint_name_raw().hash_stable(hcx, hasher);
    }
}

impl<HCX> ToStableHashKey<HCX> for LintId {
    type KeyType = &'static str;

    #[inline]
    fn to_stable_hash_key(&self, _: &HCX) -> &'static str {
        self.lint_name_raw()
    }
}

impl StableCompare for LintId {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    fn stable_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.lint_name_raw().cmp(&other.lint_name_raw())
    }
}

#[derive(Debug)]
pub struct AmbiguityErrorDiag {
    pub msg: String,
    pub span: Span,
    pub label_span: Span,
    pub label_msg: String,
    pub note_msg: String,
    pub b1_span: Span,
    pub b1_note_msg: String,
    pub b1_help_msgs: Vec<String>,
    pub b2_span: Span,
    pub b2_note_msg: String,
    pub b2_help_msgs: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum DeprecatedSinceKind {
    InEffect,
    InFuture,
    InVersion(String),
}

// This could be a closure, but then implementing derive trait
// becomes hacky (and it gets allocated).
#[derive(Debug)]
pub enum BuiltinLintDiag {
    AbsPathWithModule(Span),
    ProcMacroDeriveResolutionFallback {
        span: Span,
        ns: Namespace,
        ident: Ident,
    },
    MacroExpandedMacroExportsAccessedByAbsolutePaths(Span),
    ElidedLifetimesInPaths(usize, Span, bool, Span),
    UnknownCrateTypes {
        span: Span,
        candidate: Option<Symbol>,
    },
    UnusedImports {
        remove_whole_use: bool,
        num_to_remove: usize,
        remove_spans: Vec<Span>,
        test_module_span: Option<Span>,
        span_snippets: Vec<String>,
    },
    RedundantImport(Vec<(Span, bool)>, Ident),
    DeprecatedMacro {
        suggestion: Option<Symbol>,
        suggestion_span: Span,
        note: Option<Symbol>,
        path: String,
        since_kind: DeprecatedSinceKind,
    },
    MissingAbi(Span, ExternAbi),
    UnusedDocComment(Span),
    UnusedBuiltinAttribute {
        attr_name: Symbol,
        macro_name: String,
        invoc_span: Span,
    },
    PatternsInFnsWithoutBody {
        span: Span,
        ident: Ident,
        is_foreign: bool,
    },
    LegacyDeriveHelpers(Span),
    OrPatternsBackCompat(Span, String),
    ReservedPrefix(Span, String),
    /// `'r#` in edition < 2021.
    RawPrefix(Span),
    /// `##` or `#"` in edition < 2024.
    ReservedString {
        is_string: bool,
        suggestion: Span,
    },
    HiddenUnicodeCodepoints {
        label: String,
        count: usize,
        span_label: Span,
        labels: Option<Vec<(char, Span)>>,
        escape: bool,
        spans: Vec<(char, Span)>,
    },
    TrailingMacro(bool, Ident),
    BreakWithLabelAndLoop(Span),
    UnicodeTextFlow(Span, String),
    UnexpectedCfgName((Symbol, Span), Option<(Symbol, Span)>),
    UnexpectedCfgValue((Symbol, Span), Option<(Symbol, Span)>),
    DeprecatedWhereclauseLocation(Span, Option<(Span, String)>),
    MissingUnsafeOnExtern {
        suggestion: Span,
    },
    SingleUseLifetime {
        /// Span of the parameter which declares this lifetime.
        param_span: Span,
        /// Span of the code that should be removed when eliding this lifetime.
        /// This span should include leading or trailing comma.
        deletion_span: Option<Span>,
        /// Span of the single use, or None if the lifetime is never used.
        /// If true, the lifetime will be fully elided.
        use_span: Option<(Span, bool)>,
        ident: Ident,
    },
    NamedArgumentUsedPositionally {
        /// Span where the named argument is used by position and will be replaced with the named
        /// argument name
        position_sp_to_replace: Option<Span>,
        /// Span where the named argument is used by position and is used for lint messages
        position_sp_for_msg: Option<Span>,
        /// Span where the named argument's name is (so we know where to put the warning message)
        named_arg_sp: Span,
        /// String containing the named arguments name
        named_arg_name: String,
        /// Indicates if the named argument is used as a width/precision for formatting
        is_formatting_arg: bool,
    },
    ByteSliceInPackedStructWithDerive {
        // FIXME: enum of byte/string
        ty: String,
    },
    UnusedExternCrate {
        span: Span,
        removal_span: Span,
    },
    ExternCrateNotIdiomatic {
        vis_span: Span,
        ident_span: Span,
    },
    AmbiguousGlobImports {
        diag: AmbiguityErrorDiag,
    },
    AmbiguousGlobReexports {
        /// The name for which collision(s) have occurred.
        name: String,
        /// The name space for which the collision(s) occurred in.
        namespace: String,
        /// Span where the name is first re-exported.
        first_reexport_span: Span,
        /// Span where the same name is also re-exported.
        duplicate_reexport_span: Span,
    },
    HiddenGlobReexports {
        /// The name of the local binding which shadows the glob re-export.
        name: String,
        /// The namespace for which the shadowing occurred in.
        namespace: String,
        /// The glob reexport that is shadowed by the local binding.
        glob_reexport_span: Span,
        /// The local binding that shadows the glob reexport.
        private_item_span: Span,
    },
    UnusedQualifications {
        /// The span of the unnecessarily-qualified path to remove.
        removal_span: Span,
    },
    UnsafeAttrOutsideUnsafe {
        attribute_name_span: Span,
        sugg_spans: (Span, Span),
    },
    AssociatedConstElidedLifetime {
        elided: bool,
        span: Span,
        lifetimes_in_scope: MultiSpan,
    },
    RedundantImportVisibility {
        span: Span,
        max_vis: String,
        import_vis: String,
    },
    UnknownDiagnosticAttribute {
        span: Span,
        typo_name: Option<Symbol>,
    },
    MacroUseDeprecated,
    UnusedMacroUse,
    PrivateExternCrateReexport {
        source: Ident,
        extern_crate_span: Span,
    },
    UnusedLabel,
    MacroIsPrivate(Ident),
    UnusedMacroDefinition(Symbol),
    MacroRuleNeverUsed(usize, Symbol),
    UnstableFeature(DiagMessage),
    AvoidUsingIntelSyntax,
    AvoidUsingAttSyntax,
    IncompleteInclude,
    UnnameableTestItems,
    DuplicateMacroAttribute,
    CfgAttrNoAttributes,
    MissingFragmentSpecifier,
    MetaVariableStillRepeating(MacroRulesNormalizedIdent),
    MetaVariableWrongOperator,
    DuplicateMatcherBinding,
    UnknownMacroVariable(MacroRulesNormalizedIdent),
    UnusedCrateDependency {
        extern_crate: Symbol,
        local_crate: Symbol,
    },
    IllFormedAttributeInput {
        suggestions: Vec<String>,
    },
    InnerAttributeUnstable {
        is_macro: bool,
    },
    OutOfScopeMacroCalls {
        span: Span,
        path: String,
        location: String,
    },
    UnexpectedBuiltinCfg {
        cfg: String,
        cfg_name: Symbol,
        controlled_by: &'static str,
    },
}

/// Lints that are buffered up early on in the `Session` before the
/// `LintLevels` is calculated.
#[derive(Debug)]
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
    pub span: Option<MultiSpan>,

    /// The `NodeId` of the AST node that generated the lint.
    pub node_id: NodeId,

    /// A lint Id that can be passed to
    /// `rustc_lint::early::EarlyContextAndPass::check_id`.
    pub lint_id: LintId,

    /// Customization of the `Diag<'_>` for the lint.
    pub diagnostic: BuiltinLintDiag,
}

#[derive(Default, Debug)]
pub struct LintBuffer {
    pub map: FxIndexMap<NodeId, Vec<BufferedEarlyLint>>,
}

impl LintBuffer {
    pub fn add_early_lint(&mut self, early_lint: BufferedEarlyLint) {
        self.map.entry(early_lint.node_id).or_default().push(early_lint);
    }

    pub fn take(&mut self, id: NodeId) -> Vec<BufferedEarlyLint> {
        // FIXME(#120456) - is `swap_remove` correct?
        self.map.swap_remove(&id).unwrap_or_default()
    }

    pub fn buffer_lint(
        &mut self,
        lint: &'static Lint,
        node_id: NodeId,
        span: impl Into<MultiSpan>,
        diagnostic: BuiltinLintDiag,
    ) {
        self.add_early_lint(BufferedEarlyLint {
            lint_id: LintId::of(lint),
            node_id,
            span: Some(span.into()),
            diagnostic,
        });
    }
}

pub type RegisteredTools = FxIndexSet<Ident>;

/// Declares a static item of type `&'static Lint`.
///
/// See <https://rustc-dev-guide.rust-lang.org/diagnostics.html> for
/// documentation and guidelines on writing lints.
///
/// The macro call should start with a doc comment explaining the lint
/// which will be embedded in the rustc user documentation book. It should
/// be written in markdown and have a format that looks like this:
///
/// ```rust,ignore (doc-example)
/// /// The `my_lint_name` lint detects [short explanation here].
/// ///
/// /// ### Example
/// ///
/// /// ```rust
/// /// [insert a concise example that triggers the lint]
/// /// ```
/// ///
/// /// {{produces}}
/// ///
/// /// ### Explanation
/// ///
/// /// This should be a detailed explanation of *why* the lint exists,
/// /// and also include suggestions on how the user should fix the problem.
/// /// Try to keep the text simple enough that a beginner can understand,
/// /// and include links to other documentation for terminology that a
/// /// beginner may not be familiar with. If this is "allow" by default,
/// /// it should explain why (are there false positives or other issues?). If
/// /// this is a future-incompatible lint, it should say so, with text that
/// /// looks roughly like this:
/// ///
/// /// This is a [future-incompatible] lint to transition this to a hard
/// /// error in the future. See [issue #xxxxx] for more details.
/// ///
/// /// [issue #xxxxx]: https://github.com/rust-lang/rust/issues/xxxxx
/// ```
///
/// The `{{produces}}` tag will be automatically replaced with the output from
/// the example by the build system. If the lint example is too complex to run
/// as a simple example (for example, it needs an extern crate), mark the code
/// block with `ignore` and manually replace the `{{produces}}` line with the
/// expected output in a `text` code block.
///
/// If this is a rustdoc-only lint, then only include a brief introduction
/// with a link with the text `[rustdoc book]` so that the validator knows
/// that this is for rustdoc only (see BROKEN_INTRA_DOC_LINKS as an example).
///
/// Commands to view and test the documentation:
///
/// * `./x.py doc --stage=1 src/doc/rustc --open`: Builds the rustc book and opens it.
/// * `./x.py test src/tools/lint-docs`: Validates that the lint docs have the
///   correct style, and that the code example actually emits the expected
///   lint.
///
/// If you have already built the compiler, and you want to make changes to
/// just the doc comments, then use the `--keep-stage=0` flag with the above
/// commands to avoid rebuilding the compiler.
#[macro_export]
macro_rules! declare_lint {
    ($(#[$attr:meta])* $vis: vis $NAME: ident, $Level: ident, $desc: expr) => (
        $crate::declare_lint!(
            $(#[$attr])* $vis $NAME, $Level, $desc,
        );
    );
    ($(#[$attr:meta])* $vis: vis $NAME: ident, $Level: ident, $desc: expr,
     $(@eval_always = $eval_always:literal)?
     $(@feature_gate = $gate:ident;)?
     $(@future_incompatible = FutureIncompatibleInfo {
        reason: $reason:expr,
        $($field:ident : $val:expr),* $(,)*
     }; )?
     $(@edition $lint_edition:ident => $edition_level:ident;)?
     $($v:ident),*) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::Lint = &$crate::Lint {
            name: stringify!($NAME),
            default_level: $crate::$Level,
            desc: $desc,
            is_externally_loaded: false,
            $($v: true,)*
            $(feature_gate: Some(rustc_span::sym::$gate),)?
            $(future_incompatible: Some($crate::FutureIncompatibleInfo {
                reason: $reason,
                $($field: $val,)*
                ..$crate::FutureIncompatibleInfo::default_fields_for_macro()
            }),)?
            $(edition_lint_opts: Some(($crate::Edition::$lint_edition, $crate::$edition_level)),)?
            $(eval_always: $eval_always,)?
            ..$crate::Lint::default_fields_for_macro()
        };
    );
}

#[macro_export]
macro_rules! declare_tool_lint {
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level: ident, $desc: expr
        $(, @eval_always = $eval_always:literal)?
        $(, @feature_gate = $gate:ident;)?
    ) => (
        $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, false $(, @eval_always = $eval_always)? $(, @feature_gate = $gate;)?}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        report_in_external_macro: $rep:expr
        $(, @eval_always = $eval_always: literal)?
        $(, @feature_gate = $gate:ident;)?
    ) => (
         $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, $rep  $(, @eval_always = $eval_always)? $(, @feature_gate = $gate;)?}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        $external:expr
        $(, @eval_always = $eval_always: literal)?
        $(, @feature_gate = $gate:ident;)?
    ) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::Lint = &$crate::Lint {
            name: &concat!(stringify!($tool), "::", stringify!($NAME)),
            default_level: $crate::$Level,
            desc: $desc,
            edition_lint_opts: None,
            report_in_external_macro: $external,
            future_incompatible: None,
            is_externally_loaded: true,
            $(feature_gate: Some(rustc_span::sym::$gate),)?
            crate_level_only: false,
            $(eval_always: $eval_always,)?
            ..$crate::Lint::default_fields_for_macro()
        };
    );
}

pub type LintVec = Vec<&'static Lint>;

pub trait LintPass {
    fn name(&self) -> &'static str;
    fn get_lints(&self) -> LintVec;
}

/// Implements `LintPass for $ty` with the given list of `Lint` statics.
#[macro_export]
macro_rules! impl_lint_pass {
    ($ty:ty => [$($lint:expr),* $(,)?]) => {
        impl $crate::LintPass for $ty {
            fn name(&self) -> &'static str { stringify!($ty) }
            fn get_lints(&self) -> $crate::LintVec { vec![$($lint),*] }
        }
        impl $ty {
            #[allow(unused)]
            pub fn lint_vec() -> $crate::LintVec { vec![$($lint),*] }
        }
    };
}

/// Declares a type named `$name` which implements `LintPass`.
/// To the right of `=>` a comma separated list of `Lint` statics is given.
#[macro_export]
macro_rules! declare_lint_pass {
    ($(#[$m:meta])* $name:ident => [$($lint:expr),* $(,)?]) => {
        $(#[$m])* #[derive(Copy, Clone)] pub struct $name;
        $crate::impl_lint_pass!($name => [$($lint),*]);
    };
}
