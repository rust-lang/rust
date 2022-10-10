#![feature(min_specialization)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;

pub use self::Level::*;
use rustc_ast::node_id::{NodeId, NodeMap};
use rustc_ast::{AttrId, Attribute};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_error_messages::{DiagnosticMessage, MultiSpan};
use rustc_hir::HashStableContext;
use rustc_hir::HirId;
use rustc_span::edition::Edition;
use rustc_span::{sym, symbol::Ident, Span, Symbol};
use rustc_target::spec::abi::Abi;

use serde::{Deserialize, Serialize};

pub mod builtin;

#[macro_export]
macro_rules! pluralize {
    ($x:expr) => {
        if $x != 1 { "s" } else { "" }
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
/// Expected `Diagnostic`s get the lint level `Expect` which stores the `LintExpectationId`
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
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash, Encodable, Decodable)]
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
        let (LintExpectationId::Unstable { ref mut lint_index, .. }
        | LintExpectationId::Stable { ref mut lint_index, .. }) = self;

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
    type KeyType = (HirId, u16, u16);

    #[inline]
    fn to_stable_hash_key(&self, _: &HCX) -> Self::KeyType {
        match self {
            LintExpectationId::Stable { hir_id, attr_index, lint_index: Some(lint_index) } => {
                (*hir_id, *attr_index, *lint_index)
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
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash, HashStable_Generic)]
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
    /// The [`LintExpectationId`] is used to later link a lint emission to the actual
    /// expectation. It can be ignored in most cases.
    Expect(LintExpectationId),
    /// The `warn` level will produce a warning if the lint was violated, however the
    /// compiler will continue with its execution.
    Warn,
    /// This lint level is a special case of [`Warn`], that can't be overridden. This is used
    /// to ensure that a lint can't be suppressed. This lint level can currently only be set
    /// via the console and is therefore session specific.
    ///
    /// The [`LintExpectationId`] is intended to fulfill expectations marked via the
    /// `#[expect]` attribute, that will still be suppressed due to the level.
    ForceWarn(Option<LintExpectationId>),
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
            Level::Expect(_) => "expect",
            Level::Warn => "warn",
            Level::ForceWarn(_) => "force-warn",
            Level::Deny => "deny",
            Level::Forbid => "forbid",
        }
    }

    /// Converts a lower-case string to a level. This will never construct the expect
    /// level as that would require a [`LintExpectationId`]
    pub fn from_str(x: &str) -> Option<Level> {
        match x {
            "allow" => Some(Level::Allow),
            "warn" => Some(Level::Warn),
            "deny" => Some(Level::Deny),
            "forbid" => Some(Level::Forbid),
            "expect" | _ => None,
        }
    }

    /// Converts a symbol to a level.
    pub fn from_attr(attr: &Attribute) -> Option<Level> {
        match attr.name_or_empty() {
            sym::allow => Some(Level::Allow),
            sym::expect => Some(Level::Expect(LintExpectationId::Unstable {
                attr_id: attr.id,
                lint_index: None,
            })),
            sym::warn => Some(Level::Warn),
            sym::deny => Some(Level::Deny),
            sym::forbid => Some(Level::Forbid),
            _ => None,
        }
    }

    pub fn is_error(self) -> bool {
        match self {
            Level::Allow | Level::Expect(_) | Level::Warn | Level::ForceWarn(_) => false,
            Level::Deny | Level::Forbid => true,
        }
    }

    pub fn get_expectation_id(&self) -> Option<LintExpectationId> {
        match self {
            Level::Expect(id) | Level::ForceWarn(Some(id)) => Some(*id),
            _ => None,
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

    pub is_plugin: bool,

    /// `Some` if this lint is feature gated, otherwise `None`.
    pub feature_gate: Option<Symbol>,

    pub crate_level_only: bool,
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
}

/// The reason for future incompatibility
#[derive(Copy, Clone, Debug)]
pub enum FutureIncompatibilityReason {
    /// This will be an error in a future release
    /// for all editions
    FutureReleaseError,
    /// This will be an error in a future release, and
    /// Cargo should create a report even for dependencies
    FutureReleaseErrorReportNow,
    /// Code that changes meaning in some way in a
    /// future release.
    FutureReleaseSemanticsChange,
    /// Previously accepted code that will become an
    /// error in the provided edition
    EditionError(Edition),
    /// Code that changes meaning in some way in
    /// the provided edition
    EditionSemanticsChange(Edition),
    /// A custom reason.
    Custom(&'static str),
}

impl FutureIncompatibilityReason {
    pub fn edition(self) -> Option<Edition> {
        match self {
            Self::EditionError(e) => Some(e),
            Self::EditionSemanticsChange(e) => Some(e),
            _ => None,
        }
    }
}

impl FutureIncompatibleInfo {
    pub const fn default_fields_for_macro() -> Self {
        FutureIncompatibleInfo {
            reference: "",
            reason: FutureIncompatibilityReason::FutureReleaseError,
            explain_reason: true,
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
            is_plugin: false,
            report_in_external_macro: false,
            future_incompatible: None,
            feature_gate: None,
            crate_level_only: false,
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

// This could be a closure, but then implementing derive trait
// becomes hacky (and it gets allocated).
#[derive(Debug)]
pub enum BuiltinLintDiagnostics {
    Normal,
    AbsPathWithModule(Span),
    ProcMacroDeriveResolutionFallback(Span),
    MacroExpandedMacroExportsAccessedByAbsolutePaths(Span),
    ElidedLifetimesInPaths(usize, Span, bool, Span),
    UnknownCrateTypes(Span, String, String),
    UnusedImports(String, Vec<(Span, String)>, Option<Span>),
    RedundantImport(Vec<(Span, bool)>, Ident),
    DeprecatedMacro(Option<Symbol>, Span),
    MissingAbi(Span, Abi),
    UnusedDocComment(Span),
    UnusedBuiltinAttribute {
        attr_name: Symbol,
        macro_name: String,
        invoc_span: Span,
    },
    PatternsInFnsWithoutBody(Span, Ident),
    LegacyDeriveHelpers(Span),
    ProcMacroBackCompat(String),
    OrPatternsBackCompat(Span, String),
    ReservedPrefix(Span),
    TrailingMacro(bool, Ident),
    BreakWithLabelAndLoop(Span),
    NamedAsmLabel(String),
    UnicodeTextFlow(Span, String),
    UnexpectedCfg((Symbol, Span), Option<(Symbol, Span)>),
    DeprecatedWhereclauseLocation(Span, String),
    SingleUseLifetime {
        /// Span of the parameter which declares this lifetime.
        param_span: Span,
        /// Span of the code that should be removed when eliding this lifetime.
        /// This span should include leading or trailing comma.
        deletion_span: Span,
        /// Span of the single use, or None if the lifetime is never used.
        /// If true, the lifetime will be fully elided.
        use_span: Option<(Span, bool)>,
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
}

/// Lints that are buffered up early on in the `Session` before the
/// `LintLevels` is calculated.
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
    pub span: MultiSpan,

    /// The lint message.
    pub msg: DiagnosticMessage,

    /// The `NodeId` of the AST node that generated the lint.
    pub node_id: NodeId,

    /// A lint Id that can be passed to
    /// `rustc_lint::early::EarlyContextAndPass::check_id`.
    pub lint_id: LintId,

    /// Customization of the `DiagnosticBuilder<'_>` for the lint.
    pub diagnostic: BuiltinLintDiagnostics,
}

#[derive(Default)]
pub struct LintBuffer {
    pub map: NodeMap<Vec<BufferedEarlyLint>>,
}

impl LintBuffer {
    pub fn add_early_lint(&mut self, early_lint: BufferedEarlyLint) {
        let arr = self.map.entry(early_lint.node_id).or_default();
        arr.push(early_lint);
    }

    pub fn add_lint(
        &mut self,
        lint: &'static Lint,
        node_id: NodeId,
        span: MultiSpan,
        msg: impl Into<DiagnosticMessage>,
        diagnostic: BuiltinLintDiagnostics,
    ) {
        let lint_id = LintId::of(lint);
        let msg = msg.into();
        self.add_early_lint(BufferedEarlyLint { lint_id, node_id, span, msg, diagnostic });
    }

    pub fn take(&mut self, id: NodeId) -> Vec<BufferedEarlyLint> {
        self.map.remove(&id).unwrap_or_default()
    }

    pub fn buffer_lint(
        &mut self,
        lint: &'static Lint,
        id: NodeId,
        sp: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) {
        self.add_lint(lint, id, sp.into(), msg, BuiltinLintDiagnostics::Normal)
    }

    pub fn buffer_lint_with_diagnostic(
        &mut self,
        lint: &'static Lint,
        id: NodeId,
        sp: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        diagnostic: BuiltinLintDiagnostics,
    ) {
        self.add_lint(lint, id, sp.into(), msg, diagnostic)
    }
}

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
     $(@feature_gate = $gate:expr;)?
     $(@future_incompatible = FutureIncompatibleInfo { $($field:ident : $val:expr),* $(,)*  }; )?
     $($v:ident),*) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::Lint = &$crate::Lint {
            name: stringify!($NAME),
            default_level: $crate::$Level,
            desc: $desc,
            edition_lint_opts: None,
            is_plugin: false,
            $($v: true,)*
            $(feature_gate: Some($gate),)*
            $(future_incompatible: Some($crate::FutureIncompatibleInfo {
                $($field: $val,)*
                ..$crate::FutureIncompatibleInfo::default_fields_for_macro()
            }),)*
            ..$crate::Lint::default_fields_for_macro()
        };
    );
    ($(#[$attr:meta])* $vis: vis $NAME: ident, $Level: ident, $desc: expr,
     $lint_edition: expr => $edition_level: ident
    ) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::Lint = &$crate::Lint {
            name: stringify!($NAME),
            default_level: $crate::$Level,
            desc: $desc,
            edition_lint_opts: Some(($lint_edition, $crate::Level::$edition_level)),
            report_in_external_macro: false,
            is_plugin: false,
        };
    );
}

#[macro_export]
macro_rules! declare_tool_lint {
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level: ident, $desc: expr
        $(, @feature_gate = $gate:expr;)?
    ) => (
        $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, false $(, @feature_gate = $gate;)?}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        report_in_external_macro: $rep:expr
        $(, @feature_gate = $gate:expr;)?
    ) => (
         $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, $rep $(, @feature_gate = $gate;)?}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        $external:expr
        $(, @feature_gate = $gate:expr;)?
    ) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::Lint = &$crate::Lint {
            name: &concat!(stringify!($tool), "::", stringify!($NAME)),
            default_level: $crate::$Level,
            desc: $desc,
            edition_lint_opts: None,
            report_in_external_macro: $external,
            future_incompatible: None,
            is_plugin: true,
            $(feature_gate: Some($gate),)?
            crate_level_only: false,
            ..$crate::Lint::default_fields_for_macro()
        };
    );
}

/// Declares a static `LintArray` and return it as an expression.
#[macro_export]
macro_rules! lint_array {
    ($( $lint:expr ),* ,) => { lint_array!( $($lint),* ) };
    ($( $lint:expr ),*) => {{
        vec![$($lint),*]
    }}
}

pub type LintArray = Vec<&'static Lint>;

pub trait LintPass {
    fn name(&self) -> &'static str;
}

/// Implements `LintPass for $ty` with the given list of `Lint` statics.
#[macro_export]
macro_rules! impl_lint_pass {
    ($ty:ty => [$($lint:expr),* $(,)?]) => {
        impl $crate::LintPass for $ty {
            fn name(&self) -> &'static str { stringify!($ty) }
        }
        impl $ty {
            pub fn get_lints() -> $crate::LintArray { $crate::lint_array!($($lint),*) }
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
