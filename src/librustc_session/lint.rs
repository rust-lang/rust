pub use self::Level::*;
use rustc_ast::node_id::{NodeId, NodeMap};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder};
use rustc_span::edition::Edition;
use rustc_span::{sym, symbol::Ident, MultiSpan, Span, Symbol};

pub mod builtin;

/// Setting for how to handle a lint.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum Level {
    Allow,
    Warn,
    Deny,
    Forbid,
}

rustc_data_structures::impl_stable_hash_via_hash!(Level);

impl Level {
    /// Converts a level to a lower-case string.
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Allow => "allow",
            Level::Warn => "warn",
            Level::Deny => "deny",
            Level::Forbid => "forbid",
        }
    }

    /// Converts a lower-case string to a level.
    pub fn from_str(x: &str) -> Option<Level> {
        match x {
            "allow" => Some(Level::Allow),
            "warn" => Some(Level::Warn),
            "deny" => Some(Level::Deny),
            "forbid" => Some(Level::Forbid),
            _ => None,
        }
    }

    /// Converts a symbol to a level.
    pub fn from_symbol(x: Symbol) -> Option<Level> {
        match x {
            sym::allow => Some(Level::Allow),
            sym::warn => Some(Level::Warn),
            sym::deny => Some(Level::Deny),
            sym::forbid => Some(Level::Forbid),
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
    pub name: &'static str,

    /// Default level for the lint.
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
}

/// Extra information for a future incompatibility lint.
#[derive(Copy, Clone, Debug)]
pub struct FutureIncompatibleInfo {
    /// e.g., a URL for an issue/PR/RFC or error code
    pub reference: &'static str,
    /// If this is an edition fixing lint, the edition in which
    /// this lint becomes obsolete
    pub edition: Option<Edition>,
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
#[derive(PartialEq)]
pub enum BuiltinLintDiagnostics {
    Normal,
    BareTraitObject(Span, /* is_global */ bool),
    AbsPathWithModule(Span),
    ProcMacroDeriveResolutionFallback(Span),
    MacroExpandedMacroExportsAccessedByAbsolutePaths(Span),
    ElidedLifetimesInPaths(usize, Span, bool, Span, String),
    UnknownCrateTypes(Span, String, String),
    UnusedImports(String, Vec<(Span, String)>),
    RedundantImport(Vec<(Span, bool)>, Ident),
    DeprecatedMacro(Option<Symbol>, Span),
    UnusedDocComment(Span),
}

/// Lints that are buffered up early on in the `Session` before the
/// `LintLevels` is calculated. These are later passed to `librustc`.
#[derive(PartialEq)]
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
    pub span: MultiSpan,

    /// The lint message.
    pub msg: String,

    /// The `NodeId` of the AST node that generated the lint.
    pub node_id: NodeId,

    /// A lint Id that can be passed to `rustc::lint::Lint::from_parser_lint_id`.
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
        if !arr.contains(&early_lint) {
            arr.push(early_lint);
        }
    }

    pub fn add_lint(
        &mut self,
        lint: &'static Lint,
        node_id: NodeId,
        span: MultiSpan,
        msg: &str,
        diagnostic: BuiltinLintDiagnostics,
    ) {
        let lint_id = LintId::of(lint);
        let msg = msg.to_string();
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
        msg: &str,
    ) {
        self.add_lint(lint, id, sp.into(), msg, BuiltinLintDiagnostics::Normal)
    }

    pub fn buffer_lint_with_diagnostic(
        &mut self,
        lint: &'static Lint,
        id: NodeId,
        sp: impl Into<MultiSpan>,
        msg: &str,
        diagnostic: BuiltinLintDiagnostics,
    ) {
        self.add_lint(lint, id, sp.into(), msg, diagnostic)
    }
}

/// Declares a static item of type `&'static Lint`.
#[macro_export]
macro_rules! declare_lint {
    ($vis: vis $NAME: ident, $Level: ident, $desc: expr) => (
        $crate::declare_lint!(
            $vis $NAME, $Level, $desc,
        );
    );
    ($vis: vis $NAME: ident, $Level: ident, $desc: expr,
     $(@future_incompatible = $fi:expr;)? $($v:ident),*) => (
        $vis static $NAME: &$crate::lint::Lint = &$crate::lint::Lint {
            name: stringify!($NAME),
            default_level: $crate::lint::$Level,
            desc: $desc,
            edition_lint_opts: None,
            is_plugin: false,
            $($v: true,)*
            $(future_incompatible: Some($fi),)*
            ..$crate::lint::Lint::default_fields_for_macro()
        };
    );
    ($vis: vis $NAME: ident, $Level: ident, $desc: expr,
     $lint_edition: expr => $edition_level: ident
    ) => (
        $vis static $NAME: &$crate::lint::Lint = &$crate::lint::Lint {
            name: stringify!($NAME),
            default_level: $crate::lint::$Level,
            desc: $desc,
            edition_lint_opts: Some(($lint_edition, $crate::lint::Level::$edition_level)),
            report_in_external_macro: false,
            is_plugin: false,
        };
    );
}

#[macro_export]
macro_rules! declare_tool_lint {
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level: ident, $desc: expr
    ) => (
        $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, false}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        report_in_external_macro: $rep:expr
    ) => (
         $crate::declare_tool_lint!{$(#[$attr])* $vis $tool::$NAME, $Level, $desc, $rep}
    );
    (
        $(#[$attr:meta])* $vis:vis $tool:ident ::$NAME:ident, $Level:ident, $desc:expr,
        $external:expr
    ) => (
        $(#[$attr])*
        $vis static $NAME: &$crate::lint::Lint = &$crate::lint::Lint {
            name: &concat!(stringify!($tool), "::", stringify!($NAME)),
            default_level: $crate::lint::$Level,
            desc: $desc,
            edition_lint_opts: None,
            report_in_external_macro: $external,
            future_incompatible: None,
            is_plugin: true,
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

/// Implements `LintPass for $name` with the given list of `Lint` statics.
#[macro_export]
macro_rules! impl_lint_pass {
    ($name:ident => [$($lint:expr),* $(,)?]) => {
        impl $crate::lint::LintPass for $name {
            fn name(&self) -> &'static str { stringify!($name) }
        }
        impl $name {
            pub fn get_lints() -> $crate::lint::LintArray { $crate::lint_array!($($lint),*) }
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

pub fn add_elided_lifetime_in_path_suggestion(
    sess: &crate::Session,
    db: &mut DiagnosticBuilder<'_>,
    n: usize,
    path_span: Span,
    incl_angl_brckt: bool,
    insertion_span: Span,
    anon_lts: String,
) {
    let (replace_span, suggestion) = if incl_angl_brckt {
        (insertion_span, anon_lts)
    } else {
        // When possible, prefer a suggestion that replaces the whole
        // `Path<T>` expression with `Path<'_, T>`, rather than inserting `'_, `
        // at a point (which makes for an ugly/confusing label)
        if let Ok(snippet) = sess.source_map().span_to_snippet(path_span) {
            // But our spans can get out of whack due to macros; if the place we think
            // we want to insert `'_` isn't even within the path expression's span, we
            // should bail out of making any suggestion rather than panicking on a
            // subtract-with-overflow or string-slice-out-out-bounds (!)
            // FIXME: can we do better?
            if insertion_span.lo().0 < path_span.lo().0 {
                return;
            }
            let insertion_index = (insertion_span.lo().0 - path_span.lo().0) as usize;
            if insertion_index > snippet.len() {
                return;
            }
            let (before, after) = snippet.split_at(insertion_index);
            (path_span, format!("{}{}{}", before, anon_lts, after))
        } else {
            (insertion_span, anon_lts)
        }
    };
    db.span_suggestion(
        replace_span,
        &format!("indicate the anonymous lifetime{}", pluralize!(n)),
        suggestion,
        Applicability::MachineApplicable,
    );
}
