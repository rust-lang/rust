use syntax_pos::{MultiSpan, Symbol, sym};
use syntax_pos::edition::Edition;
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey, StableHasher};
pub use self::Level::*;
use crate::node_id::NodeId;

/// Setting for how to handle a lint.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum Level {
    Allow, Warn, Deny, Forbid,
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

impl Eq for LintId { }

impl std::hash::Hash for LintId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = self.lint as *const Lint;
        ptr.hash(state);
    }
}

impl LintId {
    /// Gets the `LintId` for a `Lint`.
    pub fn of(lint: &'static Lint) -> LintId {
        LintId {
            lint,
        }
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

/// Stores buffered lint info which can later be passed to `librustc`.
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
   pub span: MultiSpan,

   /// The lint message.
   pub msg: String,

   /// The `NodeId` of the AST node that generated the lint.
   pub id: NodeId,

   /// A lint Id that can be passed to `rustc::lint::Lint::from_parser_lint_id`.
   pub lint_id: &'static Lint,
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
