use rustc_macros::{Decodable, Encodable, PrintAttribute, StableHash};
use rustc_span::{Span, Symbol, sym};

use crate::{HashIgnoredAttrId, PrintAttribute};
#[derive(Clone, Copy, Debug, StableHash, Encodable, Decodable, PrintAttribute)]
pub enum LintCheckKind {
    Allow,
    Warn,
    Deny,
    Forbid,
    Expect,
}

impl LintCheckKind {
    pub fn sym(self) -> Symbol {
        match self {
            LintCheckKind::Allow => sym::allow,
            LintCheckKind::Warn => sym::warn,
            LintCheckKind::Deny => sym::deny,
            LintCheckKind::Forbid => sym::forbid,
            LintCheckKind::Expect => sym::expect,
        }
    }
}

/// A lint check attribute.
///
/// For example `#[deny(clippy::blah, reason = "reason")]` is lowered into this.
///
/// These are smooshed and flattened together;
/// ```rust
/// #[allow(dead_code)]
/// #[deny(unused, unsafe_code)]
/// # const _: () = ();
/// ```
/// is lowered into
/// ```text
/// #[attr = LintCheck([
///    LintCheck { lint_name: "dead_code", kind: Allow },
///    LintCheck { lint_name: "unused", kind: Deny },
///    LintCheck { lint_name: "unsafe_code", kind: Deny },
/// ])]
/// ```
#[derive(Clone, Debug, StableHash, Encodable, Decodable, PrintAttribute)]
pub struct LintCheck {
    /// The lint's tool name, if present.
    pub tool: Option<Symbol>,
    /// The full normalized lint name of an existing lint, including the tool name.
    /// This may not be what we user wrote down, in case of renamed lints the new lint name is here.
    /// The above implies that `LintStore::find_lints` is guaranteed to find this lint using this symbol.
    ///
    /// For example, `single_use_lifetime` is changed to `single_use_lifetimes`.
    pub lint_name_normalized: Symbol,
    /// The span of the lint name.
    ///
    /// For example `#[deny(foo, bar, reason = "reason")]`
    /// produces multiple `LintCheck`s, one with a span pointing to `foo`
    /// and another pointing to `bar`.
    pub lint_span: Span,
    pub kind: LintCheckKind,
    pub reason: Option<Symbol>,
    /// Needed by `LintExpectationId` to track fulfilled expectations
    pub attr_id: HashIgnoredAttrId,
    /// The span of the attribute this lintcheck came from.
    ///
    /// Like mentioned above, multiple lint attributes and multiple lints
    /// inside one attribute are all smooshed and flattened together, so
    /// multiple `LintCheck`s can have the same `attr_span`.
    pub attr_span: Span,
}
