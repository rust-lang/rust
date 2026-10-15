use std::fmt;

use rustc_ast as ast;
use rustc_ast::join_path_syms;
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
    pub lint_name: LintName,
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
    /// The span of the entire lint name.
    ///
    /// #[deny(foo::bar::baz)]
    ///        ^^^^^^^^^^^^^
    pub lint_span: Span,
    pub tool_span: Option<Span>,
}

#[derive(Copy, Clone, Debug, StableHash, Encodable, Decodable, PrintAttribute)]
pub struct LintName {
    /// The lint's tool name, if present.
    pub tool: Option<Symbol>,
    /// The lint's name, minus the tool name if any.
    ///
    /// For example:
    /// - `clippy::blah` -> `blah`
    /// - `unstable_imports` -> `unstable_imports`
    /// - `custom_tool::blah::bleh` -> `blah::bleh`
    pub name: Symbol,
}

impl LintName {
    pub fn new(path: &ast::Path) -> LintName {
        let (tool, name) = match &*path.segments {
            [] => unreachable!(),
            [lint_name] => (None, lint_name.ident.name),
            [tool, lint_name] => (Some(tool.ident.name), lint_name.ident.name),
            // Not reachable for any builtin lints,
            // but reachable using `register_tool`.
            [tool, multi_segment_lint_name @ ..] => (
                Some(tool.ident.name),
                Symbol::intern(&join_path_syms(
                    multi_segment_lint_name.iter().map(|seg| seg.ident.name),
                )),
            ),
        };
        LintName { tool, name }
    }

    pub fn is(&self, lint: Symbol) -> bool {
        lint == self.name && self.tool.is_none()
    }
}

impl fmt::Display for LintName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tool) = self.tool {
            write!(f, "{tool}::")?
        }
        write!(f, "{}", self.name)?;
        Ok(())
    }
}
