use rustc_macros::{Decodable, Encodable, PrintAttribute, StableHash};
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

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

#[derive(Clone, Debug, StableHash, Encodable, Decodable, PrintAttribute)]
pub struct LintCheck {
    pub name: ThinVec<Symbol>,
    pub span: Span,
    pub kind: LintCheckKind,
    pub reason: Option<Symbol>,
    /// Needed by `LintExpectationId` to track fulfilled expectations
    pub attr_id: HashIgnoredAttrId,
    pub attr_span: Span,
}
