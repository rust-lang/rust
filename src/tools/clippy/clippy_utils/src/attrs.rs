//! Utility functions for attributes, including Clippy's built-in ones

use crate::source::SpanExt as _;
use crate::{sym, tokenize_with_text};
use rustc_ast::attr::AttributeExt;
use rustc_attr_ir::{AttrArgs, find_attr};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;
use rustc_middle::ty::{AdtDef, TyCtxt};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use std::str::FromStr as _;

/// Validates a single clippy attribute and emits errors for unknown or deprecated ones.
pub fn check_clippy_attr(sess: &Session, attr: &rustc_ast::ast::Attribute) {
    if let [clippy, segment2] = &*attr.path()
        && *clippy == sym::clippy
    {
        let path_span = attr
            .path_span()
            .expect("Clippy attributes are tool attributes, so they have a span");

        match *segment2 {
            sym::cyclomatic_complexity => {
                sess.dcx()
                    .struct_span_err(path_span, "usage of deprecated attribute")
                    .with_span_suggestion(
                        path_span,
                        "consider using",
                        "clippy::cognitive_complexity",
                        Applicability::MachineApplicable,
                    )
                    .emit();
            },
            sym::author
            | sym::version
            | sym::cognitive_complexity
            | sym::dump
            | sym::msrv
            | sym::has_significant_drop
            | sym::format_args => {},
            _ => {
                sess.dcx().span_err(path_span, "usage of unknown attribute");
            },
        }
    }
}

/// Retrieves the span of a Clippy attribute, such as `#[clippy::format_args]`.
///
/// Panics if used on an attribute that isn't a tool attribute, so always check
/// for that first with e.g. `path` or `path_matches`.
pub fn span_of_clippy_attr(attr: &(impl AttributeExt + 'static)) -> Span {
    let a: &dyn core::any::Any = attr;
    if let Some(a) = a.downcast_ref::<rustc_ast::ast::Attribute>() {
        a.span
    } else if let Some(rustc_attr_ir::Attribute::Tool(u)) = a.downcast_ref::<rustc_attr_ir::Attribute>() {
        u.span
    } else {
        panic!("attribute {attr:?} is not a clippy attribute")
    }
}

/// Returns the first occurrence of `#[clippy::<name>]`.
///
/// Additionally, raises an error for duplicates of this attribute.
pub fn get_clippy_attr(tcx: TyCtxt<'_>, did: impl Into<DefId>, name: Symbol) -> Option<&rustc_attr_ir::AttrItem> {
    let attr_name = [sym::clippy, name];
    let mut attrs = tcx.tool_attrs(did, &attr_name);
    let first = attrs.next()?;

    //FIXME(mejrs): this should really be checked during validation
    for attr in attrs {
        tcx.dcx()
            .struct_span_err(attr.span, format!("`{name}` is defined multiple times"))
            .with_span_note(first.span, "first definition found here")
            .emit();
    }

    Some(first)
}

/// Checks whether `attrs` contain any of `proc_macro`, `proc_macro_derive` or
/// `proc_macro_attribute`
pub fn is_proc_macro(attrs: &[rustc_attr_ir::Attribute]) -> bool {
    find_attr!(
        attrs,
        ProcMacro
        | ProcMacroAttribute
        | ProcMacroDerive { .. }
    )
}

/// Checks whether `attrs` contain `#[doc(hidden)]`
pub fn is_doc_hidden(attrs: &[rustc_attr_ir::Attribute]) -> bool {
    find_attr!(
        attrs,
        Doc(rustc_attr_ir::DocAttribute { hidden: Some(_), .. })
    )
}

/// Checks whether the given ADT, or any of its fields/variants, are marked as `#[non_exhaustive]`
pub fn has_non_exhaustive_attr(tcx: TyCtxt<'_>, adt: AdtDef<'_>) -> bool {
    adt.is_variant_list_non_exhaustive()
        || find_attr!(tcx, adt.did(), NonExhaustive(..))
        || adt.variants().iter().any(|variant_def| {
            variant_def.is_field_list_non_exhaustive() || find_attr!(tcx, variant_def.def_id, NonExhaustive(..))
        })
        || adt
            .all_fields()
            .any(|field_def| find_attr!(tcx, field_def.did, NonExhaustive(..)))
}

fn span_contains_attr_idents(cx: &LateContext<'_>, s: Span, idents: &[&str]) -> bool {
    s.check_text(cx, |src| {
        // PERF: Attributes need a literal `#`, so skip the lexer when the source has none.
        if !src.contains('#') {
            return false;
        }

        let mut iter = tokenize_with_text(src);

        // Search for the token sequence [`#`, `[`, <ident>]
        while iter.any(|(t, ..)| matches!(t, TokenKind::Pound)) {
            let mut iter = iter.by_ref().skip_while(|(t, ..)| {
                matches!(
                    t,
                    TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. }
                )
            });
            if matches!(iter.next(), Some((TokenKind::OpenBracket, ..)))
                && matches!(iter.next(), Some((TokenKind::Ident, name, _)) if idents.contains(&name))
            {
                return true;
            }
        }
        false
    })
}

/// Checks whether the given span contains a `#[cfg(..)]` attribute
pub fn span_contains_cfg(cx: &LateContext<'_>, s: Span) -> bool {
    span_contains_attr_idents(cx, s, &["cfg"])
}

/// Checks whether the given span contains a `#[cfg(..)]` or `#[cfg_attr(..)]` attribute
pub fn span_contains_cfg_or_cfg_attr(cx: &LateContext<'_>, s: Span) -> bool {
    span_contains_attr_idents(cx, s, &["cfg", "cfg_attr"])
}

/// Currently used to keep track of the current value of `#[clippy::cognitive_complexity(N)]`
pub struct LimitStack {
    default: u64,
    stack: Vec<u64>,
}

impl Drop for LimitStack {
    fn drop(&mut self) {
        debug_assert_eq!(self.stack, Vec::<u64>::new()); // avoid `.is_empty()`, for a nicer error message
    }
}

#[expect(missing_docs, reason = "they're all trivial...")]
impl LimitStack {
    #[must_use]
    /// Initialize the stack starting with a default value, which usually comes from configuration
    pub fn new(limit: u64) -> Self {
        Self {
            default: limit,
            stack: vec![],
        }
    }
    pub fn limit(&self) -> u64 {
        self.stack.last().copied().unwrap_or(self.default)
    }
    pub fn push_attrs(&mut self, sess: &Session, attrs: &[rustc_attr_ir::Attribute], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| stack.push(val));
    }
    pub fn pop_attrs(&mut self, sess: &Session, attrs: &[rustc_attr_ir::Attribute], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| {
            let popped = stack.pop();
            debug_assert_eq!(popped, Some(val));
        });
    }
}

fn parse_attrs<F: FnMut(u64)>(sess: &Session, attrs: &[rustc_attr_ir::Attribute], name: Symbol, mut f: F) {
    let attrs = attrs.iter().filter_map(move |attr| {
        if let rustc_attr_ir::Attribute::Tool(t) = attr
            && *t.path.segments == [sym::clippy, name]
        {
            Some(&**t)
        } else {
            None
        }
    });
    for attr in attrs {
        let AttrArgs::Eq { expr, .. } = attr.args else {
            sess.dcx().span_err(attr.span, "bad clippy attribute");
            continue;
        };
        let Some(value) = expr.value_as_str() else {
            sess.dcx().span_err(attr.span, "not a number");
            continue;
        };
        let Ok(value) = u64::from_str(value.as_str()) else {
            sess.dcx().span_err(attr.span, "not a number");
            continue;
        };
        f(value);
    }
}
