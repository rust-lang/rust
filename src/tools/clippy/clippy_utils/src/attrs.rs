//! Utility functions for attributes, including Clippy's built-in ones

use crate::source::SpanRangeExt;
use crate::{sym, tokenize_with_text};
use rustc_ast::attr::AttributeExt;
use rustc_errors::Applicability;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::find_attr;
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;
use rustc_middle::ty::{AdtDef, TyCtxt};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use std::str::FromStr;

/// Given `attrs`, extract all the instances of a built-in Clippy attribute called `name`
pub fn get_builtin_attr<'a, A: AttributeExt + 'a>(
    sess: &'a Session,
    attrs: &'a [A],
    name: Symbol,
) -> impl Iterator<Item = &'a A> {
    attrs.iter().filter(move |attr| {
        if let Some([clippy, segment2]) = attr.ident_path().as_deref()
            && clippy.name == sym::clippy
        {
            let new_name = match segment2.name {
                sym::cyclomatic_complexity => Some("cognitive_complexity"),
                sym::author
                | sym::version
                | sym::cognitive_complexity
                | sym::dump
                | sym::msrv
                // The following attributes are for the 3rd party crate authors.
                // See book/src/attribs.md
                | sym::has_significant_drop
                | sym::format_args => None,
                _ => {
                    sess.dcx().span_err(segment2.span, "usage of unknown attribute");
                    return false;
                },
            };

            match new_name {
                Some(new_name) => {
                    sess.dcx()
                        .struct_span_err(segment2.span, "usage of deprecated attribute")
                        .with_span_suggestion(
                            segment2.span,
                            "consider using",
                            new_name,
                            Applicability::MachineApplicable,
                        )
                        .emit();
                    false
                },
                None => segment2.name == name,
            }
        } else {
            false
        }
    })
}

/// If `attrs` contain exactly one instance of a built-in Clippy attribute called `name`,
/// returns that attribute, and `None` otherwise
pub fn get_unique_builtin_attr<'a, A: AttributeExt>(sess: &'a Session, attrs: &'a [A], name: Symbol) -> Option<&'a A> {
    let mut unique_attr: Option<&A> = None;
    for attr in get_builtin_attr(sess, attrs, name) {
        if let Some(duplicate) = unique_attr {
            sess.dcx()
                .struct_span_err(attr.span(), format!("`{name}` is defined multiple times"))
                .with_span_note(duplicate.span(), "first definition found here")
                .emit();
        } else {
            unique_attr = Some(attr);
        }
    }
    unique_attr
}

/// Checks whether `attrs` contain any of `proc_macro`, `proc_macro_derive` or
/// `proc_macro_attribute`
pub fn is_proc_macro(attrs: &[impl AttributeExt]) -> bool {
    attrs.iter().any(AttributeExt::is_proc_macro_attr)
}

/// Checks whether `attrs` contain `#[doc(hidden)]`
pub fn is_doc_hidden(attrs: &[impl AttributeExt]) -> bool {
    attrs.iter().any(|attr| attr.is_doc_hidden())
}

/// Checks whether the given ADT, or any of its fields/variants, are marked as `#[non_exhaustive]`
pub fn has_non_exhaustive_attr(tcx: TyCtxt<'_>, adt: AdtDef<'_>) -> bool {
    adt.is_variant_list_non_exhaustive()
        || find_attr!(tcx.get_all_attrs(adt.did()), AttributeKind::NonExhaustive(..))
        || adt.variants().iter().any(|variant_def| {
            variant_def.is_field_list_non_exhaustive()
                || find_attr!(tcx.get_all_attrs(variant_def.def_id), AttributeKind::NonExhaustive(..))
        })
        || adt
            .all_fields()
            .any(|field_def| find_attr!(tcx.get_all_attrs(field_def.did), AttributeKind::NonExhaustive(..)))
}

/// Checks whether the given span contains a `#[cfg(..)]` attribute
pub fn span_contains_cfg(cx: &LateContext<'_>, s: Span) -> bool {
    s.check_source_text(cx, |src| {
        let mut iter = tokenize_with_text(src);

        // Search for the token sequence [`#`, `[`, `cfg`]
        while iter.any(|(t, ..)| matches!(t, TokenKind::Pound)) {
            let mut iter = iter.by_ref().skip_while(|(t, ..)| {
                matches!(
                    t,
                    TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. }
                )
            });
            if matches!(iter.next(), Some((TokenKind::OpenBracket, ..)))
                && matches!(iter.next(), Some((TokenKind::Ident, "cfg", _)))
            {
                return true;
            }
        }
        false
    })
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
    pub fn push_attrs(&mut self, sess: &Session, attrs: &[impl AttributeExt], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| stack.push(val));
    }
    pub fn pop_attrs(&mut self, sess: &Session, attrs: &[impl AttributeExt], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| debug_assert_eq!(stack.pop(), Some(val)));
    }
}

fn parse_attrs<F: FnMut(u64)>(sess: &Session, attrs: &[impl AttributeExt], name: Symbol, mut f: F) {
    for attr in get_builtin_attr(sess, attrs, name) {
        let Some(value) = attr.value_str() else {
            sess.dcx().span_err(attr.span(), "bad clippy attribute");
            continue;
        };
        let Ok(value) = u64::from_str(value.as_str()) else {
            sess.dcx().span_err(attr.span(), "not a number");
            continue;
        };
        f(value);
    }
}
