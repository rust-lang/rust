use rustc_ast::attr;
use rustc_ast::attr::AttributeExt;
use rustc_errors::Applicability;
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;
use rustc_middle::ty::{AdtDef, TyCtxt};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use std::str::FromStr;

use crate::source::SpanRangeExt;
use crate::{sym, tokenize_with_text};

/// Deprecation status of attributes known by Clippy.
pub enum DeprecationStatus {
    /// Attribute is deprecated
    Deprecated,
    /// Attribute is deprecated and was replaced by the named attribute
    Replaced(&'static str),
    None,
}

#[rustfmt::skip]
pub const BUILTIN_ATTRIBUTES: &[(Symbol, DeprecationStatus)] = &[
    (sym::author,                DeprecationStatus::None),
    (sym::version,               DeprecationStatus::None),
    (sym::cognitive_complexity,  DeprecationStatus::None),
    (sym::cyclomatic_complexity, DeprecationStatus::Replaced("cognitive_complexity")),
    (sym::dump,                  DeprecationStatus::None),
    (sym::msrv,                  DeprecationStatus::None),
    // The following attributes are for the 3rd party crate authors.
    // See book/src/attribs.md
    (sym::has_significant_drop,  DeprecationStatus::None),
    (sym::format_args,           DeprecationStatus::None),
];

pub struct LimitStack {
    stack: Vec<u64>,
}

impl Drop for LimitStack {
    fn drop(&mut self) {
        assert_eq!(self.stack.len(), 1);
    }
}

impl LimitStack {
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self { stack: vec![limit] }
    }
    pub fn limit(&self) -> u64 {
        *self.stack.last().expect("there should always be a value in the stack")
    }
    pub fn push_attrs(&mut self, sess: &Session, attrs: &[impl AttributeExt], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| stack.push(val));
    }
    pub fn pop_attrs(&mut self, sess: &Session, attrs: &[impl AttributeExt], name: Symbol) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| assert_eq!(stack.pop(), Some(val)));
    }
}

pub fn get_attr<'a, A: AttributeExt + 'a>(
    sess: &'a Session,
    attrs: &'a [A],
    name: Symbol,
) -> impl Iterator<Item = &'a A> {
    attrs.iter().filter(move |attr| {
        let Some(attr_segments) = attr.ident_path() else {
            return false;
        };

        if attr_segments.len() == 2 && attr_segments[0].name == sym::clippy {
            BUILTIN_ATTRIBUTES
                .iter()
                .find_map(|(builtin_name, deprecation_status)| {
                    if attr_segments[1].name == *builtin_name {
                        Some(deprecation_status)
                    } else {
                        None
                    }
                })
                .map_or_else(
                    || {
                        sess.dcx().span_err(attr_segments[1].span, "usage of unknown attribute");
                        false
                    },
                    |deprecation_status| {
                        let mut diag = sess
                            .dcx()
                            .struct_span_err(attr_segments[1].span, "usage of deprecated attribute");
                        match *deprecation_status {
                            DeprecationStatus::Deprecated => {
                                diag.emit();
                                false
                            },
                            DeprecationStatus::Replaced(new_name) => {
                                diag.span_suggestion(
                                    attr_segments[1].span,
                                    "consider using",
                                    new_name,
                                    Applicability::MachineApplicable,
                                );
                                diag.emit();
                                false
                            },
                            DeprecationStatus::None => {
                                diag.cancel();
                                attr_segments[1].name == name
                            },
                        }
                    },
                )
        } else {
            false
        }
    })
}

fn parse_attrs<F: FnMut(u64)>(sess: &Session, attrs: &[impl AttributeExt], name: Symbol, mut f: F) {
    for attr in get_attr(sess, attrs, name) {
        if let Some(value) = attr.value_str() {
            if let Ok(value) = FromStr::from_str(value.as_str()) {
                f(value);
            } else {
                sess.dcx().span_err(attr.span(), "not a number");
            }
        } else {
            sess.dcx().span_err(attr.span(), "bad clippy attribute");
        }
    }
}

pub fn get_unique_attr<'a, A: AttributeExt>(sess: &'a Session, attrs: &'a [A], name: Symbol) -> Option<&'a A> {
    let mut unique_attr: Option<&A> = None;
    for attr in get_attr(sess, attrs, name) {
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

/// Returns true if the attributes contain any of `proc_macro`,
/// `proc_macro_derive` or `proc_macro_attribute`, false otherwise
pub fn is_proc_macro(attrs: &[impl AttributeExt]) -> bool {
    attrs.iter().any(AttributeExt::is_proc_macro_attr)
}

/// Returns true if the attributes contain `#[doc(hidden)]`
pub fn is_doc_hidden(attrs: &[impl AttributeExt]) -> bool {
    attrs
        .iter()
        .filter(|attr| attr.has_name(sym::doc))
        .filter_map(AttributeExt::meta_item_list)
        .any(|l| attr::list_contains_name(&l, sym::hidden))
}

pub fn has_non_exhaustive_attr(tcx: TyCtxt<'_>, adt: AdtDef<'_>) -> bool {
    adt.is_variant_list_non_exhaustive()
        || tcx.has_attr(adt.did(), sym::non_exhaustive)
        || adt.variants().iter().any(|variant_def| {
            variant_def.is_field_list_non_exhaustive() || tcx.has_attr(variant_def.def_id, sym::non_exhaustive)
        })
        || adt
            .all_fields()
            .any(|field_def| tcx.has_attr(field_def.did, sym::non_exhaustive))
}

/// Checks if the given span contains a `#[cfg(..)]` attribute
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
