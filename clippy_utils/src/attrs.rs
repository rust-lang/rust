use rustc_ast::{ast, attr};
use rustc_errors::Applicability;
use rustc_session::Session;
use rustc_span::sym;
use std::str::FromStr;

/// Deprecation status of attributes known by Clippy.
#[allow(dead_code)]
pub enum DeprecationStatus {
    /// Attribute is deprecated
    Deprecated,
    /// Attribute is deprecated and was replaced by the named attribute
    Replaced(&'static str),
    None,
}

#[rustfmt::skip]
pub const BUILTIN_ATTRIBUTES: &[(&str, DeprecationStatus)] = &[
    ("author",                DeprecationStatus::None),
    ("version",               DeprecationStatus::None),
    ("cognitive_complexity",  DeprecationStatus::None),
    ("cyclomatic_complexity", DeprecationStatus::Replaced("cognitive_complexity")),
    ("dump",                  DeprecationStatus::None),
    ("msrv",                  DeprecationStatus::None),
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
    pub fn push_attrs(&mut self, sess: &Session, attrs: &[ast::Attribute], name: &'static str) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| stack.push(val));
    }
    pub fn pop_attrs(&mut self, sess: &Session, attrs: &[ast::Attribute], name: &'static str) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| assert_eq!(stack.pop(), Some(val)));
    }
}

pub fn get_attr<'a>(
    sess: &'a Session,
    attrs: &'a [ast::Attribute],
    name: &'static str,
) -> impl Iterator<Item = &'a ast::Attribute> {
    attrs.iter().filter(move |attr| {
        let attr = if let ast::AttrKind::Normal(ref attr, _) = attr.kind {
            attr
        } else {
            return false;
        };
        let attr_segments = &attr.path.segments;
        if attr_segments.len() == 2 && attr_segments[0].ident.name == sym::clippy {
            BUILTIN_ATTRIBUTES
                .iter()
                .find_map(|&(builtin_name, ref deprecation_status)| {
                    if attr_segments[1].ident.name.as_str() == builtin_name {
                        Some(deprecation_status)
                    } else {
                        None
                    }
                })
                .map_or_else(
                    || {
                        sess.span_err(attr_segments[1].ident.span, "usage of unknown attribute");
                        false
                    },
                    |deprecation_status| {
                        let mut diag =
                            sess.struct_span_err(attr_segments[1].ident.span, "usage of deprecated attribute");
                        match *deprecation_status {
                            DeprecationStatus::Deprecated => {
                                diag.emit();
                                false
                            },
                            DeprecationStatus::Replaced(new_name) => {
                                diag.span_suggestion(
                                    attr_segments[1].ident.span,
                                    "consider using",
                                    new_name.to_string(),
                                    Applicability::MachineApplicable,
                                );
                                diag.emit();
                                false
                            },
                            DeprecationStatus::None => {
                                diag.cancel();
                                attr_segments[1].ident.name.as_str() == name
                            },
                        }
                    },
                )
        } else {
            false
        }
    })
}

fn parse_attrs<F: FnMut(u64)>(sess: &Session, attrs: &[ast::Attribute], name: &'static str, mut f: F) {
    for attr in get_attr(sess, attrs, name) {
        if let Some(ref value) = attr.value_str() {
            if let Ok(value) = FromStr::from_str(value.as_str()) {
                f(value);
            } else {
                sess.span_err(attr.span, "not a number");
            }
        } else {
            sess.span_err(attr.span, "bad clippy attribute");
        }
    }
}

pub fn get_unique_inner_attr(sess: &Session, attrs: &[ast::Attribute], name: &'static str) -> Option<ast::Attribute> {
    let mut unique_attr = None;
    for attr in get_attr(sess, attrs, name) {
        match attr.style {
            ast::AttrStyle::Inner if unique_attr.is_none() => unique_attr = Some(attr.clone()),
            ast::AttrStyle::Inner => {
                sess.struct_span_err(attr.span, &format!("`{}` is defined multiple times", name))
                    .span_note(unique_attr.as_ref().unwrap().span, "first definition found here")
                    .emit();
            },
            ast::AttrStyle::Outer => {
                sess.span_err(attr.span, &format!("`{}` cannot be an outer attribute", name));
            },
        }
    }
    unique_attr
}

/// Return true if the attributes contain any of `proc_macro`,
/// `proc_macro_derive` or `proc_macro_attribute`, false otherwise
pub fn is_proc_macro(sess: &Session, attrs: &[ast::Attribute]) -> bool {
    attrs.iter().any(|attr| sess.is_proc_macro_attr(attr))
}

/// Return true if the attributes contain `#[doc(hidden)]`
pub fn is_doc_hidden(attrs: &[ast::Attribute]) -> bool {
    attrs
        .iter()
        .filter(|attr| attr.has_name(sym::doc))
        .filter_map(ast::Attribute::meta_item_list)
        .any(|l| attr::list_contains_name(&l, sym::hidden))
}

/// Return true if the attributes contain `#[unstable]`
pub fn is_unstable(attrs: &[ast::Attribute]) -> bool {
    attrs.iter().any(|attr| attr.has_name(sym::unstable))
}
