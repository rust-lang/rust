use std::num::IntErrorKind;

use rustc_ast::LitKind;
use rustc_ast::attr::AttributeExt;
use rustc_feature::is_builtin_attr_name;
use rustc_hir::RustcVersion;
use rustc_hir::limit::Limit;
use rustc_span::{Symbol, sym};

use crate::context::{AcceptContext, Stage};
use crate::parser::{ArgParser, NameValueParser};
use crate::session_diagnostics::LimitInvalid;

/// Parse a rustc version number written inside string literal in an attribute,
/// like appears in `since = "1.0.0"`. Suffixes like "-dev" and "-nightly" are
/// not accepted in this position, unlike when parsing CFG_RELEASE.
pub fn parse_version(s: Symbol) -> Option<RustcVersion> {
    let mut components = s.as_str().split('-');
    let d = components.next()?;
    if components.next().is_some() {
        return None;
    }
    let mut digits = d.splitn(3, '.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    let patch = digits.next().unwrap_or("0").parse().ok()?;
    Some(RustcVersion { major, minor, patch })
}

pub fn is_builtin_attr(attr: &impl AttributeExt) -> bool {
    attr.is_doc_comment() || attr.ident().is_some_and(|ident| is_builtin_attr_name(ident.name))
}

pub fn is_doc_alias_attrs_contain_symbol<'tcx, T: AttributeExt + 'tcx>(
    attrs: impl Iterator<Item = &'tcx T>,
    symbol: Symbol,
) -> bool {
    let doc_attrs = attrs.filter(|attr| attr.has_name(sym::doc));
    for attr in doc_attrs {
        let Some(values) = attr.meta_item_list() else {
            continue;
        };
        let alias_values = values.iter().filter(|v| v.has_name(sym::alias));
        for v in alias_values {
            if let Some(nested) = v.meta_item_list() {
                // #[doc(alias("foo", "bar"))]
                let mut iter = nested.iter().filter_map(|item| item.lit()).map(|item| item.symbol);
                if iter.any(|s| s == symbol) {
                    return true;
                }
            } else if let Some(meta) = v.meta_item()
                && let Some(lit) = meta.name_value_literal()
            {
                // #[doc(alias = "foo")]
                if lit.symbol == symbol {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse a single integer.
///
/// Used by attributes that take a single integer as argument, such as
/// `#[link_ordinal]` and `#[rustc_layout_scalar_valid_range_start]`.
/// `cx` is the context given to the attribute.
/// `args` is the parser for the attribute arguments.
pub(crate) fn parse_single_integer<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
) -> Option<u128> {
    let Some(list) = args.list() else {
        cx.expected_list(cx.attr_span);
        return None;
    };
    let Some(single) = list.single() else {
        cx.expected_single_argument(list.span);
        return None;
    };
    let Some(lit) = single.lit() else {
        cx.expected_integer_literal(single.span());
        return None;
    };
    let LitKind::Int(num, _ty) = lit.kind else {
        cx.expected_integer_literal(single.span());
        return None;
    };
    Some(num.0)
}

impl<S: Stage> AcceptContext<'_, '_, S> {
    pub(crate) fn parse_limit_int(&self, nv: &NameValueParser) -> Option<Limit> {
        let Some(limit) = nv.value_as_str() else {
            self.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        let error_str = match limit.as_str().parse() {
            Ok(i) => return Some(Limit::new(i)),
            Err(e) => match e.kind() {
                IntErrorKind::PosOverflow => "`limit` is too large",
                IntErrorKind::Empty => "`limit` must be a non-negative integer",
                IntErrorKind::InvalidDigit => "not a valid integer",
                IntErrorKind::NegOverflow => {
                    panic!(
                        "`limit` should never negatively overflow since we're parsing into a usize and we'd get Empty instead"
                    )
                }
                IntErrorKind::Zero => {
                    panic!("zero is a valid `limit` so should have returned Ok() when parsing")
                }
                kind => panic!("unimplemented IntErrorKind variant: {:?}", kind),
            },
        };

        self.emit_err(LimitInvalid { span: self.attr_span, value_span: nv.value_span, error_str });

        None
    }
}
