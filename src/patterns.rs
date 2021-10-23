use rustc_ast::ast::{self, BindingMode, Pat, PatField, PatKind, RangeEnd, RangeSyntax};
use rustc_ast::ptr;
use rustc_span::{BytePos, Span};

use crate::comment::{combine_strs_with_missing_comments, FindUncommented};
use crate::config::lists::*;
use crate::config::Version;
use crate::expr::{can_be_overflowed_expr, rewrite_unary_prefix, wrap_struct_field};
use crate::lists::{
    definitive_tactic, itemize_list, shape_for_tactic, struct_lit_formatting, struct_lit_shape,
    struct_lit_tactic, write_list, ListFormatting, ListItem, Separator,
};
use crate::macros::{rewrite_macro, MacroPosition};
use crate::overflow;
use crate::pairs::{rewrite_pair, PairParts};
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::types::{rewrite_path, PathContext};
use crate::utils::{format_mutability, mk_sp, mk_sp_lo_plus_one, rewrite_ident};

/// Returns `true` if the given pattern is "short".
/// A short pattern is defined by the following grammar:
///
/// `[small, ntp]`:
///     - single token
///     - `&[single-line, ntp]`
///
/// `[small]`:
///     - `[small, ntp]`
///     - unary tuple constructor `([small, ntp])`
///     - `&[small]`
pub(crate) fn is_short_pattern(pat: &ast::Pat, pat_str: &str) -> bool {
    // We also require that the pattern is reasonably 'small' with its literal width.
    pat_str.len() <= 20 && !pat_str.contains('\n') && is_short_pattern_inner(pat)
}

fn is_short_pattern_inner(pat: &ast::Pat) -> bool {
    match pat.kind {
        ast::PatKind::Rest | ast::PatKind::Wild | ast::PatKind::Lit(_) => true,
        ast::PatKind::Ident(_, _, ref pat) => pat.is_none(),
        ast::PatKind::Struct(..)
        | ast::PatKind::MacCall(..)
        | ast::PatKind::Slice(..)
        | ast::PatKind::Path(..)
        | ast::PatKind::Range(..) => false,
        ast::PatKind::Tuple(ref subpats) => subpats.len() <= 1,
        ast::PatKind::TupleStruct(_, ref path, ref subpats) => {
            path.segments.len() <= 1 && subpats.len() <= 1
        }
        ast::PatKind::Box(ref p) | ast::PatKind::Ref(ref p, _) | ast::PatKind::Paren(ref p) => {
            is_short_pattern_inner(&*p)
        }
        PatKind::Or(ref pats) => pats.iter().all(|p| is_short_pattern_inner(p)),
    }
}

struct RangeOperand<'a>(&'a Option<ptr::P<ast::Expr>>);

impl<'a> Rewrite for RangeOperand<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        match &self.0 {
            None => Some("".to_owned()),
            Some(ref exp) => exp.rewrite(context, shape),
        }
    }
}

impl Rewrite for Pat {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        match self.kind {
            PatKind::Or(ref pats) => {
                let pat_strs = pats
                    .iter()
                    .map(|p| p.rewrite(context, shape))
                    .collect::<Option<Vec<_>>>()?;

                let use_mixed_layout = pats
                    .iter()
                    .zip(pat_strs.iter())
                    .all(|(pat, pat_str)| is_short_pattern(pat, pat_str));
                let items: Vec<_> = pat_strs.into_iter().map(ListItem::from_str).collect();
                let tactic = if use_mixed_layout {
                    DefinitiveListTactic::Mixed
                } else {
                    definitive_tactic(
                        &items,
                        ListTactic::HorizontalVertical,
                        Separator::VerticalBar,
                        shape.width,
                    )
                };
                let fmt = ListFormatting::new(shape, context.config)
                    .tactic(tactic)
                    .separator(" |")
                    .separator_place(context.config.binop_separator())
                    .ends_with_newline(false);
                write_list(&items, &fmt)
            }
            PatKind::Box(ref pat) => rewrite_unary_prefix(context, "box ", &**pat, shape),
            PatKind::Ident(binding_mode, ident, ref sub_pat) => {
                let (prefix, mutability) = match binding_mode {
                    BindingMode::ByRef(mutability) => ("ref", mutability),
                    BindingMode::ByValue(mutability) => ("", mutability),
                };
                let mut_infix = format_mutability(mutability).trim();
                let id_str = rewrite_ident(context, ident);
                let sub_pat = match *sub_pat {
                    Some(ref p) => {
                        // 2 - `@ `.
                        let width = shape
                            .width
                            .checked_sub(prefix.len() + mut_infix.len() + id_str.len() + 2)?;
                        let lo = context.snippet_provider.span_after(self.span, "@");
                        combine_strs_with_missing_comments(
                            context,
                            "@",
                            &p.rewrite(context, Shape::legacy(width, shape.indent))?,
                            mk_sp(lo, p.span.lo()),
                            shape,
                            true,
                        )?
                    }
                    None => "".to_owned(),
                };

                // combine prefix and mut
                let (first_lo, first) = if !prefix.is_empty() && !mut_infix.is_empty() {
                    let hi = context.snippet_provider.span_before(self.span, "mut");
                    let lo = context.snippet_provider.span_after(self.span, "ref");
                    (
                        context.snippet_provider.span_after(self.span, "mut"),
                        combine_strs_with_missing_comments(
                            context,
                            prefix,
                            mut_infix,
                            mk_sp(lo, hi),
                            shape,
                            true,
                        )?,
                    )
                } else if !prefix.is_empty() {
                    (
                        context.snippet_provider.span_after(self.span, "ref"),
                        prefix.to_owned(),
                    )
                } else if !mut_infix.is_empty() {
                    (
                        context.snippet_provider.span_after(self.span, "mut"),
                        mut_infix.to_owned(),
                    )
                } else {
                    (self.span.lo(), "".to_owned())
                };

                let next = if !sub_pat.is_empty() {
                    let hi = context.snippet_provider.span_before(self.span, "@");
                    combine_strs_with_missing_comments(
                        context,
                        id_str,
                        &sub_pat,
                        mk_sp(ident.span.hi(), hi),
                        shape,
                        true,
                    )?
                } else {
                    id_str.to_owned()
                };

                combine_strs_with_missing_comments(
                    context,
                    &first,
                    &next,
                    mk_sp(first_lo, ident.span.lo()),
                    shape,
                    true,
                )
            }
            PatKind::Wild => {
                if 1 <= shape.width {
                    Some("_".to_owned())
                } else {
                    None
                }
            }
            PatKind::Rest => {
                if 1 <= shape.width {
                    Some("..".to_owned())
                } else {
                    None
                }
            }
            PatKind::Range(ref lhs, ref rhs, ref end_kind) => {
                let infix = match end_kind.node {
                    RangeEnd::Included(RangeSyntax::DotDotDot) => "...",
                    RangeEnd::Included(RangeSyntax::DotDotEq) => "..=",
                    RangeEnd::Excluded => "..",
                };
                let infix = if context.config.spaces_around_ranges() {
                    let lhs_spacing = match lhs {
                        None => "",
                        Some(_) => " ",
                    };
                    let rhs_spacing = match rhs {
                        None => "",
                        Some(_) => " ",
                    };
                    format!("{}{}{}", lhs_spacing, infix, rhs_spacing)
                } else {
                    infix.to_owned()
                };
                rewrite_pair(
                    &RangeOperand(lhs),
                    &RangeOperand(rhs),
                    PairParts::infix(&infix),
                    context,
                    shape,
                    SeparatorPlace::Front,
                )
            }
            PatKind::Ref(ref pat, mutability) => {
                let prefix = format!("&{}", format_mutability(mutability));
                rewrite_unary_prefix(context, &prefix, &**pat, shape)
            }
            PatKind::Tuple(ref items) => rewrite_tuple_pat(items, None, self.span, context, shape),
            PatKind::Path(ref q_self, ref path) => {
                rewrite_path(context, PathContext::Expr, q_self.as_ref(), path, shape)
            }
            PatKind::TupleStruct(ref q_self, ref path, ref pat_vec) => {
                let path_str =
                    rewrite_path(context, PathContext::Expr, q_self.as_ref(), path, shape)?;
                rewrite_tuple_pat(pat_vec, Some(path_str), self.span, context, shape)
            }
            PatKind::Lit(ref expr) => expr.rewrite(context, shape),
            PatKind::Slice(ref slice_pat) if context.config.version() == Version::One => {
                let rw: Vec<String> = slice_pat
                    .iter()
                    .map(|p| {
                        if let Some(rw) = p.rewrite(context, shape) {
                            rw
                        } else {
                            context.snippet(p.span).to_string()
                        }
                    })
                    .collect();
                Some(format!("[{}]", rw.join(", ")))
            }
            PatKind::Slice(ref slice_pat) => overflow::rewrite_with_square_brackets(
                context,
                "",
                slice_pat.iter(),
                shape,
                self.span,
                None,
                None,
            ),
            PatKind::Struct(ref qself, ref path, ref fields, ellipsis) => {
                rewrite_struct_pat(qself, path, fields, ellipsis, self.span, context, shape)
            }
            PatKind::MacCall(ref mac) => {
                rewrite_macro(mac, None, context, shape, MacroPosition::Pat)
            }
            PatKind::Paren(ref pat) => pat
                .rewrite(context, shape.offset_left(1)?.sub_width(1)?)
                .map(|inner_pat| format!("({})", inner_pat)),
        }
    }
}

fn rewrite_struct_pat(
    qself: &Option<ast::QSelf>,
    path: &ast::Path,
    fields: &[ast::PatField],
    ellipsis: bool,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    // 2 =  ` {`
    let path_shape = shape.sub_width(2)?;
    let path_str = rewrite_path(context, PathContext::Expr, qself.as_ref(), path, path_shape)?;

    if fields.is_empty() && !ellipsis {
        return Some(format!("{} {{}}", path_str));
    }

    let (ellipsis_str, terminator) = if ellipsis { (", ..", "..") } else { ("", "}") };

    // 3 = ` { `, 2 = ` }`.
    let (h_shape, v_shape) =
        struct_lit_shape(shape, context, path_str.len() + 3, ellipsis_str.len() + 2)?;

    let items = itemize_list(
        context.snippet_provider,
        fields.iter(),
        terminator,
        ",",
        |f| {
            if f.attrs.is_empty() {
                f.span.lo()
            } else {
                f.attrs.first().unwrap().span.lo()
            }
        },
        |f| f.span.hi(),
        |f| f.rewrite(context, v_shape),
        context.snippet_provider.span_after(span, "{"),
        span.hi(),
        false,
    );
    let item_vec = items.collect::<Vec<_>>();

    let tactic = struct_lit_tactic(h_shape, context, &item_vec);
    let nested_shape = shape_for_tactic(tactic, h_shape, v_shape);
    let fmt = struct_lit_formatting(nested_shape, tactic, context, false);

    let mut fields_str = write_list(&item_vec, &fmt)?;
    let one_line_width = h_shape.map_or(0, |shape| shape.width);

    if ellipsis {
        if fields_str.contains('\n') || fields_str.len() > one_line_width {
            // Add a missing trailing comma.
            if context.config.trailing_comma() == SeparatorTactic::Never {
                fields_str.push(',');
            }
            fields_str.push('\n');
            fields_str.push_str(&nested_shape.indent.to_string(context.config));
        } else {
            if !fields_str.is_empty() {
                // there are preceding struct fields being matched on
                if tactic == DefinitiveListTactic::Vertical {
                    // if the tactic is Vertical, write_list already added a trailing ,
                    fields_str.push(' ');
                } else {
                    fields_str.push_str(", ");
                }
            }
        }
        fields_str.push_str("..");
    }

    // ast::Pat doesn't have attrs so use &[]
    let fields_str = wrap_struct_field(context, &[], &fields_str, shape, v_shape, one_line_width)?;
    Some(format!("{} {{{}}}", path_str, fields_str))
}

impl Rewrite for PatField {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        let hi_pos = if let Some(last) = self.attrs.last() {
            last.span.hi()
        } else {
            self.pat.span.lo()
        };

        let attrs_str = if self.attrs.is_empty() {
            String::from("")
        } else {
            self.attrs.rewrite(context, shape)?
        };

        let pat_str = self.pat.rewrite(context, shape)?;
        if self.is_shorthand {
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                &pat_str,
                mk_sp(hi_pos, self.pat.span.lo()),
                shape,
                false,
            )
        } else {
            let nested_shape = shape.block_indent(context.config.tab_spaces());
            let id_str = rewrite_ident(context, self.ident);
            let one_line_width = id_str.len() + 2 + pat_str.len();
            let pat_and_id_str = if one_line_width <= shape.width {
                format!("{}: {}", id_str, pat_str)
            } else {
                format!(
                    "{}:\n{}{}",
                    id_str,
                    nested_shape.indent.to_string(context.config),
                    self.pat.rewrite(context, nested_shape)?
                )
            };
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                &pat_and_id_str,
                mk_sp(hi_pos, self.pat.span.lo()),
                nested_shape,
                false,
            )
        }
    }
}

#[derive(Debug)]
pub(crate) enum TuplePatField<'a> {
    Pat(&'a ptr::P<ast::Pat>),
    Dotdot(Span),
}

impl<'a> Rewrite for TuplePatField<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        match *self {
            TuplePatField::Pat(p) => p.rewrite(context, shape),
            TuplePatField::Dotdot(_) => Some("..".to_string()),
        }
    }
}

impl<'a> Spanned for TuplePatField<'a> {
    fn span(&self) -> Span {
        match *self {
            TuplePatField::Pat(p) => p.span(),
            TuplePatField::Dotdot(span) => span,
        }
    }
}

impl<'a> TuplePatField<'a> {
    fn is_dotdot(&self) -> bool {
        match self {
            TuplePatField::Pat(pat) => matches!(pat.kind, ast::PatKind::Rest),
            TuplePatField::Dotdot(_) => true,
        }
    }
}

pub(crate) fn can_be_overflowed_pat(
    context: &RewriteContext<'_>,
    pat: &TuplePatField<'_>,
    len: usize,
) -> bool {
    match *pat {
        TuplePatField::Pat(pat) => match pat.kind {
            ast::PatKind::Path(..)
            | ast::PatKind::Tuple(..)
            | ast::PatKind::Struct(..)
            | ast::PatKind::TupleStruct(..) => context.use_block_indent() && len == 1,
            ast::PatKind::Ref(ref p, _) | ast::PatKind::Box(ref p) => {
                can_be_overflowed_pat(context, &TuplePatField::Pat(p), len)
            }
            ast::PatKind::Lit(ref expr) => can_be_overflowed_expr(context, expr, len),
            _ => false,
        },
        TuplePatField::Dotdot(..) => false,
    }
}

fn rewrite_tuple_pat(
    pats: &[ptr::P<ast::Pat>],
    path_str: Option<String>,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    if pats.is_empty() {
        return Some(format!("{}()", path_str.unwrap_or_default()));
    }
    let mut pat_vec: Vec<_> = pats.iter().map(TuplePatField::Pat).collect();

    let wildcard_suffix_len = count_wildcard_suffix_len(context, &pat_vec, span, shape);
    let (pat_vec, span) = if context.config.condense_wildcard_suffixes() && wildcard_suffix_len >= 2
    {
        let new_item_count = 1 + pat_vec.len() - wildcard_suffix_len;
        let sp = pat_vec[new_item_count - 1].span();
        let snippet = context.snippet(sp);
        let lo = sp.lo() + BytePos(snippet.find_uncommented("_").unwrap() as u32);
        pat_vec[new_item_count - 1] = TuplePatField::Dotdot(mk_sp_lo_plus_one(lo));
        (
            &pat_vec[..new_item_count],
            mk_sp(span.lo(), lo + BytePos(1)),
        )
    } else {
        (&pat_vec[..], span)
    };

    let is_last_pat_dotdot = pat_vec.last().map_or(false, |p| p.is_dotdot());
    let add_comma = path_str.is_none() && pat_vec.len() == 1 && !is_last_pat_dotdot;
    let path_str = path_str.unwrap_or_default();

    overflow::rewrite_with_parens(
        context,
        &path_str,
        pat_vec.iter(),
        shape,
        span,
        context.config.max_width(),
        if add_comma {
            Some(SeparatorTactic::Always)
        } else {
            None
        },
    )
}

fn count_wildcard_suffix_len(
    context: &RewriteContext<'_>,
    patterns: &[TuplePatField<'_>],
    span: Span,
    shape: Shape,
) -> usize {
    let mut suffix_len = 0;

    let items: Vec<_> = itemize_list(
        context.snippet_provider,
        patterns.iter(),
        ")",
        ",",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| item.rewrite(context, shape),
        context.snippet_provider.span_after(span, "("),
        span.hi() - BytePos(1),
        false,
    )
    .collect();

    for item in items
        .iter()
        .rev()
        .take_while(|i| matches!(i.item, Some(ref internal_string) if internal_string == "_"))
    {
        suffix_len += 1;

        if item.has_comment() {
            break;
        }
    }

    suffix_len
}
