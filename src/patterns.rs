// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::{self, BindingMode, FieldPat, Pat, PatKind, RangeEnd};
use syntax::codemap::{self, BytePos, Span};
use syntax::ptr;

use {Shape, Spanned};
use codemap::SpanUtils;
use comment::FindUncommented;
use expr::{can_be_overflowed_expr, rewrite_call_inner, rewrite_pair, rewrite_unary_prefix,
           wrap_struct_field};
use lists::{itemize_list, shape_for_tactic, struct_lit_formatting, struct_lit_shape,
            struct_lit_tactic, write_list, DefinitiveListTactic, SeparatorTactic};
use rewrite::{Rewrite, RewriteContext};
use types::{rewrite_path, PathContext};
use utils::{format_mutability, mk_sp, wrap_str};

impl Rewrite for Pat {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self.node {
            PatKind::Box(ref pat) => rewrite_unary_prefix(context, "box ", &**pat, shape),
            PatKind::Ident(binding_mode, ident, ref sub_pat) => {
                let (prefix, mutability) = match binding_mode {
                    BindingMode::ByRef(mutability) => ("ref ", mutability),
                    BindingMode::ByValue(mutability) => ("", mutability),
                };
                let mut_infix = format_mutability(mutability);
                let id_str = ident.node.to_string();
                let sub_pat = match *sub_pat {
                    Some(ref p) => {
                        // 3 - ` @ `.
                        let width = try_opt!(
                            shape
                                .width
                                .checked_sub(prefix.len() + mut_infix.len() + id_str.len() + 3)
                        );
                        format!(
                            " @ {}",
                            try_opt!(p.rewrite(context, Shape::legacy(width, shape.indent)))
                        )
                    }
                    None => "".to_owned(),
                };

                let result = format!("{}{}{}{}", prefix, mut_infix, id_str, sub_pat);
                wrap_str(result, context.config.max_width(), shape)
            }
            PatKind::Wild => if 1 <= shape.width {
                Some("_".to_owned())
            } else {
                None
            },
            PatKind::Range(ref lhs, ref rhs, ref end_kind) => match *end_kind {
                RangeEnd::Included => rewrite_pair(&**lhs, &**rhs, "", "...", "", context, shape),
                RangeEnd::Excluded => rewrite_pair(&**lhs, &**rhs, "", "..", "", context, shape),
            },
            PatKind::Ref(ref pat, mutability) => {
                let prefix = format!("&{}", format_mutability(mutability));
                rewrite_unary_prefix(context, &prefix, &**pat, shape)
            }
            PatKind::Tuple(ref items, dotdot_pos) => {
                rewrite_tuple_pat(items, dotdot_pos, None, self.span, context, shape)
            }
            PatKind::Path(ref q_self, ref path) => {
                rewrite_path(context, PathContext::Expr, q_self.as_ref(), path, shape)
            }
            PatKind::TupleStruct(ref path, ref pat_vec, dotdot_pos) => {
                let path_str =
                    try_opt!(rewrite_path(context, PathContext::Expr, None, path, shape));
                rewrite_tuple_pat(
                    pat_vec,
                    dotdot_pos,
                    Some(path_str),
                    self.span,
                    context,
                    shape,
                )
            }
            PatKind::Lit(ref expr) => expr.rewrite(context, shape),
            PatKind::Slice(ref prefix, ref slice_pat, ref suffix) => {
                // Rewrite all the sub-patterns.
                let prefix = prefix.iter().map(|p| p.rewrite(context, shape));
                let slice_pat = slice_pat.as_ref().map(|p| {
                    Some(format!("{}..", try_opt!(p.rewrite(context, shape))))
                });
                let suffix = suffix.iter().map(|p| p.rewrite(context, shape));

                // Munge them together.
                let pats: Option<Vec<String>> =
                    prefix.chain(slice_pat.into_iter()).chain(suffix).collect();

                // Check that all the rewrites succeeded, and if not return None.
                let pats = try_opt!(pats);

                // Unwrap all the sub-strings and join them with commas.
                let result = if context.config.spaces_within_square_brackets() {
                    format!("[ {} ]", pats.join(", "))
                } else {
                    format!("[{}]", pats.join(", "))
                };
                wrap_str(result, context.config.max_width(), shape)
            }
            PatKind::Struct(ref path, ref fields, elipses) => {
                rewrite_struct_pat(path, fields, elipses, self.span, context, shape)
            }
            // FIXME(#819) format pattern macros.
            PatKind::Mac(..) => wrap_str(
                context.snippet(self.span),
                context.config.max_width(),
                shape,
            ),
        }
    }
}

fn rewrite_struct_pat(
    path: &ast::Path,
    fields: &[codemap::Spanned<ast::FieldPat>],
    elipses: bool,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    // 2 =  ` {`
    let path_shape = try_opt!(shape.sub_width(2));
    let path_str = try_opt!(rewrite_path(
        context,
        PathContext::Expr,
        None,
        path,
        path_shape,
    ));

    if fields.len() == 0 && !elipses {
        return Some(format!("{} {{}}", path_str));
    }

    let (elipses_str, terminator) = if elipses { (", ..", "..") } else { ("", "}") };

    // 3 = ` { `, 2 = ` }`.
    let (h_shape, v_shape) = try_opt!(struct_lit_shape(
        shape,
        context,
        path_str.len() + 3,
        elipses_str.len() + 2,
    ));

    let items = itemize_list(
        context.codemap,
        fields.iter(),
        terminator,
        |f| f.span.lo(),
        |f| f.span.hi(),
        |f| f.node.rewrite(context, v_shape),
        context.codemap.span_after(span, "{"),
        span.hi(),
        false,
    );
    let item_vec = items.collect::<Vec<_>>();

    let tactic = struct_lit_tactic(h_shape, context, &item_vec);
    let nested_shape = shape_for_tactic(tactic, h_shape, v_shape);
    let fmt = struct_lit_formatting(nested_shape, tactic, context, false);

    let mut fields_str = try_opt!(write_list(&item_vec, &fmt));
    let one_line_width = h_shape.map_or(0, |shape| shape.width);

    if elipses {
        if fields_str.contains('\n') || fields_str.len() > one_line_width {
            // Add a missing trailing comma.
            if fmt.trailing_separator == SeparatorTactic::Never {
                fields_str.push_str(",");
            }
            fields_str.push_str("\n");
            fields_str.push_str(&nested_shape.indent.to_string(context.config));
            fields_str.push_str("..");
        } else {
            if !fields_str.is_empty() {
                // there are preceeding struct fields being matched on
                if fmt.tactic == DefinitiveListTactic::Vertical {
                    // if the tactic is Vertical, write_list already added a trailing ,
                    fields_str.push_str(" ");
                } else {
                    fields_str.push_str(", ");
                }
            }
            fields_str.push_str("..");
        }
    }

    let fields_str = wrap_struct_field(context, &fields_str, shape, v_shape, one_line_width);
    Some(format!("{} {{{}}}", path_str, fields_str))
}

impl Rewrite for FieldPat {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let pat = self.pat.rewrite(context, shape);
        if self.is_shorthand {
            pat
        } else {
            wrap_str(
                format!("{}: {}", self.ident.to_string(), try_opt!(pat)),
                context.config.max_width(),
                shape,
            )
        }
    }
}

pub enum TuplePatField<'a> {
    Pat(&'a ptr::P<ast::Pat>),
    Dotdot(Span),
}

impl<'a> Rewrite for TuplePatField<'a> {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            TuplePatField::Pat(ref p) => p.rewrite(context, shape),
            TuplePatField::Dotdot(_) => Some("..".to_string()),
        }
    }
}

impl<'a> Spanned for TuplePatField<'a> {
    fn span(&self) -> Span {
        match *self {
            TuplePatField::Pat(ref p) => p.span(),
            TuplePatField::Dotdot(span) => span,
        }
    }
}

pub fn can_be_overflowed_pat(context: &RewriteContext, pat: &TuplePatField, len: usize) -> bool {
    match pat {
        &TuplePatField::Pat(ref pat) => match pat.node {
            ast::PatKind::Path(..) | ast::PatKind::Tuple(..) | ast::PatKind::Struct(..) => {
                context.use_block_indent() && len == 1
            }
            ast::PatKind::Ref(ref p, _) | ast::PatKind::Box(ref p) => {
                can_be_overflowed_pat(context, &TuplePatField::Pat(p), len)
            }
            ast::PatKind::Lit(ref expr) => can_be_overflowed_expr(context, expr, len),
            _ => false,
        },
        &TuplePatField::Dotdot(..) => false,
    }
}

fn rewrite_tuple_pat(
    pats: &[ptr::P<ast::Pat>],
    dotdot_pos: Option<usize>,
    path_str: Option<String>,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let mut pat_vec: Vec<_> = pats.into_iter().map(|x| TuplePatField::Pat(x)).collect();

    if let Some(pos) = dotdot_pos {
        let prev = if pos == 0 {
            span.lo()
        } else {
            pats[pos - 1].span().hi()
        };
        let next = if pos + 1 >= pats.len() {
            span.hi()
        } else {
            pats[pos + 1].span().lo()
        };
        let dot_span = mk_sp(prev, next);
        let snippet = context.snippet(dot_span);
        let lo = dot_span.lo() + BytePos(snippet.find_uncommented("..").unwrap() as u32);
        let dotdot = TuplePatField::Dotdot(Span::new(
            lo,
            // 2 == "..".len()
            lo + BytePos(2),
            codemap::NO_EXPANSION,
        ));
        pat_vec.insert(pos, dotdot);
    }

    if pat_vec.is_empty() {
        return Some(format!("{}()", path_str.unwrap_or(String::new())));
    }

    let wildcard_suffix_len = count_wildcard_suffix_len(context, &pat_vec, span, shape);
    let (pat_vec, span) = if context.config.condense_wildcard_suffixes() && wildcard_suffix_len >= 2
    {
        let new_item_count = 1 + pat_vec.len() - wildcard_suffix_len;
        let sp = pat_vec[new_item_count - 1].span();
        let snippet = context.snippet(sp);
        let lo = sp.lo() + BytePos(snippet.find_uncommented("_").unwrap() as u32);
        pat_vec[new_item_count - 1] = TuplePatField::Dotdot(mk_sp(lo, lo + BytePos(1)));
        (
            &pat_vec[..new_item_count],
            mk_sp(span.lo(), lo + BytePos(1)),
        )
    } else {
        (&pat_vec[..], span)
    };

    // add comma if `(x,)`
    let add_comma = path_str.is_none() && pat_vec.len() == 1 && dotdot_pos.is_none();
    let mut context = context.clone();
    if let Some(&TuplePatField::Dotdot(..)) = pat_vec.last() {
        context.inside_macro = true;
    }
    let path_str = path_str.unwrap_or(String::new());
    let mut pat_ref_vec = Vec::with_capacity(pat_vec.len());
    for pat in pat_vec {
        pat_ref_vec.push(pat);
    }
    return rewrite_call_inner(
        &context,
        &path_str,
        &pat_ref_vec[..],
        span,
        shape,
        shape.width,
        add_comma,
    ).ok();
}

fn count_wildcard_suffix_len(
    context: &RewriteContext,
    patterns: &[TuplePatField],
    span: Span,
    shape: Shape,
) -> usize {
    let mut suffix_len = 0;

    let items: Vec<_> = itemize_list(
        context.codemap,
        patterns.iter(),
        ")",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| item.rewrite(context, shape),
        context.codemap.span_after(span, "("),
        span.hi() - BytePos(1),
        false,
    ).collect();

    for item in items.iter().rev().take_while(|i| match i.item {
        Some(ref internal_string) if internal_string == "_" => true,
        _ => false,
    }) {
        suffix_len += 1;

        if item.pre_comment.is_some() || item.post_comment.is_some() {
            break;
        }
    }

    suffix_len
}
