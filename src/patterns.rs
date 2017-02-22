// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Shape;
use codemap::SpanUtils;
use rewrite::{Rewrite, RewriteContext};
use utils::{wrap_str, format_mutability};
use lists::{format_item_list, itemize_list, ListItem};
use expr::{rewrite_unary_prefix, rewrite_pair};
use types::{rewrite_path, PathContext};
use super::Spanned;
use comment::FindUncommented;

use syntax::ast::{self, BindingMode, Pat, PatKind, FieldPat};
use syntax::ptr;
use syntax::codemap::{self, BytePos, Span};

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
                        let width = try_opt!(shape.width
                                                 .checked_sub(prefix.len() + mut_infix.len() +
                                                              id_str.len() +
                                                              3));
                        format!(" @ {}",
                                try_opt!(p.rewrite(context, Shape::legacy(width, shape.indent))))
                    }
                    None => "".to_owned(),
                };

                let result = format!("{}{}{}{}", prefix, mut_infix, id_str, sub_pat);
                wrap_str(result, context.config.max_width, shape)
            }
            PatKind::Wild => {
                if 1 <= shape.width {
                    Some("_".to_owned())
                } else {
                    None
                }
            }
            PatKind::Range(ref lhs, ref rhs) => {
                rewrite_pair(&**lhs, &**rhs, "", "...", "", context, shape)
            }
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
                rewrite_tuple_pat(pat_vec,
                                  dotdot_pos,
                                  Some(path_str),
                                  self.span,
                                  context,
                                  shape)
            }
            PatKind::Lit(ref expr) => expr.rewrite(context, shape),
            PatKind::Slice(ref prefix, ref slice_pat, ref suffix) => {
                // Rewrite all the sub-patterns.
                let prefix = prefix.iter().map(|p| p.rewrite(context, shape));
                let slice_pat =
                    slice_pat.as_ref()
                        .map(|p| Some(format!("{}..", try_opt!(p.rewrite(context, shape)))));
                let suffix = suffix.iter().map(|p| p.rewrite(context, shape));

                // Munge them together.
                let pats: Option<Vec<String>> = prefix.chain(slice_pat.into_iter())
                    .chain(suffix)
                    .collect();

                // Check that all the rewrites succeeded, and if not return None.
                let pats = try_opt!(pats);

                // Unwrap all the sub-strings and join them with commas.
                let result = if context.config.spaces_within_square_brackets {
                    format!("[ {} ]", pats.join(", "))
                } else {
                    format!("[{}]", pats.join(", "))
                };
                wrap_str(result, context.config.max_width, shape)
            }
            PatKind::Struct(ref path, ref fields, elipses) => {
                let path = try_opt!(rewrite_path(context, PathContext::Expr, None, path, shape));

                let (elipses_str, terminator) = if elipses { (", ..", "..") } else { ("", "}") };

                // 5 = `{` plus space before and after plus `}` plus space before.
                let budget = try_opt!(shape.width.checked_sub(path.len() + 5 + elipses_str.len()));
                // FIXME Using visual indenting, should use block or visual to match
                // struct lit preference (however, in practice I think it is rare
                // for struct patterns to be multi-line).
                // 3 = `{` plus space before and after.
                let offset = shape.indent + path.len() + 3;

                let items =
                    itemize_list(context.codemap,
                                 fields.iter(),
                                 terminator,
                                 |f| f.span.lo,
                                 |f| f.span.hi,
                                 |f| f.node.rewrite(context, Shape::legacy(budget, offset)),
                                 context.codemap.span_after(self.span, "{"),
                                 self.span.hi);
                let mut field_string = try_opt!(format_item_list(items,
                                                                 Shape::legacy(budget, offset),
                                                                 context.config));
                if elipses {
                    if field_string.contains('\n') {
                        field_string.push_str(",\n");
                        field_string.push_str(&offset.to_string(context.config));
                        field_string.push_str("..");
                    } else {
                        if !field_string.is_empty() {
                            field_string.push_str(", ");
                        }
                        field_string.push_str("..");
                    }
                }

                if field_string.is_empty() {
                    Some(format!("{} {{}}", path))
                } else {
                    Some(format!("{} {{ {} }}", path, field_string))
                }
            }
            // FIXME(#819) format pattern macros.
            PatKind::Mac(..) => {
                wrap_str(context.snippet(self.span), context.config.max_width, shape)
            }
        }
    }
}

impl Rewrite for FieldPat {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let pat = self.pat.rewrite(context, shape);
        if self.is_shorthand {
            pat
        } else {
            wrap_str(format!("{}: {}", self.ident.to_string(), try_opt!(pat)),
                     context.config.max_width,
                     shape)
        }
    }
}


enum TuplePatField<'a> {
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

fn rewrite_tuple_pat(pats: &[ptr::P<ast::Pat>],
                     dotdot_pos: Option<usize>,
                     path_str: Option<String>,
                     span: Span,
                     context: &RewriteContext,
                     shape: Shape)
                     -> Option<String> {
    let mut pat_vec: Vec<_> = pats.into_iter().map(|x| TuplePatField::Pat(x)).collect();

    if let Some(pos) = dotdot_pos {
        let snippet = context.snippet(span);
        let lo = span.lo + BytePos(snippet.find_uncommented("..").unwrap() as u32);
        let span = Span {
            lo: lo,
            // 2 == "..".len()
            hi: lo + BytePos(2),
            expn_id: codemap::NO_EXPANSION,
        };
        let dotdot = TuplePatField::Dotdot(span);
        pat_vec.insert(pos, dotdot);
    }

    if pat_vec.is_empty() {
        path_str
    } else {
        // add comma if `(x,)`
        let add_comma = path_str.is_none() && pat_vec.len() == 1 && dotdot_pos.is_none();

        let path_len = path_str.as_ref().map(|p| p.len()).unwrap_or(0);
        // 2 = "()".len(), 3 = "(,)".len()
        let nested_shape = try_opt!(shape.sub_width(path_len + if add_comma { 3 } else { 2 }));
        // 1 = "(".len()
        let nested_shape = nested_shape.visual_indent(path_len + 1);
        let mut items: Vec<_> = itemize_list(context.codemap,
                                             pat_vec.iter(),
                                             if add_comma { ",)" } else { ")" },
                                             |item| item.span().lo,
                                             |item| item.span().hi,
                                             |item| item.rewrite(context, nested_shape),
                                             context.codemap.span_after(span, "("),
                                             span.hi - BytePos(1))
                .collect();

        // Condense wildcard string suffix into a single ..
        let wildcard_suffix_len = count_wildcard_suffix_len(&items);

        let list = if context.config.condense_wildcard_suffices && wildcard_suffix_len >= 2 {
            let new_item_count = 1 + pats.len() - wildcard_suffix_len;
            items[new_item_count - 1].item = Some("..".to_owned());

            let da_iter = items.into_iter().take(new_item_count);
            try_opt!(format_item_list(da_iter, nested_shape, context.config))
        } else {
            try_opt!(format_item_list(items.into_iter(), nested_shape, context.config))
        };

        match path_str {
            Some(path_str) => {
                Some(if context.config.spaces_within_parens {
                         format!("{}( {} )", path_str, list)
                     } else {
                         format!("{}({})", path_str, list)
                     })
            }
            None => {
                let comma = if add_comma { "," } else { "" };
                Some(if context.config.spaces_within_parens {
                         format!("( {}{} )", list, comma)
                     } else {
                         format!("({}{})", list, comma)
                     })
            }
        }
    }
}

fn count_wildcard_suffix_len(items: &[ListItem]) -> usize {
    let mut suffix_len = 0;

    for item in items.iter().rev().take_while(|i| match i.item {
                                                  Some(ref internal_string) if internal_string ==
                                                                               "_" => true,
                                                  _ => false,
                                              }) {
        suffix_len += 1;

        if item.pre_comment.is_some() || item.post_comment.is_some() {
            break;
        }
    }

    suffix_len
}
