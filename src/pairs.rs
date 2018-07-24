// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;

use config::lists::*;
use config::IndentStyle;
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use utils::{first_line_width, is_single_line, last_line_width, trimmed_last_line_width, wrap_str};

/// Sigils that decorate a binop pair.
#[derive(new, Clone, Copy)]
pub(crate) struct PairParts<'a> {
    prefix: &'a str,
    infix: &'a str,
    suffix: &'a str,
}

impl<'a> PairParts<'a> {
    pub(crate) fn infix(infix: &'a str) -> PairParts<'a> {
        PairParts {
            prefix: "",
            infix,
            suffix: "",
        }
    }
}

// Flattens a tree of pairs into a list and tries to rewrite them all at once.
// FIXME would be nice to reuse the lists API for this, but because each separator
// can be different, we can't.
pub(crate) fn rewrite_all_pairs(
    expr: &ast::Expr,
    shape: Shape,
    context: &RewriteContext,
) -> Option<String> {
    // First we try formatting on one line.
    if let Some(list) = expr.flatten(context, false) {
        if let Some(r) = rewrite_pairs_one_line(&list, shape, context) {
            return Some(r);
        }
    }

    // We can't format on line, so try many. When we flatten here we make sure
    // to only flatten pairs with the same operator, that way we don't
    // necessarily need one line per sub-expression, but we don't do anything
    // too funny wrt precedence.
    expr.flatten(context, true)
        .and_then(|list| rewrite_pairs_multiline(list, shape, context))
}

// This may return a multi-line result since we allow the last expression to go
// multiline in a 'single line' formatting.
fn rewrite_pairs_one_line<T: Rewrite>(
    list: &PairList<T>,
    shape: Shape,
    context: &RewriteContext,
) -> Option<String> {
    assert!(list.list.len() >= 2, "Not a pair?");

    let mut result = String::new();
    let base_shape = shape.block();

    for (e, s) in list.list.iter().zip(list.separators.iter()) {
        let cur_shape = base_shape.offset_left(last_line_width(&result))?;
        let rewrite = e.rewrite(context, cur_shape)?;

        if !is_single_line(&rewrite) || result.len() > shape.width {
            return None;
        }

        result.push_str(&rewrite);
        result.push(' ');
        result.push_str(s);
        result.push(' ');
    }

    let last = list.list.last().unwrap();
    let cur_shape = base_shape.offset_left(last_line_width(&result))?;
    let rewrite = last.rewrite(context, cur_shape)?;
    result.push_str(&rewrite);

    if first_line_width(&result) > shape.width {
        return None;
    }

    // Check the last expression in the list. We let this expression go over
    // multiple lines, but we check that if this is necessary, then we can't
    // do better using multi-line formatting.
    if !is_single_line(&result) {
        let multiline_shape = shape.offset_left(list.separators.last().unwrap().len() + 1)?;
        let multiline_list: PairList<T> = PairList {
            list: vec![last],
            separators: vec![],
            separator_place: list.separator_place,
        };
        // Format as if we were multi-line.
        if let Some(rewrite) = rewrite_pairs_multiline(multiline_list, multiline_shape, context) {
            // Also, don't let expressions surrounded by parens go multi-line,
            // this looks really bad.
            if rewrite.starts_with('(') || is_single_line(&rewrite) {
                return None;
            }
        }
    }

    wrap_str(result, context.config.max_width(), shape)
}

fn rewrite_pairs_multiline<T: Rewrite>(
    list: PairList<T>,
    shape: Shape,
    context: &RewriteContext,
) -> Option<String> {
    let rhs_offset = shape.rhs_overhead(&context.config);
    let nested_shape = (match context.config.indent_style() {
        IndentStyle::Visual => shape.visual_indent(0),
        IndentStyle::Block => shape.block_indent(context.config.tab_spaces()),
    }).with_max_width(&context.config)
    .sub_width(rhs_offset)?;

    let indent_str = nested_shape.indent.to_string_with_newline(context.config);
    let mut result = String::new();

    let rewrite = list.list[0].rewrite(context, shape)?;
    result.push_str(&rewrite);

    for (e, s) in list.list[1..].iter().zip(list.separators.iter()) {
        // The following test checks if we should keep two subexprs on the same
        // line. We do this if not doing so would create an orphan and there is
        // enough space to do so.
        let offset = if result.contains('\n') {
            0
        } else {
            shape.used_width()
        };
        if last_line_width(&result) + offset <= nested_shape.used_width() {
            // We must snuggle the next line onto the previous line to avoid an orphan.
            if let Some(line_shape) =
                shape.offset_left(s.len() + 2 + trimmed_last_line_width(&result))
            {
                if let Some(rewrite) = e.rewrite(context, line_shape) {
                    result.push(' ');
                    result.push_str(s);
                    result.push(' ');
                    result.push_str(&rewrite);
                    continue;
                }
            }
        }

        let nested_overhead = s.len() + 1;
        let line_shape = match context.config.binop_separator() {
            SeparatorPlace::Back => {
                result.push(' ');
                result.push_str(s);
                result.push_str(&indent_str);
                nested_shape.sub_width(nested_overhead)?
            }
            SeparatorPlace::Front => {
                result.push_str(&indent_str);
                result.push_str(s);
                result.push(' ');
                nested_shape.offset_left(nested_overhead)?
            }
        };

        let rewrite = e.rewrite(context, line_shape)?;
        result.push_str(&rewrite);
    }
    Some(result)
}

// Rewrites a single pair.
pub(crate) fn rewrite_pair<LHS, RHS>(
    lhs: &LHS,
    rhs: &RHS,
    pp: PairParts,
    context: &RewriteContext,
    shape: Shape,
    separator_place: SeparatorPlace,
) -> Option<String>
where
    LHS: Rewrite,
    RHS: Rewrite,
{
    let tab_spaces = context.config.tab_spaces();
    let lhs_overhead = match separator_place {
        SeparatorPlace::Back => shape.used_width() + pp.prefix.len() + pp.infix.trim_right().len(),
        SeparatorPlace::Front => shape.used_width(),
    };
    let lhs_shape = Shape {
        width: context.budget(lhs_overhead),
        ..shape
    };
    let lhs_result = lhs
        .rewrite(context, lhs_shape)
        .map(|lhs_str| format!("{}{}", pp.prefix, lhs_str))?;

    // Try to put both lhs and rhs on the same line.
    let rhs_orig_result = shape
        .offset_left(last_line_width(&lhs_result) + pp.infix.len())
        .and_then(|s| s.sub_width(pp.suffix.len()))
        .and_then(|rhs_shape| rhs.rewrite(context, rhs_shape));
    if let Some(ref rhs_result) = rhs_orig_result {
        // If the length of the lhs is equal to or shorter than the tab width or
        // the rhs looks like block expression, we put the rhs on the same
        // line with the lhs even if the rhs is multi-lined.
        let allow_same_line = lhs_result.len() <= tab_spaces || rhs_result
            .lines()
            .next()
            .map(|first_line| first_line.ends_with('{'))
            .unwrap_or(false);
        if !rhs_result.contains('\n') || allow_same_line {
            let one_line_width = last_line_width(&lhs_result)
                + pp.infix.len()
                + first_line_width(rhs_result)
                + pp.suffix.len();
            if one_line_width <= shape.width {
                return Some(format!(
                    "{}{}{}{}",
                    lhs_result, pp.infix, rhs_result, pp.suffix
                ));
            }
        }
    }

    // We have to use multiple lines.
    // Re-evaluate the rhs because we have more space now:
    let mut rhs_shape = match context.config.indent_style() {
        IndentStyle::Visual => shape
            .sub_width(pp.suffix.len() + pp.prefix.len())?
            .visual_indent(pp.prefix.len()),
        IndentStyle::Block => {
            // Try to calculate the initial constraint on the right hand side.
            let rhs_overhead = shape.rhs_overhead(context.config);
            Shape::indented(shape.indent.block_indent(context.config), context.config)
                .sub_width(rhs_overhead)?
        }
    };
    let infix = match separator_place {
        SeparatorPlace::Back => pp.infix.trim_right(),
        SeparatorPlace::Front => pp.infix.trim_left(),
    };
    if separator_place == SeparatorPlace::Front {
        rhs_shape = rhs_shape.offset_left(infix.len())?;
    }
    let rhs_result = rhs.rewrite(context, rhs_shape)?;
    let indent_str = rhs_shape.indent.to_string_with_newline(context.config);
    let infix_with_sep = match separator_place {
        SeparatorPlace::Back => format!("{}{}", infix, indent_str),
        SeparatorPlace::Front => format!("{}{}", indent_str, infix),
    };
    Some(format!(
        "{}{}{}{}",
        lhs_result, infix_with_sep, rhs_result, pp.suffix
    ))
}

// A pair which forms a tree and can be flattened (e.g., binops).
trait FlattenPair: Rewrite + Sized {
    // If `_same_op` is `true`, then we only combine binops with the same
    // operator into the list. E.g,, if the source is `a * b + c`, if `_same_op`
    // is true, we make `[(a * b), c]` if `_same_op` is false, we make
    // `[a, b, c]`
    fn flatten(&self, _context: &RewriteContext, _same_op: bool) -> Option<PairList<Self>> {
        None
    }
}

struct PairList<'a, 'b, T: Rewrite + 'b> {
    list: Vec<&'b T>,
    separators: Vec<&'a str>,
    separator_place: SeparatorPlace,
}

impl FlattenPair for ast::Expr {
    fn flatten(&self, context: &RewriteContext, same_op: bool) -> Option<PairList<ast::Expr>> {
        let top_op = match self.node {
            ast::ExprKind::Binary(op, _, _) => op.node,
            _ => return None,
        };

        // Turn a tree of binop expressions into a list using a depth-first,
        // in-order traversal.
        let mut stack = vec![];
        let mut list = vec![];
        let mut separators = vec![];
        let mut node = self;
        loop {
            match node.node {
                ast::ExprKind::Binary(op, ref lhs, _) if !same_op || op.node == top_op => {
                    stack.push(node);
                    node = lhs;
                }
                _ => {
                    list.push(node);
                    if let Some(pop) = stack.pop() {
                        match pop.node {
                            ast::ExprKind::Binary(op, _, ref rhs) => {
                                separators.push(op.node.to_string());
                                node = rhs;
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        assert_eq!(list.len() - 1, separators.len());
        Some(PairList {
            list,
            separators,
            separator_place: context.config.binop_separator(),
        })
    }
}

impl FlattenPair for ast::Ty {}
impl FlattenPair for ast::Pat {}
