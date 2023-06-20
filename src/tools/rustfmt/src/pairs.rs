use rustc_ast::ast;

use crate::config::lists::*;
use crate::config::IndentStyle;
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::Shape;
use crate::utils::{
    first_line_width, is_single_line, last_line_width, trimmed_last_line_width, wrap_str,
};

/// Sigils that decorate a binop pair.
#[derive(Clone, Copy)]
pub(crate) struct PairParts<'a> {
    prefix: &'a str,
    infix: &'a str,
    suffix: &'a str,
}

impl<'a> PairParts<'a> {
    pub(crate) const fn new(prefix: &'a str, infix: &'a str, suffix: &'a str) -> Self {
        Self {
            prefix,
            infix,
            suffix,
        }
    }
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
    context: &RewriteContext<'_>,
) -> Option<String> {
    expr.flatten(context, shape).and_then(|list| {
        // First we try formatting on one line.
        rewrite_pairs_one_line(&list, shape, context)
            .or_else(|| rewrite_pairs_multiline(&list, shape, context))
    })
}

// This may return a multi-line result since we allow the last expression to go
// multiline in a 'single line' formatting.
fn rewrite_pairs_one_line<T: Rewrite>(
    list: &PairList<'_, '_, T>,
    shape: Shape,
    context: &RewriteContext<'_>,
) -> Option<String> {
    assert!(list.list.len() >= 2, "Not a pair?");

    let mut result = String::new();
    let base_shape = shape.block();

    for ((_, rewrite), s) in list.list.iter().zip(list.separators.iter()) {
        if let Some(rewrite) = rewrite {
            if !is_single_line(rewrite) || result.len() > shape.width {
                return None;
            }

            result.push_str(rewrite);
            result.push(' ');
            result.push_str(s);
            result.push(' ');
        } else {
            return None;
        }
    }

    let prefix_len = result.len();
    let last = list.list.last()?.0;
    let cur_shape = base_shape.offset_left(last_line_width(&result))?;
    let last_rewrite = last.rewrite(context, cur_shape)?;
    result.push_str(&last_rewrite);

    if first_line_width(&result) > shape.width {
        return None;
    }

    // Check the last expression in the list. We sometimes let this expression
    // go over multiple lines, but we check for some ugly conditions.
    if !(is_single_line(&result) || last_rewrite.starts_with('{'))
        && (last_rewrite.starts_with('(') || prefix_len > context.config.tab_spaces())
    {
        return None;
    }

    wrap_str(result, context.config.max_width(), shape)
}

fn rewrite_pairs_multiline<T: Rewrite>(
    list: &PairList<'_, '_, T>,
    shape: Shape,
    context: &RewriteContext<'_>,
) -> Option<String> {
    let rhs_offset = shape.rhs_overhead(context.config);
    let nested_shape = (match context.config.indent_style() {
        IndentStyle::Visual => shape.visual_indent(0),
        IndentStyle::Block => shape.block_indent(context.config.tab_spaces()),
    })
    .with_max_width(context.config)
    .sub_width(rhs_offset)?;

    let indent_str = nested_shape.indent.to_string_with_newline(context.config);
    let mut result = String::new();

    result.push_str(list.list[0].1.as_ref()?);

    for ((e, default_rw), s) in list.list[1..].iter().zip(list.separators.iter()) {
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

        match context.config.binop_separator() {
            SeparatorPlace::Back => {
                result.push(' ');
                result.push_str(s);
                result.push_str(&indent_str);
            }
            SeparatorPlace::Front => {
                result.push_str(&indent_str);
                result.push_str(s);
                result.push(' ');
            }
        }

        result.push_str(default_rw.as_ref()?);
    }
    Some(result)
}

// Rewrites a single pair.
pub(crate) fn rewrite_pair<LHS, RHS>(
    lhs: &LHS,
    rhs: &RHS,
    pp: PairParts<'_>,
    context: &RewriteContext<'_>,
    shape: Shape,
    separator_place: SeparatorPlace,
) -> Option<String>
where
    LHS: Rewrite,
    RHS: Rewrite,
{
    let tab_spaces = context.config.tab_spaces();
    let lhs_overhead = match separator_place {
        SeparatorPlace::Back => shape.used_width() + pp.prefix.len() + pp.infix.trim_end().len(),
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
        let allow_same_line = lhs_result.len() <= tab_spaces
            || rhs_result
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
        SeparatorPlace::Back => pp.infix.trim_end(),
        SeparatorPlace::Front => pp.infix.trim_start(),
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
    fn flatten(&self, _: &RewriteContext<'_>, _: Shape) -> Option<PairList<'_, '_, Self>> {
        None
    }
}

struct PairList<'a, 'b, T: Rewrite> {
    list: Vec<(&'b T, Option<String>)>,
    separators: Vec<&'a str>,
}

impl FlattenPair for ast::Expr {
    fn flatten(
        &self,
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> Option<PairList<'_, '_, ast::Expr>> {
        let top_op = match self.kind {
            ast::ExprKind::Binary(op, _, _) => op.node,
            _ => return None,
        };

        let default_rewrite = |node: &ast::Expr, sep: usize, is_first: bool| {
            if is_first {
                return node.rewrite(context, shape);
            }
            let nested_overhead = sep + 1;
            let rhs_offset = shape.rhs_overhead(context.config);
            let nested_shape = (match context.config.indent_style() {
                IndentStyle::Visual => shape.visual_indent(0),
                IndentStyle::Block => shape.block_indent(context.config.tab_spaces()),
            })
            .with_max_width(context.config)
            .sub_width(rhs_offset)?;
            let default_shape = match context.config.binop_separator() {
                SeparatorPlace::Back => nested_shape.sub_width(nested_overhead)?,
                SeparatorPlace::Front => nested_shape.offset_left(nested_overhead)?,
            };
            node.rewrite(context, default_shape)
        };

        // Turn a tree of binop expressions into a list using a depth-first,
        // in-order traversal.
        let mut stack = vec![];
        let mut list = vec![];
        let mut separators = vec![];
        let mut node = self;
        loop {
            match node.kind {
                ast::ExprKind::Binary(op, ref lhs, _) if op.node == top_op => {
                    stack.push(node);
                    node = lhs;
                }
                _ => {
                    let op_len = separators.last().map_or(0, |s: &&str| s.len());
                    let rw = default_rewrite(node, op_len, list.is_empty());
                    list.push((node, rw));
                    if let Some(pop) = stack.pop() {
                        match pop.kind {
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
        Some(PairList { list, separators })
    }
}

impl FlattenPair for ast::Ty {}
impl FlattenPair for ast::Pat {}
