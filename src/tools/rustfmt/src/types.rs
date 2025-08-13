use std::ops::Deref;

use rustc_ast::ast::{self, FnRetTy, Mutability, Term};
use rustc_span::{BytePos, Pos, Span, symbol::kw};
use tracing::debug;

use crate::comment::{combine_strs_with_missing_comments, contains_comment};
use crate::config::lists::*;
use crate::config::{IndentStyle, StyleEdition, TypeDensity};
use crate::expr::{
    ExprType, RhsAssignKind, format_expr, rewrite_assign_rhs, rewrite_call, rewrite_tuple,
    rewrite_unary_prefix,
};
use crate::lists::{
    ListFormatting, ListItem, Separator, definitive_tactic, itemize_list, write_list,
};
use crate::macros::{MacroPosition, rewrite_macro};
use crate::overflow;
use crate::pairs::{PairParts, rewrite_pair};
use crate::patterns::rewrite_range_pat;
use crate::rewrite::{Rewrite, RewriteContext, RewriteError, RewriteErrorExt, RewriteResult};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::utils::{
    colon_spaces, extra_offset, first_line_width, format_extern, format_mutability,
    last_line_extendable, last_line_width, mk_sp, rewrite_ident,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum PathContext {
    Expr,
    Type,
    Import,
}

// Does not wrap on simple segments.
pub(crate) fn rewrite_path(
    context: &RewriteContext<'_>,
    path_context: PathContext,
    qself: &Option<Box<ast::QSelf>>,
    path: &ast::Path,
    shape: Shape,
) -> RewriteResult {
    let skip_count = qself.as_ref().map_or(0, |x| x.position);

    // 32 covers almost all path lengths measured when compiling core, and there isn't a big
    // downside from allocating slightly more than necessary.
    let mut result = String::with_capacity(32);

    if path.is_global() && qself.is_none() && path_context != PathContext::Import {
        result.push_str("::");
    }

    let mut span_lo = path.span.lo();

    if let Some(qself) = qself {
        result.push('<');

        let fmt_ty = qself.ty.rewrite_result(context, shape)?;
        result.push_str(&fmt_ty);

        if skip_count > 0 {
            result.push_str(" as ");
            if path.is_global() && path_context != PathContext::Import {
                result.push_str("::");
            }

            // 3 = ">::".len()
            let shape = shape.sub_width(3).max_width_error(shape.width, path.span)?;

            result = rewrite_path_segments(
                PathContext::Type,
                result,
                path.segments.iter().take(skip_count),
                span_lo,
                path.span.hi(),
                context,
                shape,
            )?;
        }

        result.push_str(">::");
        span_lo = qself.ty.span.hi() + BytePos(1);
    }

    rewrite_path_segments(
        path_context,
        result,
        path.segments.iter().skip(skip_count),
        span_lo,
        path.span.hi(),
        context,
        shape,
    )
}

fn rewrite_path_segments<'a, I>(
    path_context: PathContext,
    mut buffer: String,
    iter: I,
    mut span_lo: BytePos,
    span_hi: BytePos,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult
where
    I: Iterator<Item = &'a ast::PathSegment>,
{
    let mut first = true;
    let shape = shape.visual_indent(0);

    for segment in iter {
        // Indicates a global path, shouldn't be rendered.
        if segment.ident.name == kw::PathRoot {
            continue;
        }
        if first {
            first = false;
        } else {
            buffer.push_str("::");
        }

        let extra_offset = extra_offset(&buffer, shape);
        let new_shape = shape
            .shrink_left(extra_offset)
            .max_width_error(shape.width, mk_sp(span_lo, span_hi))?;
        let segment_string = rewrite_segment(
            path_context,
            segment,
            &mut span_lo,
            span_hi,
            context,
            new_shape,
        )?;

        buffer.push_str(&segment_string);
    }

    Ok(buffer)
}

#[derive(Debug)]
pub(crate) enum SegmentParam<'a> {
    Const(&'a ast::AnonConst),
    LifeTime(&'a ast::Lifetime),
    Type(&'a ast::Ty),
    Binding(&'a ast::AssocItemConstraint),
}

impl<'a> SegmentParam<'a> {
    fn from_generic_arg(arg: &ast::GenericArg) -> SegmentParam<'_> {
        match arg {
            ast::GenericArg::Lifetime(ref lt) => SegmentParam::LifeTime(lt),
            ast::GenericArg::Type(ref ty) => SegmentParam::Type(ty),
            ast::GenericArg::Const(const_) => SegmentParam::Const(const_),
        }
    }
}

impl<'a> Spanned for SegmentParam<'a> {
    fn span(&self) -> Span {
        match *self {
            SegmentParam::Const(const_) => const_.value.span,
            SegmentParam::LifeTime(lt) => lt.ident.span,
            SegmentParam::Type(ty) => ty.span,
            SegmentParam::Binding(binding) => binding.span,
        }
    }
}

impl<'a> Rewrite for SegmentParam<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match *self {
            SegmentParam::Const(const_) => const_.rewrite_result(context, shape),
            SegmentParam::LifeTime(lt) => lt.rewrite_result(context, shape),
            SegmentParam::Type(ty) => ty.rewrite_result(context, shape),
            SegmentParam::Binding(atc) => atc.rewrite_result(context, shape),
        }
    }
}

impl Rewrite for ast::PreciseCapturingArg {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match self {
            ast::PreciseCapturingArg::Lifetime(lt) => lt.rewrite_result(context, shape),
            ast::PreciseCapturingArg::Arg(p, _) => {
                rewrite_path(context, PathContext::Type, &None, p, shape)
            }
        }
    }
}

impl Rewrite for ast::AssocItemConstraint {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        use ast::AssocItemConstraintKind::{Bound, Equality};

        let mut result = String::with_capacity(128);
        result.push_str(rewrite_ident(context, self.ident));

        if let Some(ref gen_args) = self.gen_args {
            let budget = shape
                .width
                .checked_sub(result.len())
                .max_width_error(shape.width, self.span)?;
            let shape = Shape::legacy(budget, shape.indent + result.len());
            let gen_str = rewrite_generic_args(gen_args, context, shape, gen_args.span())?;
            result.push_str(&gen_str);
        }

        let infix = match (&self.kind, context.config.type_punctuation_density()) {
            (Bound { .. }, _) => ": ",
            (Equality { .. }, TypeDensity::Wide) => " = ",
            (Equality { .. }, TypeDensity::Compressed) => "=",
        };
        result.push_str(infix);

        let budget = shape
            .width
            .checked_sub(result.len())
            .max_width_error(shape.width, self.span)?;
        let shape = Shape::legacy(budget, shape.indent + result.len());
        let rewrite = self.kind.rewrite_result(context, shape)?;
        result.push_str(&rewrite);

        Ok(result)
    }
}

impl Rewrite for ast::AssocItemConstraintKind {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match self {
            ast::AssocItemConstraintKind::Equality { term } => match term {
                Term::Ty(ty) => ty.rewrite_result(context, shape),
                Term::Const(c) => c.rewrite_result(context, shape),
            },
            ast::AssocItemConstraintKind::Bound { bounds } => bounds.rewrite_result(context, shape),
        }
    }
}

// Formats a path segment. There are some hacks involved to correctly determine
// the segment's associated span since it's not part of the AST.
//
// The span_lo is assumed to be greater than the end of any previous segment's
// parameters and lesser or equal than the start of current segment.
//
// span_hi is assumed equal to the end of the entire path.
//
// When the segment contains a positive number of parameters, we update span_lo
// so that invariants described above will hold for the next segment.
fn rewrite_segment(
    path_context: PathContext,
    segment: &ast::PathSegment,
    span_lo: &mut BytePos,
    span_hi: BytePos,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    let mut result = String::with_capacity(128);
    result.push_str(rewrite_ident(context, segment.ident));

    let ident_len = result.len();
    let shape = if context.use_block_indent() {
        shape.offset_left(ident_len)
    } else {
        shape.shrink_left(ident_len)
    }
    .max_width_error(shape.width, mk_sp(*span_lo, span_hi))?;

    if let Some(ref args) = segment.args {
        let generics_str = rewrite_generic_args(args, context, shape, mk_sp(*span_lo, span_hi))?;
        match **args {
            ast::GenericArgs::AngleBracketed(ref data) if !data.args.is_empty() => {
                // HACK: squeeze out the span between the identifier and the parameters.
                // The hack is required so that we don't remove the separator inside macro calls.
                // This does not work in the presence of comment, hoping that people are
                // sane about where to put their comment.
                let separator_snippet = context
                    .snippet(mk_sp(segment.ident.span.hi(), data.span.lo()))
                    .trim();
                let force_separator = context.inside_macro() && separator_snippet.starts_with("::");
                let separator = if path_context == PathContext::Expr || force_separator {
                    "::"
                } else {
                    ""
                };
                result.push_str(separator);

                // Update position of last bracket.
                *span_lo = context
                    .snippet_provider
                    .span_after(mk_sp(*span_lo, span_hi), "<");
            }
            _ => (),
        }
        result.push_str(&generics_str)
    }

    Ok(result)
}

fn format_function_type<'a, I>(
    inputs: I,
    output: &FnRetTy,
    variadic: bool,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult
where
    I: ExactSizeIterator,
    <I as Iterator>::Item: Deref,
    <I::Item as Deref>::Target: Rewrite + Spanned + 'a,
{
    debug!("format_function_type {:#?}", shape);

    let ty_shape = match context.config.indent_style() {
        // 4 = " -> "
        IndentStyle::Block => shape.offset_left(4).max_width_error(shape.width, span)?,
        IndentStyle::Visual => shape.block_left(4).max_width_error(shape.width, span)?,
    };
    let output = match *output {
        FnRetTy::Ty(ref ty) => {
            let type_str = ty.rewrite_result(context, ty_shape)?;
            format!(" -> {type_str}")
        }
        FnRetTy::Default(..) => String::new(),
    };

    let list_shape = if context.use_block_indent() {
        Shape::indented(
            shape.block().indent.block_indent(context.config),
            context.config,
        )
    } else {
        // 2 for ()
        let budget = shape
            .width
            .checked_sub(2)
            .max_width_error(shape.width, span)?;
        // 1 for (
        let offset = shape.indent + 1;
        Shape::legacy(budget, offset)
    };

    let is_inputs_empty = inputs.len() == 0;
    let list_lo = context.snippet_provider.span_after(span, "(");
    let (list_str, tactic) = if is_inputs_empty {
        let tactic = get_tactics(&[], &output, shape);
        let list_hi = context.snippet_provider.span_before(span, ")");
        let comment = context
            .snippet_provider
            .span_to_snippet(mk_sp(list_lo, list_hi))
            .unknown_error()?
            .trim();
        let comment = if comment.starts_with("//") {
            format!(
                "{}{}{}",
                &list_shape.indent.to_string_with_newline(context.config),
                comment,
                &shape.block().indent.to_string_with_newline(context.config)
            )
        } else {
            comment.to_string()
        };
        (comment, tactic)
    } else {
        let items = itemize_list(
            context.snippet_provider,
            inputs,
            ")",
            ",",
            |arg| arg.span().lo(),
            |arg| arg.span().hi(),
            |arg| arg.rewrite_result(context, list_shape),
            list_lo,
            span.hi(),
            false,
        );

        let item_vec: Vec<_> = items.collect();
        let tactic = get_tactics(&item_vec, &output, shape);
        let trailing_separator = if !context.use_block_indent() || variadic {
            SeparatorTactic::Never
        } else {
            context.config.trailing_comma()
        };

        let fmt = ListFormatting::new(list_shape, context.config)
            .tactic(tactic)
            .trailing_separator(trailing_separator)
            .ends_with_newline(tactic.ends_with_newline(context.config.indent_style()))
            .preserve_newline(true);
        (write_list(&item_vec, &fmt)?, tactic)
    };

    let args = if tactic == DefinitiveListTactic::Horizontal
        || !context.use_block_indent()
        || is_inputs_empty
    {
        format!("({list_str})")
    } else {
        format!(
            "({}{}{})",
            list_shape.indent.to_string_with_newline(context.config),
            list_str,
            shape.block().indent.to_string_with_newline(context.config),
        )
    };
    if output.is_empty() || last_line_width(&args) + first_line_width(&output) <= shape.width {
        Ok(format!("{args}{output}"))
    } else {
        Ok(format!(
            "{}\n{}{}",
            args,
            list_shape.indent.to_string(context.config),
            output.trim_start()
        ))
    }
}

fn type_bound_colon(context: &RewriteContext<'_>) -> &'static str {
    colon_spaces(context.config)
}

// If the return type is multi-lined, then force to use multiple lines for
// arguments as well.
fn get_tactics(item_vec: &[ListItem], output: &str, shape: Shape) -> DefinitiveListTactic {
    if output.contains('\n') {
        DefinitiveListTactic::Vertical
    } else {
        definitive_tactic(
            item_vec,
            ListTactic::HorizontalVertical,
            Separator::Comma,
            // 2 is for the case of ',\n'
            shape.width.saturating_sub(2 + output.len()),
        )
    }
}

impl Rewrite for ast::WherePredicate {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let attrs_str = self.attrs.rewrite_result(context, shape)?;
        // FIXME: dead spans?
        let pred_str = &match self.kind {
            ast::WherePredicateKind::BoundPredicate(ast::WhereBoundPredicate {
                ref bound_generic_params,
                ref bounded_ty,
                ref bounds,
                ..
            }) => {
                let type_str = bounded_ty.rewrite_result(context, shape)?;
                let colon = type_bound_colon(context).trim_end();
                let lhs = if let Some(binder_str) =
                    rewrite_bound_params(context, shape, bound_generic_params)
                {
                    format!("for<{binder_str}> {type_str}{colon}")
                } else {
                    format!("{type_str}{colon}")
                };

                rewrite_assign_rhs(context, lhs, bounds, &RhsAssignKind::Bounds, shape)?
            }
            ast::WherePredicateKind::RegionPredicate(ast::WhereRegionPredicate {
                ref lifetime,
                ref bounds,
            }) => rewrite_bounded_lifetime(lifetime, bounds, self.span, context, shape)?,
            ast::WherePredicateKind::EqPredicate(ast::WhereEqPredicate {
                ref lhs_ty,
                ref rhs_ty,
                ..
            }) => {
                let lhs_ty_str = lhs_ty
                    .rewrite_result(context, shape)
                    .map(|lhs| lhs + " =")?;
                rewrite_assign_rhs(context, lhs_ty_str, &**rhs_ty, &RhsAssignKind::Ty, shape)?
            }
        };

        let mut result = String::with_capacity(attrs_str.len() + pred_str.len() + 1);
        result.push_str(&attrs_str);
        let pred_start = self.span.lo();
        let line_len = last_line_width(&attrs_str) + 1 + first_line_width(&pred_str);
        if let Some(last_attr) = self.attrs.last().filter(|last_attr| {
            contains_comment(context.snippet(mk_sp(last_attr.span.hi(), pred_start)))
        }) {
            result = combine_strs_with_missing_comments(
                context,
                &result,
                &pred_str,
                mk_sp(last_attr.span.hi(), pred_start),
                Shape {
                    width: shape.width.min(context.config.inline_attribute_width()),
                    ..shape
                },
                !last_attr.is_doc_comment(),
            )?;
        } else {
            if !self.attrs.is_empty() {
                if context.config.inline_attribute_width() < line_len
                    || self.attrs.len() > 1
                    || self.attrs.last().is_some_and(|a| a.is_doc_comment())
                {
                    result.push_str(&shape.indent.to_string_with_newline(context.config));
                } else {
                    result.push(' ');
                }
            }
            result.push_str(&pred_str);
        }

        Ok(result)
    }
}

impl Rewrite for ast::GenericArg {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match *self {
            ast::GenericArg::Lifetime(ref lt) => lt.rewrite_result(context, shape),
            ast::GenericArg::Type(ref ty) => ty.rewrite_result(context, shape),
            ast::GenericArg::Const(ref const_) => const_.rewrite_result(context, shape),
        }
    }
}

fn rewrite_generic_args(
    gen_args: &ast::GenericArgs,
    context: &RewriteContext<'_>,
    shape: Shape,
    span: Span,
) -> RewriteResult {
    match gen_args {
        ast::GenericArgs::AngleBracketed(ref data) => {
            if data.args.is_empty() {
                Ok("".to_owned())
            } else {
                let args = data
                    .args
                    .iter()
                    .map(|x| match x {
                        ast::AngleBracketedArg::Arg(generic_arg) => {
                            SegmentParam::from_generic_arg(generic_arg)
                        }
                        ast::AngleBracketedArg::Constraint(constraint) => {
                            SegmentParam::Binding(constraint)
                        }
                    })
                    .collect::<Vec<_>>();

                overflow::rewrite_with_angle_brackets(context, "", args.iter(), shape, span)
            }
        }
        ast::GenericArgs::Parenthesized(ref data) => format_function_type(
            data.inputs.iter().map(|x| &**x),
            &data.output,
            false,
            data.span,
            context,
            shape,
        ),
        ast::GenericArgs::ParenthesizedElided(..) => Ok("(..)".to_owned()),
    }
}

fn rewrite_bounded_lifetime(
    lt: &ast::Lifetime,
    bounds: &[ast::GenericBound],
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    let result = lt.rewrite_result(context, shape)?;

    if bounds.is_empty() {
        Ok(result)
    } else {
        let colon = type_bound_colon(context);
        let overhead = last_line_width(&result) + colon.len();
        let shape = shape
            .sub_width(overhead)
            .max_width_error(shape.width, span)?;
        let result = format!(
            "{}{}{}",
            result,
            colon,
            join_bounds(context, shape, bounds, true)?
        );
        Ok(result)
    }
}

impl Rewrite for ast::AnonConst {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        format_expr(&self.value, ExprType::SubExpression, context, shape)
    }
}

impl Rewrite for ast::Lifetime {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, _: Shape) -> RewriteResult {
        Ok(context.snippet(self.ident.span).to_owned())
    }
}

impl Rewrite for ast::GenericBound {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match *self {
            ast::GenericBound::Trait(ref poly_trait_ref) => {
                let snippet = context.snippet(self.span());
                let has_paren = snippet.starts_with('(') && snippet.ends_with(')');
                poly_trait_ref
                    .rewrite_result(context, shape)
                    .map(|s| if has_paren { format!("({})", s) } else { s })
            }
            ast::GenericBound::Use(ref args, span) => {
                overflow::rewrite_with_angle_brackets(context, "use", args.iter(), shape, span)
            }
            ast::GenericBound::Outlives(ref lifetime) => lifetime.rewrite_result(context, shape),
        }
    }
}

impl Rewrite for ast::GenericBounds {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        if self.is_empty() {
            return Ok(String::new());
        }

        join_bounds(context, shape, self, true)
    }
}

impl Rewrite for ast::GenericParam {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        // FIXME: If there are more than one attributes, this will force multiline.
        let mut result = self
            .attrs
            .rewrite_result(context, shape)
            .unwrap_or(String::new());
        let has_attrs = !result.is_empty();

        let mut param = String::with_capacity(128);

        let param_start = if let ast::GenericParamKind::Const {
            ref ty,
            span,
            default,
        } = &self.kind
        {
            param.push_str("const ");
            param.push_str(rewrite_ident(context, self.ident));
            param.push_str(": ");
            param.push_str(&ty.rewrite_result(context, shape)?);
            if let Some(default) = default {
                let eq_str = match context.config.type_punctuation_density() {
                    TypeDensity::Compressed => "=",
                    TypeDensity::Wide => " = ",
                };
                param.push_str(eq_str);
                let budget = shape
                    .width
                    .checked_sub(param.len())
                    .max_width_error(shape.width, self.span())?;
                let rewrite =
                    default.rewrite_result(context, Shape::legacy(budget, shape.indent))?;
                param.push_str(&rewrite);
            }
            span.lo()
        } else {
            param.push_str(rewrite_ident(context, self.ident));
            self.ident.span.lo()
        };

        if !self.bounds.is_empty() {
            param.push_str(type_bound_colon(context));
            param.push_str(&self.bounds.rewrite_result(context, shape)?)
        }
        if let ast::GenericParamKind::Type {
            default: Some(ref def),
        } = self.kind
        {
            let eq_str = match context.config.type_punctuation_density() {
                TypeDensity::Compressed => "=",
                TypeDensity::Wide => " = ",
            };
            param.push_str(eq_str);
            let budget = shape
                .width
                .checked_sub(param.len())
                .max_width_error(shape.width, self.span())?;
            let rewrite =
                def.rewrite_result(context, Shape::legacy(budget, shape.indent + param.len()))?;
            param.push_str(&rewrite);
        }

        if let Some(last_attr) = self.attrs.last().filter(|last_attr| {
            contains_comment(context.snippet(mk_sp(last_attr.span.hi(), param_start)))
        }) {
            result = combine_strs_with_missing_comments(
                context,
                &result,
                &param,
                mk_sp(last_attr.span.hi(), param_start),
                shape,
                !last_attr.is_doc_comment(),
            )?;
        } else {
            // When rewriting generic params, an extra newline should be put
            // if the attributes end with a doc comment
            if let Some(true) = self.attrs.last().map(|a| a.is_doc_comment()) {
                result.push_str(&shape.indent.to_string_with_newline(context.config));
            } else if has_attrs {
                result.push(' ');
            }
            result.push_str(&param);
        }

        Ok(result)
    }
}

impl Rewrite for ast::PolyTraitRef {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let (binder, shape) = if let Some(lifetime_str) =
            rewrite_bound_params(context, shape, &self.bound_generic_params)
        {
            // 6 is "for<> ".len()
            let extra_offset = lifetime_str.len() + 6;
            let shape = shape
                .offset_left(extra_offset)
                .max_width_error(shape.width, self.span)?;
            (format!("for<{lifetime_str}> "), shape)
        } else {
            (String::new(), shape)
        };

        let ast::TraitBoundModifiers {
            constness,
            asyncness,
            polarity,
        } = self.modifiers;
        let mut constness = constness.as_str().to_string();
        if !constness.is_empty() {
            constness.push(' ');
        }
        let mut asyncness = asyncness.as_str().to_string();
        if !asyncness.is_empty() {
            asyncness.push(' ');
        }
        let polarity = polarity.as_str();
        let shape = shape
            .offset_left(constness.len() + polarity.len())
            .max_width_error(shape.width, self.span)?;

        let path_str = self.trait_ref.rewrite_result(context, shape)?;
        Ok(format!(
            "{binder}{constness}{asyncness}{polarity}{path_str}"
        ))
    }
}

impl Rewrite for ast::TraitRef {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        rewrite_path(context, PathContext::Type, &None, &self.path, shape)
    }
}

impl Rewrite for ast::Ty {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match self.kind {
            ast::TyKind::TraitObject(ref bounds, tobj_syntax) => {
                // we have to consider 'dyn' keyword is used or not!!!
                let (shape, prefix) = match tobj_syntax {
                    ast::TraitObjectSyntax::Dyn => {
                        let shape = shape
                            .offset_left(4)
                            .max_width_error(shape.width, self.span())?;
                        (shape, "dyn ")
                    }
                    ast::TraitObjectSyntax::None => (shape, ""),
                };
                let mut res = bounds.rewrite_result(context, shape)?;
                // We may have falsely removed a trailing `+` inside macro call.
                if context.inside_macro()
                    && bounds.len() == 1
                    && context.snippet(self.span).ends_with('+')
                    && !res.ends_with('+')
                {
                    res.push('+');
                }
                Ok(format!("{prefix}{res}"))
            }
            ast::TyKind::Ptr(ref mt) => {
                let prefix = match mt.mutbl {
                    Mutability::Mut => "*mut ",
                    Mutability::Not => "*const ",
                };

                rewrite_unary_prefix(context, prefix, &*mt.ty, shape)
            }
            ast::TyKind::Ref(ref lifetime, ref mt)
            | ast::TyKind::PinnedRef(ref lifetime, ref mt) => {
                let mut_str = format_mutability(mt.mutbl);
                let mut_len = mut_str.len();
                let mut result = String::with_capacity(128);
                result.push('&');
                let ref_hi = context.snippet_provider.span_after(self.span(), "&");
                let mut cmnt_lo = ref_hi;

                if let Some(ref lifetime) = *lifetime {
                    let lt_budget = shape
                        .width
                        .checked_sub(2 + mut_len)
                        .max_width_error(shape.width, self.span())?;
                    let lt_str = lifetime.rewrite_result(
                        context,
                        Shape::legacy(lt_budget, shape.indent + 2 + mut_len),
                    )?;
                    let before_lt_span = mk_sp(cmnt_lo, lifetime.ident.span.lo());
                    if contains_comment(context.snippet(before_lt_span)) {
                        result = combine_strs_with_missing_comments(
                            context,
                            &result,
                            &lt_str,
                            before_lt_span,
                            shape,
                            true,
                        )?;
                    } else {
                        result.push_str(&lt_str);
                    }
                    result.push(' ');
                    cmnt_lo = lifetime.ident.span.hi();
                }

                if let ast::TyKind::PinnedRef(..) = self.kind {
                    result.push_str("pin ");
                    if ast::Mutability::Not == mt.mutbl {
                        result.push_str("const ");
                    }
                }

                if ast::Mutability::Mut == mt.mutbl {
                    let mut_hi = context.snippet_provider.span_after(self.span(), "mut");
                    let before_mut_span = mk_sp(cmnt_lo, mut_hi - BytePos::from_usize(3));
                    if contains_comment(context.snippet(before_mut_span)) {
                        result = combine_strs_with_missing_comments(
                            context,
                            result.trim_end(),
                            mut_str,
                            before_mut_span,
                            shape,
                            true,
                        )?;
                    } else {
                        result.push_str(mut_str);
                    }
                    cmnt_lo = mut_hi;
                }

                let before_ty_span = mk_sp(cmnt_lo, mt.ty.span.lo());
                if contains_comment(context.snippet(before_ty_span)) {
                    result = combine_strs_with_missing_comments(
                        context,
                        result.trim_end(),
                        &mt.ty.rewrite_result(context, shape)?,
                        before_ty_span,
                        shape,
                        true,
                    )?;
                } else {
                    let used_width = last_line_width(&result);
                    let budget = shape
                        .width
                        .checked_sub(used_width)
                        .max_width_error(shape.width, self.span())?;
                    let ty_str = mt.ty.rewrite_result(
                        context,
                        Shape::legacy(budget, shape.indent + used_width),
                    )?;
                    result.push_str(&ty_str);
                }

                Ok(result)
            }
            // FIXME: we drop any comments here, even though it's a silly place to put
            // comments.
            ast::TyKind::Paren(ref ty) => {
                if context.config.style_edition() <= StyleEdition::Edition2021
                    || context.config.indent_style() == IndentStyle::Visual
                {
                    let budget = shape
                        .width
                        .checked_sub(2)
                        .max_width_error(shape.width, self.span())?;
                    return ty
                        .rewrite_result(context, Shape::legacy(budget, shape.indent + 1))
                        .map(|ty_str| format!("({})", ty_str));
                }

                // 2 = ()
                if let Some(sh) = shape.sub_width(2) {
                    if let Ok(ref s) = ty.rewrite_result(context, sh) {
                        if !s.contains('\n') {
                            return Ok(format!("({s})"));
                        }
                    }
                }

                let indent_str = shape.indent.to_string_with_newline(context.config);
                let shape = shape
                    .block_indent(context.config.tab_spaces())
                    .with_max_width(context.config);
                let rw = ty.rewrite_result(context, shape)?;
                Ok(format!(
                    "({}{}{})",
                    shape.to_string_with_newline(context.config),
                    rw,
                    indent_str
                ))
            }
            ast::TyKind::Slice(ref ty) => {
                let budget = shape
                    .width
                    .checked_sub(4)
                    .max_width_error(shape.width, self.span())?;
                ty.rewrite_result(context, Shape::legacy(budget, shape.indent + 1))
                    .map(|ty_str| format!("[{}]", ty_str))
            }
            ast::TyKind::Tup(ref items) => {
                rewrite_tuple(context, items.iter(), self.span, shape, items.len() == 1)
            }
            ast::TyKind::Path(ref q_self, ref path) => {
                rewrite_path(context, PathContext::Type, q_self, path, shape)
            }
            ast::TyKind::Array(ref ty, ref repeats) => rewrite_pair(
                &**ty,
                &*repeats.value,
                PairParts::new("[", "; ", "]"),
                context,
                shape,
                SeparatorPlace::Back,
            ),
            ast::TyKind::Infer => {
                if shape.width >= 1 {
                    Ok("_".to_owned())
                } else {
                    Err(RewriteError::ExceedsMaxWidth {
                        configured_width: shape.width,
                        span: self.span(),
                    })
                }
            }
            ast::TyKind::FnPtr(ref fn_ptr) => rewrite_fn_ptr(fn_ptr, self.span, context, shape),
            ast::TyKind::Never => Ok(String::from("!")),
            ast::TyKind::MacCall(ref mac) => {
                rewrite_macro(mac, context, shape, MacroPosition::Expression)
            }
            ast::TyKind::ImplicitSelf => Ok(String::from("")),
            ast::TyKind::ImplTrait(_, ref it) => {
                // Empty trait is not a parser error.
                if it.is_empty() {
                    return Ok("impl".to_owned());
                }
                let rw = if context.config.style_edition() <= StyleEdition::Edition2021 {
                    it.rewrite_result(context, shape)
                } else {
                    join_bounds(context, shape, it, false)
                };
                rw.map(|it_str| {
                    let space = if it_str.is_empty() { "" } else { " " };
                    format!("impl{}{}", space, it_str)
                })
            }
            ast::TyKind::CVarArgs => Ok("...".to_owned()),
            ast::TyKind::Dummy | ast::TyKind::Err(_) => Ok(context.snippet(self.span).to_owned()),
            ast::TyKind::Typeof(ref anon_const) => rewrite_call(
                context,
                "typeof",
                &[anon_const.value.clone()],
                self.span,
                shape,
            ),
            ast::TyKind::Pat(ref ty, ref pat) => {
                let ty = ty.rewrite_result(context, shape)?;
                let pat = pat.rewrite_result(context, shape)?;
                Ok(format!("{ty} is {pat}"))
            }
            ast::TyKind::UnsafeBinder(ref binder) => {
                let mut result = String::new();
                if binder.generic_params.is_empty() {
                    // We always want to write `unsafe<>` since `unsafe<> Ty`
                    // and `Ty` are distinct types.
                    result.push_str("unsafe<> ")
                } else if let Some(ref lifetime_str) =
                    rewrite_bound_params(context, shape, &binder.generic_params)
                {
                    result.push_str("unsafe<");
                    result.push_str(lifetime_str);
                    result.push_str("> ");
                }

                let inner_ty_shape = if context.use_block_indent() {
                    shape
                        .offset_left(result.len())
                        .max_width_error(shape.width, self.span())?
                } else {
                    shape
                        .visual_indent(result.len())
                        .sub_width(result.len())
                        .max_width_error(shape.width, self.span())?
                };

                let rewrite = binder.inner_ty.rewrite_result(context, inner_ty_shape)?;
                result.push_str(&rewrite);
                Ok(result)
            }
        }
    }
}

impl Rewrite for ast::TyPat {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match self.kind {
            ast::TyPatKind::Range(ref lhs, ref rhs, ref end_kind) => {
                rewrite_range_pat(context, shape, lhs, rhs, end_kind, self.span)
            }
            ast::TyPatKind::Or(ref variants) => {
                let mut first = true;
                let mut s = String::new();
                for variant in variants {
                    if first {
                        first = false
                    } else {
                        s.push_str(" | ");
                    }
                    s.push_str(&variant.rewrite_result(context, shape)?);
                }
                Ok(s)
            }
            ast::TyPatKind::Err(_) => Err(RewriteError::Unknown),
        }
    }
}

fn rewrite_fn_ptr(
    fn_ptr: &ast::FnPtrTy,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    debug!("rewrite_bare_fn {:#?}", shape);

    let mut result = String::with_capacity(128);

    if let Some(ref lifetime_str) = rewrite_bound_params(context, shape, &fn_ptr.generic_params) {
        result.push_str("for<");
        // 6 = "for<> ".len(), 4 = "for<".
        // This doesn't work out so nicely for multiline situation with lots of
        // rightward drift. If that is a problem, we could use the list stuff.
        result.push_str(lifetime_str);
        result.push_str("> ");
    }

    result.push_str(crate::utils::format_safety(fn_ptr.safety));

    result.push_str(&format_extern(
        fn_ptr.ext,
        context.config.force_explicit_abi(),
    ));

    result.push_str("fn");

    let func_ty_shape = if context.use_block_indent() {
        shape
            .offset_left(result.len())
            .max_width_error(shape.width, span)?
    } else {
        shape
            .visual_indent(result.len())
            .sub_width(result.len())
            .max_width_error(shape.width, span)?
    };

    let rewrite = format_function_type(
        fn_ptr.decl.inputs.iter(),
        &fn_ptr.decl.output,
        fn_ptr.decl.c_variadic(),
        span,
        context,
        func_ty_shape,
    )?;

    result.push_str(&rewrite);

    Ok(result)
}

fn is_generic_bounds_in_order(generic_bounds: &[ast::GenericBound]) -> bool {
    let is_trait = |b: &ast::GenericBound| match b {
        ast::GenericBound::Outlives(..) => false,
        ast::GenericBound::Trait(..) | ast::GenericBound::Use(..) => true,
    };
    let is_lifetime = |b: &ast::GenericBound| !is_trait(b);
    let last_trait_index = generic_bounds.iter().rposition(is_trait);
    let first_lifetime_index = generic_bounds.iter().position(is_lifetime);
    match (last_trait_index, first_lifetime_index) {
        (Some(last_trait_index), Some(first_lifetime_index)) => {
            last_trait_index < first_lifetime_index
        }
        _ => true,
    }
}

fn join_bounds(
    context: &RewriteContext<'_>,
    shape: Shape,
    items: &[ast::GenericBound],
    need_indent: bool,
) -> RewriteResult {
    join_bounds_inner(context, shape, items, need_indent, false)
}

fn join_bounds_inner(
    context: &RewriteContext<'_>,
    shape: Shape,
    items: &[ast::GenericBound],
    need_indent: bool,
    force_newline: bool,
) -> RewriteResult {
    debug_assert!(!items.is_empty());

    let generic_bounds_in_order = is_generic_bounds_in_order(items);
    let is_bound_extendable = |s: &str, b: &ast::GenericBound| match b {
        ast::GenericBound::Outlives(..) => true,
        // We treat `use<>` like a trait bound here.
        ast::GenericBound::Trait(..) | ast::GenericBound::Use(..) => last_line_extendable(s),
    };

    // Whether a GenericBound item is a PathSegment segment that includes internal array
    // that contains more than one item
    let is_item_with_multi_items_array = |item: &ast::GenericBound| match item {
        ast::GenericBound::Trait(ref poly_trait_ref, ..) => {
            let segments = &poly_trait_ref.trait_ref.path.segments;
            if segments.len() > 1 {
                true
            } else {
                if let Some(args_in) = &segments[0].args {
                    matches!(
                        args_in.deref(),
                        ast::GenericArgs::AngleBracketed(bracket_args)
                            if bracket_args.args.len() > 1
                    )
                } else {
                    false
                }
            }
        }
        ast::GenericBound::Use(args, _) => args.len() > 1,
        _ => false,
    };

    let result = items.iter().enumerate().try_fold(
        (String::new(), None, false),
        |(strs, prev_trailing_span, prev_extendable), (i, item)| {
            let trailing_span = if i < items.len() - 1 {
                let hi = context
                    .snippet_provider
                    .span_before(mk_sp(items[i + 1].span().lo(), item.span().hi()), "+");

                Some(mk_sp(item.span().hi(), hi))
            } else {
                None
            };
            let (leading_span, has_leading_comment) = if i > 0 {
                let lo = context
                    .snippet_provider
                    .span_after(mk_sp(items[i - 1].span().hi(), item.span().lo()), "+");

                let span = mk_sp(lo, item.span().lo());

                let has_comments = contains_comment(context.snippet(span));

                (Some(mk_sp(lo, item.span().lo())), has_comments)
            } else {
                (None, false)
            };
            let prev_has_trailing_comment = match prev_trailing_span {
                Some(ts) => contains_comment(context.snippet(ts)),
                _ => false,
            };

            let shape = if need_indent && force_newline {
                shape
                    .block_indent(context.config.tab_spaces())
                    .with_max_width(context.config)
            } else {
                shape
            };
            let whitespace = if force_newline && (!prev_extendable || !generic_bounds_in_order) {
                shape
                    .indent
                    .to_string_with_newline(context.config)
                    .to_string()
            } else {
                String::from(" ")
            };

            let joiner = match context.config.type_punctuation_density() {
                TypeDensity::Compressed => String::from("+"),
                TypeDensity::Wide => whitespace + "+ ",
            };
            let joiner = if has_leading_comment {
                joiner.trim_end()
            } else {
                &joiner
            };
            let joiner = if prev_has_trailing_comment {
                joiner.trim_start()
            } else {
                joiner
            };

            let (extendable, trailing_str) = if i == 0 {
                let bound_str = item.rewrite_result(context, shape)?;
                (is_bound_extendable(&bound_str, item), bound_str)
            } else {
                let bound_str = &item.rewrite_result(context, shape)?;
                match leading_span {
                    Some(ls) if has_leading_comment => (
                        is_bound_extendable(bound_str, item),
                        combine_strs_with_missing_comments(
                            context, joiner, bound_str, ls, shape, true,
                        )?,
                    ),
                    _ => (
                        is_bound_extendable(bound_str, item),
                        String::from(joiner) + bound_str,
                    ),
                }
            };
            match prev_trailing_span {
                Some(ts) if prev_has_trailing_comment => combine_strs_with_missing_comments(
                    context,
                    &strs,
                    &trailing_str,
                    ts,
                    shape,
                    true,
                )
                .map(|v| (v, trailing_span, extendable)),
                _ => Ok((strs + &trailing_str, trailing_span, extendable)),
            }
        },
    )?;

    // Whether to retry with a forced newline:
    //   Only if result is not already multiline and did not exceed line width,
    //   and either there is more than one item;
    //       or the single item is of type `Trait`,
    //          and any of the internal arrays contains more than one item;
    let retry_with_force_newline = match context.config.style_edition() {
        style_edition @ _ if style_edition <= StyleEdition::Edition2021 => {
            !force_newline
                && items.len() > 1
                && (result.0.contains('\n') || result.0.len() > shape.width)
        }
        _ if force_newline => false,
        _ if (!result.0.contains('\n') && result.0.len() <= shape.width) => false,
        _ if items.len() > 1 => true,
        _ => is_item_with_multi_items_array(&items[0]),
    };

    if retry_with_force_newline {
        join_bounds_inner(context, shape, items, need_indent, true)
    } else {
        Ok(result.0)
    }
}

pub(crate) fn opaque_ty(ty: &Option<Box<ast::Ty>>) -> Option<&ast::GenericBounds> {
    ty.as_ref().and_then(|t| match &t.kind {
        ast::TyKind::ImplTrait(_, bounds) => Some(bounds),
        _ => None,
    })
}

pub(crate) fn can_be_overflowed_type(
    context: &RewriteContext<'_>,
    ty: &ast::Ty,
    len: usize,
) -> bool {
    match ty.kind {
        ast::TyKind::Tup(..) => context.use_block_indent() && len == 1,
        ast::TyKind::Ref(_, ref mutty)
        | ast::TyKind::PinnedRef(_, ref mutty)
        | ast::TyKind::Ptr(ref mutty) => can_be_overflowed_type(context, &*mutty.ty, len),
        _ => false,
    }
}

/// Returns `None` if there is no `GenericParam` in the list
pub(crate) fn rewrite_bound_params(
    context: &RewriteContext<'_>,
    shape: Shape,
    generic_params: &[ast::GenericParam],
) -> Option<String> {
    let result = generic_params
        .iter()
        .map(|param| param.rewrite(context, shape))
        .collect::<Option<Vec<_>>>()?
        .join(", ");
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
