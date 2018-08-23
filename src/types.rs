// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::ExactSizeIterator;
use std::ops::Deref;

use config::lists::*;
use syntax::ast::{self, FunctionRetTy, Mutability};
use syntax::source_map::{self, BytePos, Span};
use syntax::symbol::keywords;

use config::{IndentStyle, TypeDensity};
use expr::{rewrite_assign_rhs, rewrite_tuple, rewrite_unary_prefix, ToExpr};
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, Separator};
use macros::{rewrite_macro, MacroPosition};
use overflow;
use pairs::{rewrite_pair, PairParts};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use source_map::SpanUtils;
use spanned::Spanned;
use utils::{
    colon_spaces, extra_offset, first_line_width, format_abi, format_mutability,
    last_line_extendable, last_line_width, mk_sp, rewrite_ident,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PathContext {
    Expr,
    Type,
    Import,
}

// Does not wrap on simple segments.
pub fn rewrite_path(
    context: &RewriteContext,
    path_context: PathContext,
    qself: Option<&ast::QSelf>,
    path: &ast::Path,
    shape: Shape,
) -> Option<String> {
    let skip_count = qself.map_or(0, |x| x.position);

    let mut result = if path.is_global() && qself.is_none() && path_context != PathContext::Import {
        "::".to_owned()
    } else {
        String::new()
    };

    let mut span_lo = path.span.lo();

    if let Some(qself) = qself {
        result.push('<');

        let fmt_ty = qself.ty.rewrite(context, shape)?;
        result.push_str(&fmt_ty);

        if skip_count > 0 {
            result.push_str(" as ");
            if path.is_global() && path_context != PathContext::Import {
                result.push_str("::");
            }

            // 3 = ">::".len()
            let shape = shape.sub_width(3)?;

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
    context: &RewriteContext,
    shape: Shape,
) -> Option<String>
where
    I: Iterator<Item = &'a ast::PathSegment>,
{
    let mut first = true;
    let shape = shape.visual_indent(0);

    for segment in iter {
        // Indicates a global path, shouldn't be rendered.
        if segment.ident.name == keywords::CrateRoot.name() {
            continue;
        }
        if first {
            first = false;
        } else {
            buffer.push_str("::");
        }

        let extra_offset = extra_offset(&buffer, shape);
        let new_shape = shape.shrink_left(extra_offset)?;
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

    Some(buffer)
}

#[derive(Debug)]
enum SegmentParam<'a> {
    LifeTime(&'a ast::Lifetime),
    Type(&'a ast::Ty),
    Binding(&'a ast::TypeBinding),
}

impl<'a> SegmentParam<'a> {
    fn from_generic_arg(arg: &ast::GenericArg) -> SegmentParam {
        match arg {
            ast::GenericArg::Lifetime(ref lt) => SegmentParam::LifeTime(lt),
            ast::GenericArg::Type(ref ty) => SegmentParam::Type(ty),
        }
    }
}

impl<'a> Spanned for SegmentParam<'a> {
    fn span(&self) -> Span {
        match *self {
            SegmentParam::LifeTime(lt) => lt.ident.span,
            SegmentParam::Type(ty) => ty.span,
            SegmentParam::Binding(binding) => binding.span,
        }
    }
}

impl<'a> ToExpr for SegmentParam<'a> {
    fn to_expr(&self) -> Option<&ast::Expr> {
        None
    }

    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool {
        match *self {
            SegmentParam::Type(ty) => ty.can_be_overflowed(context, len),
            _ => false,
        }
    }
}

impl<'a> Rewrite for SegmentParam<'a> {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            SegmentParam::LifeTime(lt) => lt.rewrite(context, shape),
            SegmentParam::Type(ty) => ty.rewrite(context, shape),
            SegmentParam::Binding(binding) => {
                let mut result = match context.config.type_punctuation_density() {
                    TypeDensity::Wide => format!("{} = ", rewrite_ident(context, binding.ident)),
                    TypeDensity::Compressed => {
                        format!("{}=", rewrite_ident(context, binding.ident))
                    }
                };
                let budget = shape.width.checked_sub(result.len())?;
                let rewrite = binding
                    .ty
                    .rewrite(context, Shape::legacy(budget, shape.indent + result.len()))?;
                result.push_str(&rewrite);
                Some(result)
            }
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
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    result.push_str(rewrite_ident(context, segment.ident));

    let ident_len = result.len();
    let shape = if context.use_block_indent() {
        shape.offset_left(ident_len)?
    } else {
        shape.shrink_left(ident_len)?
    };

    if let Some(ref args) = segment.args {
        match **args {
            ast::GenericArgs::AngleBracketed(ref data)
                if !data.args.is_empty() || !data.bindings.is_empty() =>
            {
                let param_list = data
                    .args
                    .iter()
                    .map(SegmentParam::from_generic_arg)
                    .chain(data.bindings.iter().map(|x| SegmentParam::Binding(&*x)))
                    .collect::<Vec<_>>();

                let separator = if path_context == PathContext::Expr {
                    "::"
                } else {
                    ""
                };
                result.push_str(separator);

                let generics_str = overflow::rewrite_with_angle_brackets(
                    context,
                    "",
                    &param_list.iter().map(|e| &*e).collect::<Vec<_>>(),
                    shape,
                    mk_sp(*span_lo, span_hi),
                )?;

                // Update position of last bracket.
                *span_lo = context
                    .snippet_provider
                    .span_after(mk_sp(*span_lo, span_hi), "<");

                result.push_str(&generics_str)
            }
            ast::GenericArgs::Parenthesized(ref data) => {
                let output = match data.output {
                    Some(ref ty) => FunctionRetTy::Ty(ty.clone()),
                    None => FunctionRetTy::Default(source_map::DUMMY_SP),
                };
                result.push_str(&format_function_type(
                    data.inputs.iter().map(|x| &**x),
                    &output,
                    false,
                    data.span,
                    context,
                    shape,
                )?);
            }
            _ => (),
        }
    }

    Some(result)
}

fn format_function_type<'a, I>(
    inputs: I,
    output: &FunctionRetTy,
    variadic: bool,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String>
where
    I: ExactSizeIterator,
    <I as Iterator>::Item: Deref,
    <I::Item as Deref>::Target: Rewrite + Spanned + 'a,
{
    // Code for handling variadics is somewhat duplicated for items, but they
    // are different enough to need some serious refactoring to share code.
    enum ArgumentKind<T>
    where
        T: Deref,
        <T as Deref>::Target: Rewrite + Spanned,
    {
        Regular(T),
        Variadic(BytePos),
    }

    let variadic_arg = if variadic {
        let variadic_start = context.snippet_provider.span_before(span, "...");
        Some(ArgumentKind::Variadic(variadic_start))
    } else {
        None
    };

    // 2 for ()
    let budget = shape.width.checked_sub(2)?;
    // 1 for (
    let offset = match context.config.indent_style() {
        IndentStyle::Block => {
            shape
                .block()
                .block_indent(context.config.tab_spaces())
                .indent
        }
        IndentStyle::Visual => shape.indent + 1,
    };
    let list_shape = Shape::legacy(budget, offset);
    let list_lo = context.snippet_provider.span_after(span, "(");
    let items = itemize_list(
        context.snippet_provider,
        inputs.map(ArgumentKind::Regular).chain(variadic_arg),
        ")",
        ",",
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.span().lo(),
            ArgumentKind::Variadic(start) => start,
        },
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.span().hi(),
            ArgumentKind::Variadic(start) => start + BytePos(3),
        },
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.rewrite(context, list_shape),
            ArgumentKind::Variadic(_) => Some("...".to_owned()),
        },
        list_lo,
        span.hi(),
        false,
    );

    let item_vec: Vec<_> = items.collect();

    let tactic = definitive_tactic(
        &*item_vec,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        budget,
    );
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
    let list_str = write_list(&item_vec, &fmt)?;

    let ty_shape = match context.config.indent_style() {
        // 4 = " -> "
        IndentStyle::Block => shape.offset_left(4)?,
        IndentStyle::Visual => shape.block_left(4)?,
    };
    let output = match *output {
        FunctionRetTy::Ty(ref ty) => {
            let type_str = ty.rewrite(context, ty_shape)?;
            format!(" -> {}", type_str)
        }
        FunctionRetTy::Default(..) => String::new(),
    };

    let args = if (!list_str.contains('\n') || list_str.is_empty()) && !output.contains('\n')
        || !context.use_block_indent()
    {
        format!("({})", list_str)
    } else {
        format!(
            "({}{}{})",
            offset.to_string_with_newline(context.config),
            list_str,
            shape.block().indent.to_string_with_newline(context.config),
        )
    };
    if last_line_width(&args) + first_line_width(&output) <= shape.width {
        Some(format!("{}{}", args, output))
    } else {
        Some(format!(
            "{}\n{}{}",
            args,
            offset.to_string(context.config),
            output.trim_left()
        ))
    }
}

fn type_bound_colon(context: &RewriteContext) -> &'static str {
    colon_spaces(
        context.config.space_before_colon(),
        context.config.space_after_colon(),
    )
}

impl Rewrite for ast::WherePredicate {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        // FIXME: dead spans?
        let result = match *self {
            ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                ref bound_generic_params,
                ref bounded_ty,
                ref bounds,
                ..
            }) => {
                let type_str = bounded_ty.rewrite(context, shape)?;
                let colon = type_bound_colon(context).trim_right();
                let lhs = if let Some(lifetime_str) =
                    rewrite_lifetime_param(context, shape, bound_generic_params)
                {
                    format!("for<{}> {}{}", lifetime_str, type_str, colon)
                } else {
                    format!("{}{}", type_str, colon)
                };

                rewrite_assign_rhs(context, lhs, bounds, shape)?
            }
            ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                ref lifetime,
                ref bounds,
                ..
            }) => rewrite_bounded_lifetime(lifetime, bounds, context, shape)?,
            ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                ref lhs_ty,
                ref rhs_ty,
                ..
            }) => {
                let lhs_ty_str = lhs_ty.rewrite(context, shape).map(|lhs| lhs + " =")?;
                rewrite_assign_rhs(context, lhs_ty_str, &**rhs_ty, shape)?
            }
        };

        Some(result)
    }
}

impl Rewrite for ast::GenericArg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            ast::GenericArg::Lifetime(ref lt) => lt.rewrite(context, shape),
            ast::GenericArg::Type(ref ty) => ty.rewrite(context, shape),
        }
    }
}

fn rewrite_bounded_lifetime(
    lt: &ast::Lifetime,
    bounds: &[ast::GenericBound],
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let result = lt.rewrite(context, shape)?;

    if bounds.is_empty() {
        Some(result)
    } else {
        let colon = type_bound_colon(context);
        let overhead = last_line_width(&result) + colon.len();
        let result = format!(
            "{}{}{}",
            result,
            colon,
            join_bounds(context, shape.sub_width(overhead)?, bounds, true)?
        );
        Some(result)
    }
}

impl Rewrite for ast::Lifetime {
    fn rewrite(&self, context: &RewriteContext, _: Shape) -> Option<String> {
        Some(rewrite_ident(context, self.ident).to_owned())
    }
}

impl Rewrite for ast::GenericBound {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            ast::GenericBound::Trait(ref poly_trait_ref, trait_bound_modifier) => {
                match trait_bound_modifier {
                    ast::TraitBoundModifier::None => poly_trait_ref.rewrite(context, shape),
                    ast::TraitBoundModifier::Maybe => {
                        let rw = poly_trait_ref.rewrite(context, shape.offset_left(1)?)?;
                        Some(format!("?{}", rw))
                    }
                }
            }
            ast::GenericBound::Outlives(ref lifetime) => lifetime.rewrite(context, shape),
        }
    }
}

impl Rewrite for ast::GenericBounds {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if self.is_empty() {
            return Some(String::new());
        }

        let span = mk_sp(self.get(0)?.span().lo(), self.last()?.span().hi());
        let has_paren = context.snippet(span).starts_with("(");
        let bounds_shape = if has_paren {
            shape.offset_left(1)?.sub_width(1)?
        } else {
            shape
        };
        join_bounds(context, bounds_shape, self, true).map(|s| {
            if has_paren {
                format!("({})", s)
            } else {
                s
            }
        })
    }
}

impl Rewrite for ast::GenericParam {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let mut result = String::with_capacity(128);
        // FIXME: If there are more than one attributes, this will force multiline.
        match self.attrs.rewrite(context, shape) {
            Some(ref rw) if !rw.is_empty() => result.push_str(&format!("{} ", rw)),
            _ => (),
        }
        result.push_str(rewrite_ident(context, self.ident));
        if !self.bounds.is_empty() {
            result.push_str(type_bound_colon(context));
            result.push_str(&self.bounds.rewrite(context, shape)?)
        }
        if let ast::GenericParamKind::Type {
            default: Some(ref def),
        } = self.kind
        {
            let eq_str = match context.config.type_punctuation_density() {
                TypeDensity::Compressed => "=",
                TypeDensity::Wide => " = ",
            };
            result.push_str(eq_str);
            let budget = shape.width.checked_sub(result.len())?;
            let rewrite =
                def.rewrite(context, Shape::legacy(budget, shape.indent + result.len()))?;
            result.push_str(&rewrite);
        }

        Some(result)
    }
}

impl Rewrite for ast::PolyTraitRef {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if let Some(lifetime_str) =
            rewrite_lifetime_param(context, shape, &self.bound_generic_params)
        {
            // 6 is "for<> ".len()
            let extra_offset = lifetime_str.len() + 6;
            let path_str = self
                .trait_ref
                .rewrite(context, shape.offset_left(extra_offset)?)?;

            Some(format!("for<{}> {}", lifetime_str, path_str))
        } else {
            self.trait_ref.rewrite(context, shape)
        }
    }
}

impl Rewrite for ast::TraitRef {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        rewrite_path(context, PathContext::Type, None, &self.path, shape)
    }
}

impl Rewrite for ast::Ty {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self.node {
            ast::TyKind::TraitObject(ref bounds, tobj_syntax) => {
                // we have to consider 'dyn' keyword is used or not!!!
                let is_dyn = tobj_syntax == ast::TraitObjectSyntax::Dyn;
                // 4 is length of 'dyn '
                let shape = if is_dyn { shape.offset_left(4)? } else { shape };
                let res = bounds.rewrite(context, shape)?;
                if is_dyn {
                    Some(format!("dyn {}", res))
                } else {
                    Some(res)
                }
            }
            ast::TyKind::Ptr(ref mt) => {
                let prefix = match mt.mutbl {
                    Mutability::Mutable => "*mut ",
                    Mutability::Immutable => "*const ",
                };

                rewrite_unary_prefix(context, prefix, &*mt.ty, shape)
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                let mut_str = format_mutability(mt.mutbl);
                let mut_len = mut_str.len();
                Some(match *lifetime {
                    Some(ref lifetime) => {
                        let lt_budget = shape.width.checked_sub(2 + mut_len)?;
                        let lt_str = lifetime.rewrite(
                            context,
                            Shape::legacy(lt_budget, shape.indent + 2 + mut_len),
                        )?;
                        let lt_len = lt_str.len();
                        let budget = shape.width.checked_sub(2 + mut_len + lt_len)?;
                        format!(
                            "&{} {}{}",
                            lt_str,
                            mut_str,
                            mt.ty.rewrite(
                                context,
                                Shape::legacy(budget, shape.indent + 2 + mut_len + lt_len)
                            )?
                        )
                    }
                    None => {
                        let budget = shape.width.checked_sub(1 + mut_len)?;
                        format!(
                            "&{}{}",
                            mut_str,
                            mt.ty.rewrite(
                                context,
                                Shape::legacy(budget, shape.indent + 1 + mut_len)
                            )?
                        )
                    }
                })
            }
            // FIXME: we drop any comments here, even though it's a silly place to put
            // comments.
            ast::TyKind::Paren(ref ty) => {
                let budget = shape.width.checked_sub(2)?;
                ty.rewrite(context, Shape::legacy(budget, shape.indent + 1))
                    .map(|ty_str| format!("({})", ty_str))
            }
            ast::TyKind::Slice(ref ty) => {
                let budget = shape.width.checked_sub(4)?;
                ty.rewrite(context, Shape::legacy(budget, shape.indent + 1))
                    .map(|ty_str| format!("[{}]", ty_str))
            }
            ast::TyKind::Tup(ref items) => rewrite_tuple(
                context,
                &::utils::ptr_vec_to_ref_vec(items),
                self.span,
                shape,
            ),
            ast::TyKind::Path(ref q_self, ref path) => {
                rewrite_path(context, PathContext::Type, q_self.as_ref(), path, shape)
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
                    Some("_".to_owned())
                } else {
                    None
                }
            }
            ast::TyKind::BareFn(ref bare_fn) => rewrite_bare_fn(bare_fn, self.span, context, shape),
            ast::TyKind::Never => Some(String::from("!")),
            ast::TyKind::Mac(ref mac) => {
                rewrite_macro(mac, None, context, shape, MacroPosition::Expression)
            }
            ast::TyKind::ImplicitSelf => Some(String::from("")),
            ast::TyKind::ImplTrait(_, ref it) => it
                .rewrite(context, shape)
                .map(|it_str| format!("impl {}", it_str)),
            ast::TyKind::Err | ast::TyKind::Typeof(..) => unreachable!(),
        }
    }
}

fn rewrite_bare_fn(
    bare_fn: &ast::BareFnTy,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let mut result = String::with_capacity(128);

    if let Some(ref lifetime_str) = rewrite_lifetime_param(context, shape, &bare_fn.generic_params)
    {
        result.push_str("for<");
        // 6 = "for<> ".len(), 4 = "for<".
        // This doesn't work out so nicely for multiline situation with lots of
        // rightward drift. If that is a problem, we could use the list stuff.
        result.push_str(lifetime_str);
        result.push_str("> ");
    }

    result.push_str(::utils::format_unsafety(bare_fn.unsafety));

    result.push_str(&format_abi(
        bare_fn.abi,
        context.config.force_explicit_abi(),
        false,
    ));

    result.push_str("fn");

    let func_ty_shape = shape.offset_left(result.len())?;

    let rewrite = format_function_type(
        bare_fn.decl.inputs.iter(),
        &bare_fn.decl.output,
        bare_fn.decl.variadic,
        span,
        context,
        func_ty_shape,
    )?;

    result.push_str(&rewrite);

    Some(result)
}

fn is_generic_bounds_in_order(generic_bounds: &[ast::GenericBound]) -> bool {
    let is_trait = |b: &ast::GenericBound| match b {
        ast::GenericBound::Outlives(..) => false,
        ast::GenericBound::Trait(..) => true,
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
    context: &RewriteContext,
    shape: Shape,
    items: &[ast::GenericBound],
    need_indent: bool,
) -> Option<String> {
    // Try to join types in a single line
    let joiner = match context.config.type_punctuation_density() {
        TypeDensity::Compressed => "+",
        TypeDensity::Wide => " + ",
    };
    let type_strs = items
        .iter()
        .map(|item| item.rewrite(context, shape))
        .collect::<Option<Vec<_>>>()?;
    let result = type_strs.join(joiner);
    if items.len() <= 1 || (!result.contains('\n') && result.len() <= shape.width) {
        return Some(result);
    }

    // We need to use multiple lines.
    let (type_strs, offset) = if need_indent {
        // Rewrite with additional indentation.
        let nested_shape = shape.block_indent(context.config.tab_spaces());
        let type_strs = items
            .iter()
            .map(|item| item.rewrite(context, nested_shape))
            .collect::<Option<Vec<_>>>()?;
        (type_strs, nested_shape.indent)
    } else {
        (type_strs, shape.indent)
    };

    let is_bound_extendable = |s: &str, b: &ast::GenericBound| match b {
        ast::GenericBound::Outlives(..) => true,
        ast::GenericBound::Trait(..) => last_line_extendable(s),
    };
    let mut result = String::with_capacity(128);
    result.push_str(&type_strs[0]);
    let mut can_be_put_on_the_same_line = is_bound_extendable(&result, &items[0]);
    let generic_bounds_in_order = is_generic_bounds_in_order(items);
    for (bound, bound_str) in items[1..].iter().zip(type_strs[1..].iter()) {
        if generic_bounds_in_order && can_be_put_on_the_same_line {
            result.push_str(joiner);
        } else {
            result.push_str(&offset.to_string_with_newline(context.config));
            result.push_str("+ ");
        }
        result.push_str(bound_str);
        can_be_put_on_the_same_line = is_bound_extendable(bound_str, bound);
    }

    Some(result)
}

pub fn can_be_overflowed_type(context: &RewriteContext, ty: &ast::Ty, len: usize) -> bool {
    match ty.node {
        ast::TyKind::Tup(..) => context.use_block_indent() && len == 1,
        ast::TyKind::Rptr(_, ref mutty) | ast::TyKind::Ptr(ref mutty) => {
            can_be_overflowed_type(context, &*mutty.ty, len)
        }
        _ => false,
    }
}

/// Returns `None` if there is no `LifetimeDef` in the given generic parameters.
fn rewrite_lifetime_param(
    context: &RewriteContext,
    shape: Shape,
    generic_params: &[ast::GenericParam],
) -> Option<String> {
    let result = generic_params
        .iter()
        .filter(|p| match p.kind {
            ast::GenericParamKind::Lifetime => true,
            _ => false,
        }).map(|lt| lt.rewrite(context, shape))
        .collect::<Option<Vec<_>>>()?
        .join(", ");
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
