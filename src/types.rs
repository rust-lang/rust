// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;
use std::iter::ExactSizeIterator;

use syntax::abi;
use syntax::ast::{self, Mutability, FunctionRetTy};
use syntax::codemap::{self, Span, BytePos};
use syntax::print::pprust;
use syntax::symbol::keywords;

use {Shape, Spanned};
use codemap::SpanUtils;
use items::{format_generics_item_list, generics_shape_from_config};
use lists::{itemize_list, format_fn_args};
use rewrite::{Rewrite, RewriteContext};
use utils::{extra_offset, format_mutability, colon_spaces, wrap_str, mk_sp, last_line_width};
use expr::{rewrite_unary_prefix, rewrite_pair, rewrite_tuple_type};
use config::{Style, TypeDensity};

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

    let mut result =
        if path.is_global() && qself.is_none() && path_context != PathContext::Import {
            "::".to_owned()
        } else {
            String::new()
        };

    let mut span_lo = path.span.lo;

    if let Some(qself) = qself {
        result.push('<');
        if context.config.spaces_within_angle_brackets() {
            result.push_str(" ")
        }

        let fmt_ty = try_opt!(qself.ty.rewrite(context, shape));
        result.push_str(&fmt_ty);

        if skip_count > 0 {
            result.push_str(" as ");
            if path.is_global() && path_context != PathContext::Import {
                result.push_str("::");
            }

            let extra_offset = extra_offset(&result, shape);
            // 3 = ">::".len()
            let shape = try_opt!(try_opt!(shape.shrink_left(extra_offset)).sub_width(3));

            result = try_opt!(rewrite_path_segments(
                PathContext::Type,
                result,
                path.segments.iter().take(skip_count),
                span_lo,
                path.span.hi,
                context,
                shape,
            ));
        }

        if context.config.spaces_within_angle_brackets() {
            result.push_str(" ")
        }

        result.push_str(">::");
        span_lo = qself.ty.span.hi + BytePos(1);
    }

    rewrite_path_segments(
        path_context,
        result,
        path.segments.iter().skip(skip_count),
        span_lo,
        path.span.hi,
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
        if segment.identifier.name == keywords::CrateRoot.name() {
            continue;
        }
        if first {
            first = false;
        } else {
            buffer.push_str("::");
        }

        let extra_offset = extra_offset(&buffer, shape);
        let new_shape = try_opt!(shape.shrink_left(extra_offset));
        let segment_string = try_opt!(rewrite_segment(
            path_context,
            segment,
            &mut span_lo,
            span_hi,
            context,
            new_shape,
        ));

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
    fn get_span(&self) -> Span {
        match *self {
            SegmentParam::LifeTime(lt) => lt.span,
            SegmentParam::Type(ty) => ty.span,
            SegmentParam::Binding(binding) => binding.span,
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
                    TypeDensity::Wide => format!("{} = ", binding.ident),
                    TypeDensity::Compressed => format!("{}=", binding.ident),
                };
                let budget = try_opt!(shape.width.checked_sub(result.len()));
                let rewrite = try_opt!(binding.ty.rewrite(
                    context,
                    Shape::legacy(budget, shape.indent + result.len()),
                ));
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
    let ident_len = segment.identifier.to_string().len();
    let shape = try_opt!(shape.shrink_left(ident_len));

    let params = if let Some(ref params) = segment.parameters {
        match **params {
            ast::PathParameters::AngleBracketed(ref data)
                if !data.lifetimes.is_empty() || !data.types.is_empty() ||
                       !data.bindings.is_empty() => {
                let param_list = data.lifetimes
                    .iter()
                    .map(SegmentParam::LifeTime)
                    .chain(data.types.iter().map(|x| SegmentParam::Type(&*x)))
                    .chain(data.bindings.iter().map(|x| SegmentParam::Binding(&*x)))
                    .collect::<Vec<_>>();

                let next_span_lo = param_list.last().unwrap().get_span().hi + BytePos(1);
                let list_lo = context.codemap.span_after(mk_sp(*span_lo, span_hi), "<");
                let separator = if path_context == PathContext::Expr {
                    "::"
                } else {
                    ""
                };

                let generics_shape =
                    generics_shape_from_config(context.config, shape, separator.len());
                let items = itemize_list(
                    context.codemap,
                    param_list.into_iter(),
                    ">",
                    |param| param.get_span().lo,
                    |param| param.get_span().hi,
                    |seg| seg.rewrite(context, generics_shape),
                    list_lo,
                    span_hi,
                );
                let generics_str = try_opt!(format_generics_item_list(
                    context,
                    items,
                    generics_shape,
                    generics_shape.width,
                ));

                // Update position of last bracket.
                *span_lo = next_span_lo;

                format!("{}{}", separator, generics_str)
            }
            ast::PathParameters::Parenthesized(ref data) => {
                let output = match data.output {
                    Some(ref ty) => FunctionRetTy::Ty(ty.clone()),
                    None => FunctionRetTy::Default(codemap::DUMMY_SP),
                };
                try_opt!(format_function_type(
                    data.inputs.iter().map(|x| &**x),
                    &output,
                    false,
                    data.span,
                    context,
                    shape,
                ))
            }
            _ => String::new(),
        }
    } else {
        String::new()
    };

    Some(format!("{}{}", segment.identifier, params))
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
        Regular(Box<T>),
        Variadic(BytePos),
    }

    let variadic_arg = if variadic {
        let variadic_start = context.codemap.span_before(span, "...");
        Some(ArgumentKind::Variadic(variadic_start))
    } else {
        None
    };

    // 2 for ()
    let budget = try_opt!(shape.width.checked_sub(2));
    // 1 for (
    let offset = shape.indent + 1;
    let list_lo = context.codemap.span_after(span, "(");
    let items = itemize_list(
        context.codemap,
        // FIXME Would be nice to avoid this allocation,
        // but I couldn't get the types to work out.
        inputs.map(|i| ArgumentKind::Regular(Box::new(i))).chain(
            variadic_arg,
        ),
        ")",
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.span().lo,
            ArgumentKind::Variadic(start) => start,
        },
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.span().hi,
            ArgumentKind::Variadic(start) => start + BytePos(3),
        },
        |arg| match *arg {
            ArgumentKind::Regular(ref ty) => ty.rewrite(context, Shape::legacy(budget, offset)),
            ArgumentKind::Variadic(_) => Some("...".to_owned()),
        },
        list_lo,
        span.hi,
    );

    let list_str = try_opt!(format_fn_args(
        items,
        Shape::legacy(budget, offset),
        context.config,
    ));

    let output = match *output {
        FunctionRetTy::Ty(ref ty) => {
            let budget = try_opt!(shape.width.checked_sub(4));
            let type_str = try_opt!(ty.rewrite(context, Shape::legacy(budget, offset + 4)));
            format!(" -> {}", type_str)
        }
        FunctionRetTy::Default(..) => String::new(),
    };

    let infix = if !output.is_empty() && output.len() + list_str.len() > shape.width {
        format!("\n{}", (offset - 1).to_string(context.config))
    } else {
        String::new()
    };

    Some(if context.config.spaces_within_parens() {
        format!("( {} ){}{}", list_str, infix, output)
    } else {
        format!("({}){}{}", list_str, infix, output)
    })
}

fn type_bound_colon(context: &RewriteContext) -> &'static str {
    colon_spaces(
        context.config.space_before_bound(),
        context.config.space_after_bound_colon(),
    )
}

impl Rewrite for ast::WherePredicate {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        // TODO: dead spans?
        let result = match *self {
            ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                                                    ref bound_lifetimes,
                                                    ref bounded_ty,
                                                    ref bounds,
                                                    ..
                                                }) => {
                let type_str = try_opt!(bounded_ty.rewrite(context, shape));

                let colon = type_bound_colon(context);

                if !bound_lifetimes.is_empty() {
                    let lifetime_str: String = try_opt!(
                        bound_lifetimes
                            .iter()
                            .map(|lt| lt.rewrite(context, shape))
                            .collect::<Option<Vec<_>>>()
                    ).join(", ");

                    // 6 = "for<> ".len()
                    let used_width = lifetime_str.len() + type_str.len() + colon.len() + 6;
                    let ty_shape = try_opt!(shape.block_left(used_width));
                    let bounds: Vec<_> = try_opt!(
                        bounds
                            .iter()
                            .map(|ty_bound| ty_bound.rewrite(context, ty_shape))
                            .collect()
                    );
                    let bounds_str = join_bounds(context, ty_shape, &bounds);

                    if context.config.spaces_within_angle_brackets() && lifetime_str.len() > 0 {
                        format!(
                            "for< {} > {}{}{}",
                            lifetime_str,
                            type_str,
                            colon,
                            bounds_str
                        )
                    } else {
                        format!("for<{}> {}{}{}", lifetime_str, type_str, colon, bounds_str)
                    }
                } else {
                    let used_width = type_str.len() + colon.len();
                    let ty_shape = match context.config.where_style() {
                        Style::Legacy => try_opt!(shape.block_left(used_width)),
                        Style::Rfc => shape.block_indent(context.config.tab_spaces()),
                    };
                    let bounds: Vec<_> = try_opt!(
                        bounds
                            .iter()
                            .map(|ty_bound| ty_bound.rewrite(context, ty_shape))
                            .collect()
                    );
                    let bounds_str = join_bounds(context, ty_shape, &bounds);

                    format!("{}{}{}", type_str, colon, bounds_str)
                }
            }
            ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                                                     ref lifetime,
                                                     ref bounds,
                                                     ..
                                                 }) => {
                try_opt!(rewrite_bounded_lifetime(
                    lifetime,
                    bounds.iter(),
                    context,
                    shape,
                ))
            }
            ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                                                 ref lhs_ty,
                                                 ref rhs_ty,
                                                 ..
                                             }) => {
                let lhs_ty_str = try_opt!(lhs_ty.rewrite(context, shape));
                // 3 = " = ".len()
                let used_width = 3 + lhs_ty_str.len();
                let budget = try_opt!(shape.width.checked_sub(used_width));
                let rhs_ty_str = try_opt!(rhs_ty.rewrite(
                    context,
                    Shape::legacy(budget, shape.indent + used_width),
                ));
                format!("{} = {}", lhs_ty_str, rhs_ty_str)
            }
        };

        wrap_str(result, context.config.max_width(), shape)
    }
}

impl Rewrite for ast::LifetimeDef {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        rewrite_bounded_lifetime(&self.lifetime, self.bounds.iter(), context, shape)
    }
}

fn rewrite_bounded_lifetime<'b, I>(
    lt: &ast::Lifetime,
    bounds: I,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String>
where
    I: ExactSizeIterator<Item = &'b ast::Lifetime>,
{
    let result = try_opt!(lt.rewrite(context, shape));

    if bounds.len() == 0 {
        Some(result)
    } else {
        let appendix: Vec<_> = try_opt!(
            bounds
                .into_iter()
                .map(|b| b.rewrite(context, shape))
                .collect()
        );
        let colon = type_bound_colon(context);
        let overhead = last_line_width(&result) + colon.len();
        let result = format!(
            "{}{}{}",
            result,
            colon,
            join_bounds(context, try_opt!(shape.sub_width(overhead)), &appendix)
        );
        wrap_str(result, context.config.max_width(), shape)
    }
}

impl Rewrite for ast::TyParamBound {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::None) => {
                tref.rewrite(context, shape)
            }
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::Maybe) => {
                let budget = try_opt!(shape.width.checked_sub(1));
                Some(format!(
                    "?{}",
                    try_opt!(tref.rewrite(
                        context,
                        Shape::legacy(budget, shape.indent + 1),
                    ))
                ))
            }
            ast::TyParamBound::RegionTyParamBound(ref l) => l.rewrite(context, shape),
        }
    }
}

impl Rewrite for ast::Lifetime {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        wrap_str(
            pprust::lifetime_to_string(self),
            context.config.max_width(),
            shape,
        )
    }
}

impl Rewrite for ast::TyParamBounds {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let strs: Vec<_> = try_opt!(self.iter().map(|b| b.rewrite(context, shape)).collect());
        join_bounds(context, shape, &strs).rewrite(context, shape)
    }
}

impl Rewrite for ast::TyParam {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let mut result = String::with_capacity(128);
        // FIXME: If there are more than one attributes, this will force multiline.
        let attr_str = match (&*self.attrs).rewrite(context, shape) {
            Some(ref rw) if !rw.is_empty() => format!("{} ", rw),
            _ => String::new(),
        };
        result.push_str(&attr_str);
        result.push_str(&self.ident.to_string());
        if !self.bounds.is_empty() {
            result.push_str(type_bound_colon(context));
            let strs: Vec<_> = try_opt!(
                self.bounds
                    .iter()
                    .map(|ty_bound| ty_bound.rewrite(context, shape))
                    .collect()
            );
            result.push_str(&join_bounds(context, shape, &strs));
        }
        if let Some(ref def) = self.default {

            let eq_str = match context.config.type_punctuation_density() {
                TypeDensity::Compressed => "=",
                TypeDensity::Wide => " = ",
            };
            result.push_str(eq_str);
            let budget = try_opt!(shape.width.checked_sub(result.len()));
            let rewrite = try_opt!(def.rewrite(
                context,
                Shape::legacy(budget, shape.indent + result.len()),
            ));
            result.push_str(&rewrite);
        }

        wrap_str(result, context.config.max_width(), shape)
    }
}

impl Rewrite for ast::PolyTraitRef {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if !self.bound_lifetimes.is_empty() {
            let lifetime_str: String = try_opt!(
                self.bound_lifetimes
                    .iter()
                    .map(|lt| lt.rewrite(context, shape))
                    .collect::<Option<Vec<_>>>()
            ).join(", ");

            // 6 is "for<> ".len()
            let extra_offset = lifetime_str.len() + 6;
            let max_path_width = try_opt!(shape.width.checked_sub(extra_offset));
            let path_str = try_opt!(self.trait_ref.rewrite(
                context,
                Shape::legacy(
                    max_path_width,
                    shape.indent + extra_offset,
                ),
            ));

            Some(if context.config.spaces_within_angle_brackets() &&
                lifetime_str.len() > 0
            {
                format!("for< {} > {}", lifetime_str, path_str)
            } else {
                format!("for<{}> {}", lifetime_str, path_str)
            })
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
            ast::TyKind::TraitObject(ref bounds) => bounds.rewrite(context, shape),
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
                        let lt_budget = try_opt!(shape.width.checked_sub(2 + mut_len));
                        let lt_str = try_opt!(lifetime.rewrite(
                            context,
                            Shape::legacy(lt_budget, shape.indent + 2 + mut_len),
                        ));
                        let lt_len = lt_str.len();
                        let budget = try_opt!(shape.width.checked_sub(2 + mut_len + lt_len));
                        format!(
                            "&{} {}{}",
                            lt_str,
                            mut_str,
                            try_opt!(mt.ty.rewrite(
                                context,
                                Shape::legacy(
                                    budget,
                                    shape.indent + 2 + mut_len + lt_len,
                                ),
                            ))
                        )
                    }
                    None => {
                        let budget = try_opt!(shape.width.checked_sub(1 + mut_len));
                        format!(
                            "&{}{}",
                            mut_str,
                            try_opt!(mt.ty.rewrite(
                                context,
                                Shape::legacy(budget, shape.indent + 1 + mut_len),
                            ))
                        )
                    }
                })
            }
            // FIXME: we drop any comments here, even though it's a silly place to put
            // comments.
            ast::TyKind::Paren(ref ty) => {
                let budget = try_opt!(shape.width.checked_sub(2));
                ty.rewrite(context, Shape::legacy(budget, shape.indent + 1))
                    .map(|ty_str| if context.config.spaces_within_parens() {
                        format!("( {} )", ty_str)
                    } else {
                        format!("({})", ty_str)
                    })
            }
            ast::TyKind::Slice(ref ty) => {
                let budget = if context.config.spaces_within_square_brackets() {
                    try_opt!(shape.width.checked_sub(4))
                } else {
                    try_opt!(shape.width.checked_sub(2))
                };
                ty.rewrite(context, Shape::legacy(budget, shape.indent + 1))
                    .map(|ty_str| if context.config.spaces_within_square_brackets() {
                        format!("[ {} ]", ty_str)
                    } else {
                        format!("[{}]", ty_str)
                    })
            }
            ast::TyKind::Tup(ref items) => {
                rewrite_tuple_type(context, items.iter().map(|x| &**x), self.span, shape)
            }
            ast::TyKind::Path(ref q_self, ref path) => {
                rewrite_path(context, PathContext::Type, q_self.as_ref(), path, shape)
            }
            ast::TyKind::Array(ref ty, ref repeats) => {
                let use_spaces = context.config.spaces_within_square_brackets();
                let lbr = if use_spaces { "[ " } else { "[" };
                let rbr = if use_spaces { " ]" } else { "]" };
                rewrite_pair(&**ty, &**repeats, lbr, "; ", rbr, context, shape)
            }
            ast::TyKind::Infer => {
                if shape.width >= 1 {
                    Some("_".to_owned())
                } else {
                    None
                }
            }
            ast::TyKind::BareFn(ref bare_fn) => rewrite_bare_fn(bare_fn, self.span, context, shape),
            ast::TyKind::Never => Some(String::from("!")),
            ast::TyKind::Mac(..) => None,
            ast::TyKind::ImplicitSelf => Some(String::from("")),
            ast::TyKind::ImplTrait(ref it) => {
                it.rewrite(context, shape).map(|it_str| {
                    format!("impl {}", it_str)
                })
            }
            ast::TyKind::Err |
            ast::TyKind::Typeof(..) => unreachable!(),
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

    if !bare_fn.lifetimes.is_empty() {
        result.push_str("for<");
        // 6 = "for<> ".len(), 4 = "for<".
        // This doesn't work out so nicely for mutliline situation with lots of
        // rightward drift. If that is a problem, we could use the list stuff.
        result.push_str(&try_opt!(
            bare_fn
                .lifetimes
                .iter()
                .map(|l| {
                    l.rewrite(
                        context,
                        Shape::legacy(try_opt!(shape.width.checked_sub(6)), shape.indent + 4),
                    )
                })
                .collect::<Option<Vec<_>>>()
        ).join(", "));
        result.push_str("> ");
    }

    result.push_str(::utils::format_unsafety(bare_fn.unsafety));

    if bare_fn.abi != abi::Abi::Rust {
        result.push_str(&::utils::format_abi(
            bare_fn.abi,
            context.config.force_explicit_abi(),
        ));
    }

    result.push_str("fn");

    let budget = try_opt!(shape.width.checked_sub(result.len()));
    let indent = shape.indent + result.len();

    let rewrite = try_opt!(format_function_type(
        bare_fn.decl.inputs.iter(),
        &bare_fn.decl.output,
        bare_fn.decl.variadic,
        span,
        context,
        Shape::legacy(budget, indent),
    ));

    result.push_str(&rewrite);

    Some(result)
}

pub fn join_bounds(context: &RewriteContext, shape: Shape, type_strs: &Vec<String>) -> String {
    // Try to join types in a single line
    let joiner = match context.config.type_punctuation_density() {
        TypeDensity::Compressed => "+",
        TypeDensity::Wide => " + ",
    };
    let result = type_strs.join(joiner);
    if result.contains('\n') || result.len() > shape.width {
        let joiner_indent = shape.indent.block_indent(context.config);
        let joiner = format!("\n{}+ ", joiner_indent.to_string(context.config));
        type_strs.join(&joiner)
    } else {
        result
    }
}
