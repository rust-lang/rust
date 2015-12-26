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

use syntax::ast::{self, Mutability, FunctionRetTy};
use syntax::print::pprust;
use syntax::codemap::{self, Span, BytePos};
use syntax::abi;

use {Indent, Spanned};
use lists::{format_item_list, itemize_list, format_fn_args};
use rewrite::{Rewrite, RewriteContext};
use utils::{extra_offset, span_after, format_mutability, wrap_str};
use expr::{rewrite_unary_prefix, rewrite_pair, rewrite_tuple};

// Does not wrap on simple segments.
pub fn rewrite_path(context: &RewriteContext,
                    expr_context: bool,
                    qself: Option<&ast::QSelf>,
                    path: &ast::Path,
                    width: usize,
                    offset: Indent)
                    -> Option<String> {
    let skip_count = qself.map(|x| x.position).unwrap_or(0);

    let mut result = if path.global {
        "::".to_owned()
    } else {
        String::new()
    };

    let mut span_lo = path.span.lo;

    if let Some(ref qself) = qself {
        result.push('<');
        let fmt_ty = try_opt!(qself.ty.rewrite(context, width, offset));
        result.push_str(&fmt_ty);

        if skip_count > 0 {
            result.push_str(" as ");

            let extra_offset = extra_offset(&result, offset);
            // 3 = ">::".len()
            let budget = try_opt!(width.checked_sub(extra_offset + 3));

            result = try_opt!(rewrite_path_segments(expr_context,
                                                    result,
                                                    path.segments.iter().take(skip_count),
                                                    span_lo,
                                                    path.span.hi,
                                                    context,
                                                    budget,
                                                    offset + extra_offset));
        }

        result.push_str(">::");
        span_lo = qself.ty.span.hi + BytePos(1);
    }

    let extra_offset = extra_offset(&result, offset);
    let budget = try_opt!(width.checked_sub(extra_offset));
    rewrite_path_segments(expr_context,
                          result,
                          path.segments.iter().skip(skip_count),
                          span_lo,
                          path.span.hi,
                          context,
                          budget,
                          offset + extra_offset)
}

fn rewrite_path_segments<'a, I>(expr_context: bool,
                                mut buffer: String,
                                iter: I,
                                mut span_lo: BytePos,
                                span_hi: BytePos,
                                context: &RewriteContext,
                                width: usize,
                                offset: Indent)
                                -> Option<String>
    where I: Iterator<Item = &'a ast::PathSegment>
{
    let mut first = true;

    for segment in iter {
        if first {
            first = false;
        } else {
            buffer.push_str("::");
        }

        let extra_offset = extra_offset(&buffer, offset);
        let remaining_width = try_opt!(width.checked_sub(extra_offset));
        let new_offset = offset + extra_offset;
        let segment_string = try_opt!(rewrite_segment(expr_context,
                                                      segment,
                                                      &mut span_lo,
                                                      span_hi,
                                                      context,
                                                      remaining_width,
                                                      new_offset));

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
            SegmentParam::LifeTime(ref lt) => lt.span,
            SegmentParam::Type(ref ty) => ty.span,
            SegmentParam::Binding(ref binding) => binding.span,
        }
    }
}

impl<'a> Rewrite for SegmentParam<'a> {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match *self {
            SegmentParam::LifeTime(ref lt) => lt.rewrite(context, width, offset),
            SegmentParam::Type(ref ty) => ty.rewrite(context, width, offset),
            SegmentParam::Binding(ref binding) => {
                let mut result = format!("{} = ", binding.ident);
                let budget = try_opt!(width.checked_sub(result.len()));
                let rewrite = try_opt!(binding.ty.rewrite(context, budget, offset + result.len()));
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
fn rewrite_segment(expr_context: bool,
                   segment: &ast::PathSegment,
                   span_lo: &mut BytePos,
                   span_hi: BytePos,
                   context: &RewriteContext,
                   width: usize,
                   offset: Indent)
                   -> Option<String> {
    let ident_len = segment.identifier.to_string().len();
    let width = try_opt!(width.checked_sub(ident_len));
    let offset = offset + ident_len;

    let params = match segment.parameters {
        ast::PathParameters::AngleBracketedParameters(ref data) if !data.lifetimes.is_empty() ||
                                                                   !data.types.is_empty() ||
                                                                   !data.bindings.is_empty() => {
            let param_list = data.lifetimes
                                 .iter()
                                 .map(SegmentParam::LifeTime)
                                 .chain(data.types.iter().map(|x| SegmentParam::Type(&*x)))
                                 .chain(data.bindings.iter().map(|x| SegmentParam::Binding(&*x)))
                                 .collect::<Vec<_>>();

            let next_span_lo = param_list.last().unwrap().get_span().hi + BytePos(1);
            let list_lo = span_after(codemap::mk_sp(*span_lo, span_hi), "<", context.codemap);
            let separator = if expr_context {
                "::"
            } else {
                ""
            };

            // 1 for <
            let extra_offset = 1 + separator.len();
            // 1 for >
            let list_width = try_opt!(width.checked_sub(extra_offset + 1));

            let items = itemize_list(context.codemap,
                                     param_list.into_iter(),
                                     ">",
                                     |param| param.get_span().lo,
                                     |param| param.get_span().hi,
                                     |seg| seg.rewrite(context, list_width, offset + extra_offset),
                                     list_lo,
                                     span_hi);
            let list_str = try_opt!(format_item_list(items,
                                                     list_width,
                                                     offset + extra_offset,
                                                     context.config));

            // Update position of last bracket.
            *span_lo = next_span_lo;

            format!("{}<{}>", separator, list_str)
        }
        ast::PathParameters::ParenthesizedParameters(ref data) => {
            let output = match data.output {
                Some(ref ty) => FunctionRetTy::Return(ty.clone()),
                None => FunctionRetTy::DefaultReturn(codemap::DUMMY_SP),
            };
            try_opt!(format_function_type(data.inputs.iter().map(|x| &**x),
                                          &output,
                                          data.span,
                                          context,
                                          width,
                                          offset))
        }
        _ => String::new(),
    };

    Some(format!("{}{}", segment.identifier, params))
}

fn format_function_type<'a, I>(inputs: I,
                               output: &FunctionRetTy,
                               span: Span,
                               context: &RewriteContext,
                               width: usize,
                               offset: Indent)
                               -> Option<String>
    where I: ExactSizeIterator,
          <I as Iterator>::Item: Deref,
          <I::Item as Deref>::Target: Rewrite + Spanned + 'a
{
    // 2 for ()
    let budget = try_opt!(width.checked_sub(2));
    // 1 for (
    let offset = offset + 1;
    let list_lo = span_after(span, "(", context.codemap);
    let items = itemize_list(context.codemap,
                             inputs,
                             ")",
                             |ty| ty.span().lo,
                             |ty| ty.span().hi,
                             |ty| ty.rewrite(context, budget, offset),
                             list_lo,
                             span.hi);

    let list_str = try_opt!(format_fn_args(items, budget, offset, context.config));

    let output = match *output {
        FunctionRetTy::Return(ref ty) => {
            let budget = try_opt!(width.checked_sub(4));
            let type_str = try_opt!(ty.rewrite(context, budget, offset + 4));
            format!(" -> {}", type_str)
        }
        FunctionRetTy::NoReturn(..) => " -> !".to_owned(),
        FunctionRetTy::DefaultReturn(..) => String::new(),
    };

    let infix = if output.len() + list_str.len() > width {
        format!("\n{}", (offset - 1).to_string(context.config))
    } else {
        String::new()
    };

    Some(format!("({}){}{}", list_str, infix, output))
}

impl Rewrite for ast::WherePredicate {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        // TODO: dead spans?
        let result = match *self {
            ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate { ref bound_lifetimes,
                                                                           ref bounded_ty,
                                                                           ref bounds,
                                                                           .. }) => {
                let type_str = try_opt!(bounded_ty.rewrite(context, width, offset));

                if !bound_lifetimes.is_empty() {
                    let lifetime_str = try_opt!(bound_lifetimes.iter()
                                                               .map(|lt| {
                                                                   lt.rewrite(context,
                                                                              width,
                                                                              offset)
                                                               })
                                                               .collect::<Option<Vec<_>>>())
                                           .join(", ");
                    // 8 = "for<> : ".len()
                    let used_width = lifetime_str.len() + type_str.len() + 8;
                    let budget = try_opt!(width.checked_sub(used_width));
                    let bounds_str = try_opt!(bounds.iter()
                                                    .map(|ty_bound| {
                                                        ty_bound.rewrite(context,
                                                                         budget,
                                                                         offset + used_width)
                                                    })
                                                    .collect::<Option<Vec<_>>>())
                                         .join(" + ");

                    format!("for<{}> {}: {}", lifetime_str, type_str, bounds_str)
                } else {
                    // 2 = ": ".len()
                    let used_width = type_str.len() + 2;
                    let budget = try_opt!(width.checked_sub(used_width));
                    let bounds_str = try_opt!(bounds.iter()
                                                    .map(|ty_bound| {
                                                        ty_bound.rewrite(context,
                                                                         budget,
                                                                         offset + used_width)
                                                    })
                                                    .collect::<Option<Vec<_>>>())
                                         .join(" + ");

                    format!("{}: {}", type_str, bounds_str)
                }
            }
            ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate { ref lifetime,
                                                                             ref bounds,
                                                                             .. }) => {
                try_opt!(rewrite_bounded_lifetime(lifetime, bounds.iter(), context, width, offset))
            }
            ast::WherePredicate::EqPredicate(ast::WhereEqPredicate { ref path, ref ty, .. }) => {
                let ty_str = try_opt!(ty.rewrite(context, width, offset));
                // 3 = " = ".len()
                let used_width = 3 + ty_str.len();
                let budget = try_opt!(width.checked_sub(used_width));
                let path_str = try_opt!(rewrite_path(context,
                                                     false,
                                                     None,
                                                     path,
                                                     budget,
                                                     offset + used_width));
                format!("{} = {}", path_str, ty_str)
            }
        };

        wrap_str(result, context.config.max_width, width, offset)
    }
}

impl Rewrite for ast::LifetimeDef {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        rewrite_bounded_lifetime(&self.lifetime, self.bounds.iter(), context, width, offset)
    }
}

fn rewrite_bounded_lifetime<'b, I>(lt: &ast::Lifetime,
                                   bounds: I,
                                   context: &RewriteContext,
                                   width: usize,
                                   offset: Indent)
                                   -> Option<String>
    where I: ExactSizeIterator<Item = &'b ast::Lifetime>
{
    let result = try_opt!(lt.rewrite(context, width, offset));

    if bounds.len() == 0 {
        Some(result)
    } else {
        let appendix: Vec<_> = try_opt!(bounds.into_iter()
                                              .map(|b| b.rewrite(context, width, offset))
                                              .collect());
        let result = format!("{}: {}", result, appendix.join(" + "));
        wrap_str(result, context.config.max_width, width, offset)
    }
}

impl Rewrite for ast::TyParamBound {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match *self {
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::None) => {
                tref.rewrite(context, width, offset)
            }
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::Maybe) => {
                let budget = try_opt!(width.checked_sub(1));
                Some(format!("?{}", try_opt!(tref.rewrite(context, budget, offset + 1))))
            }
            ast::TyParamBound::RegionTyParamBound(ref l) => l.rewrite(context, width, offset),
        }
    }
}

impl Rewrite for ast::Lifetime {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        wrap_str(pprust::lifetime_to_string(self),
                 context.config.max_width,
                 width,
                 offset)
    }
}

impl Rewrite for ast::TyParamBounds {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        let strs: Vec<_> = try_opt!(self.iter()
                                        .map(|b| b.rewrite(context, width, offset))
                                        .collect());
        wrap_str(strs.join(" + "), context.config.max_width, width, offset)
    }
}

impl Rewrite for ast::TyParam {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        let mut result = String::with_capacity(128);
        result.push_str(&self.ident.to_string());
        if !self.bounds.is_empty() {
            result.push_str(": ");

            let bounds = try_opt!(self.bounds
                                      .iter()
                                      .map(|ty_bound| ty_bound.rewrite(context, width, offset))
                                      .collect::<Option<Vec<_>>>())
                             .join(" + ");

            result.push_str(&bounds);
        }
        if let Some(ref def) = self.default {
            result.push_str(" = ");
            let budget = try_opt!(width.checked_sub(result.len()));
            let rewrite = try_opt!(def.rewrite(context, budget, offset + result.len()));
            result.push_str(&rewrite);
        }

        wrap_str(result, context.config.max_width, width, offset)
    }
}

impl Rewrite for ast::PolyTraitRef {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        if !self.bound_lifetimes.is_empty() {
            let lifetime_str = try_opt!(self.bound_lifetimes
                                            .iter()
                                            .map(|lt| lt.rewrite(context, width, offset))
                                            .collect::<Option<Vec<_>>>())
                                   .join(", ");
            // 6 is "for<> ".len()
            let extra_offset = lifetime_str.len() + 6;
            let max_path_width = try_opt!(width.checked_sub(extra_offset));
            let path_str = try_opt!(self.trait_ref
                                        .rewrite(context, max_path_width, offset + extra_offset));

            Some(format!("for<{}> {}", lifetime_str, path_str))
        } else {
            self.trait_ref.rewrite(context, width, offset)
        }
    }
}

impl Rewrite for ast::TraitRef {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        rewrite_path(context, false, None, &self.path, width, offset)
    }
}

impl Rewrite for ast::Ty {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match self.node {
            ast::TyObjectSum(ref ty, ref bounds) => {
                let ty_str = try_opt!(ty.rewrite(context, width, offset));
                let overhead = ty_str.len() + 3;
                Some(format!("{} + {}",
                             ty_str,
                             try_opt!(bounds.rewrite(context,
                                                     try_opt!(width.checked_sub(overhead)),
                                                     offset + overhead))))
            }
            ast::TyPtr(ref mt) => {
                let prefix = match mt.mutbl {
                    Mutability::MutMutable => "*mut ",
                    Mutability::MutImmutable => "*const ",
                };

                rewrite_unary_prefix(context, prefix, &*mt.ty, width, offset)
            }
            ast::TyRptr(ref lifetime, ref mt) => {
                let mut_str = format_mutability(mt.mutbl);
                let mut_len = mut_str.len();
                Some(match *lifetime {
                    Some(ref lifetime) => {
                        let lt_budget = try_opt!(width.checked_sub(2 + mut_len));
                        let lt_str = try_opt!(lifetime.rewrite(context,
                                                               lt_budget,
                                                               offset + 2 + mut_len));
                        let lt_len = lt_str.len();
                        let budget = try_opt!(width.checked_sub(2 + mut_len + lt_len));
                        format!("&{} {}{}",
                                lt_str,
                                mut_str,
                                try_opt!(mt.ty.rewrite(context,
                                                       budget,
                                                       offset + 2 + mut_len + lt_len)))
                    }
                    None => {
                        let budget = try_opt!(width.checked_sub(1 + mut_len));
                        format!("&{}{}",
                                mut_str,
                                try_opt!(mt.ty.rewrite(context, budget, offset + 1 + mut_len)))
                    }
                })
            }
            // FIXME: we drop any comments here, even though it's a silly place to put
            // comments.
            ast::TyParen(ref ty) => {
                let budget = try_opt!(width.checked_sub(2));
                ty.rewrite(context, budget, offset + 1).map(|ty_str| format!("({})", ty_str))
            }
            ast::TyVec(ref ty) => {
                let budget = try_opt!(width.checked_sub(2));
                ty.rewrite(context, budget, offset + 1).map(|ty_str| format!("[{}]", ty_str))
            }
            ast::TyTup(ref items) => {
                rewrite_tuple(context,
                              items.iter().map(|x| &**x),
                              self.span,
                              width,
                              offset)
            }
            ast::TyPolyTraitRef(ref trait_ref) => trait_ref.rewrite(context, width, offset),
            ast::TyPath(ref q_self, ref path) => {
                rewrite_path(context, false, q_self.as_ref(), path, width, offset)
            }
            ast::TyFixedLengthVec(ref ty, ref repeats) => {
                rewrite_pair(&**ty, &**repeats, "[", "; ", "]", context, width, offset)
            }
            ast::TyInfer => {
                if width >= 1 {
                    Some("_".to_owned())
                } else {
                    None
                }
            }
            ast::TyBareFn(ref bare_fn) => {
                rewrite_bare_fn(bare_fn, self.span, context, width, offset)
            }
            ast::TyMac(..) | ast::TyTypeof(..) => unreachable!(),
        }
    }
}

fn rewrite_bare_fn(bare_fn: &ast::BareFnTy,
                   span: Span,
                   context: &RewriteContext,
                   width: usize,
                   offset: Indent)
                   -> Option<String> {
    let mut result = String::with_capacity(128);

    result.push_str(&::utils::format_unsafety(bare_fn.unsafety));

    if bare_fn.abi != abi::Rust {
        result.push_str(&::utils::format_abi(bare_fn.abi));
    }

    result.push_str("fn");

    let budget = try_opt!(width.checked_sub(result.len()));
    let indent = offset + result.len();

    let rewrite = try_opt!(format_function_type(bare_fn.decl.inputs.iter(),
                                                &bare_fn.decl.output,
                                                span,
                                                context,
                                                budget,
                                                indent));

    result.push_str(&rewrite);

    Some(result)
}
