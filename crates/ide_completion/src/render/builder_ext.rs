//! Extensions for `Builder` structure required for item rendering.

use either::Either;
use itertools::Itertools;
use syntax::ast::{self, HasName};

use crate::{context::CallKind, item::Builder, patterns::ImmediateLocation, CompletionContext};

#[derive(Debug)]
pub(super) enum Params {
    Named(Vec<(Either<ast::SelfParam, ast::Param>, hir::Param)>),
    Anonymous(usize),
}

impl Params {
    pub(super) fn len(&self) -> usize {
        match self {
            Params::Named(xs) => xs.len(),
            Params::Anonymous(len) => *len,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Builder {
    fn should_add_parens(&self, ctx: &CompletionContext) -> bool {
        if !ctx.config.add_call_parenthesis {
            return false;
        }
        if ctx.in_use_tree() {
            cov_mark::hit!(no_parens_in_use_item);
            return false;
        }
        if matches!(ctx.path_call_kind(), Some(CallKind::Expr | CallKind::Pat))
            | matches!(
                ctx.completion_location,
                Some(ImmediateLocation::MethodCall { has_parens: true, .. })
            )
        {
            return false;
        }

        // Don't add parentheses if the expected type is some function reference.
        if let Some(ty) = &ctx.expected_type {
            if ty.is_fn() {
                cov_mark::hit!(no_call_parens_if_fn_ptr_needed);
                return false;
            }
        }

        // Nothing prevents us from adding parentheses
        true
    }

    pub(super) fn add_call_parens(
        &mut self,
        ctx: &CompletionContext,
        name: String,
        params: Params,
    ) -> &mut Builder {
        if !self.should_add_parens(ctx) {
            return self;
        }

        let cap = match ctx.config.snippet_cap {
            Some(it) => it,
            None => return self,
        };
        // If not an import, add parenthesis automatically.
        cov_mark::hit!(inserts_parens_for_function_calls);

        let (snippet, label) = if params.is_empty() {
            (format!("{}()$0", name), format!("{}()", name))
        } else {
            self.trigger_call_info();
            let snippet = match (ctx.config.add_call_argument_snippets, params) {
                (true, Params::Named(params)) => {
                    let function_params_snippet = params.iter().enumerate().format_with(
                        ", ",
                        |(index, (param_source, param)), f| {
                            let name;
                            let text;
                            let (ref_, name) = match param_source {
                                Either::Left(self_param) => (
                                    match self_param.kind() {
                                        ast::SelfParamKind::Owned => "",
                                        ast::SelfParamKind::Ref => "&",
                                        ast::SelfParamKind::MutRef => "&mut ",
                                    },
                                    "self",
                                ),
                                Either::Right(it) => {
                                    let n = (|| {
                                        let mut pat = it.pat()?;
                                        loop {
                                            match pat {
                                                ast::Pat::IdentPat(pat) => break pat.name(),
                                                ast::Pat::RefPat(it) => pat = it.pat()?,
                                                _ => return None,
                                            }
                                        }
                                    })();
                                    match n {
                                        Some(n) => {
                                            name = n;
                                            text = name.text();
                                            let text = text.as_str().trim_start_matches('_');
                                            let ref_ = ref_of_param(ctx, text, param.ty());
                                            (ref_, text)
                                        }
                                        None => ("", "_"),
                                    }
                                }
                            };
                            f(&format_args!("${{{}:{}{}}}", index + 1, ref_, name))
                        },
                    );
                    format!("{}({})$0", name, function_params_snippet)
                }
                _ => {
                    cov_mark::hit!(suppress_arg_snippets);
                    format!("{}($0)", name)
                }
            };

            (snippet, format!("{}(…)", name))
        };
        self.lookup_by(name).label(label).insert_snippet(cap, snippet)
    }
}
fn ref_of_param(ctx: &CompletionContext, arg: &str, ty: &hir::Type) -> &'static str {
    if let Some(derefed_ty) = ty.remove_ref() {
        for (name, local) in ctx.locals.iter() {
            if name.as_text().as_deref() == Some(arg) && local.ty(ctx.db) == derefed_ty {
                return if ty.is_mutable_reference() { "&mut " } else { "&" };
            }
        }
    }
    ""
}
