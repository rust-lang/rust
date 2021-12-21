//! Extensions for `Builder` structure required for item rendering.

use itertools::Itertools;
use syntax::SmolStr;

use crate::{context::PathKind, item::Builder, patterns::ImmediateLocation, CompletionContext};

#[derive(Debug)]
pub(super) enum Params {
    Named(Option<hir::SelfParam>, Vec<hir::Param>),
    Anonymous(usize),
}

impl Params {
    pub(super) fn len(&self) -> usize {
        match self {
            Params::Named(selv, params) => params.len() + if selv.is_some() { 1 } else { 0 },
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
        if let Some(PathKind::Use) = ctx.path_kind() {
            cov_mark::hit!(no_parens_in_use_item);
            return false;
        }
        if matches!(ctx.path_kind(), Some(PathKind::Expr | PathKind::Pat) if ctx.path_is_call())
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
        name: SmolStr,
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
                (true, Params::Named(self_param, params)) => {
                    let offset = if self_param.is_some() { 2 } else { 1 };
                    let function_params_snippet = params.iter().enumerate().format_with(
                        ", ",
                        |(index, param), f| match param.name(ctx.db) {
                            Some(n) => {
                                let smol_str = n.to_smol_str();
                                let text = smol_str.as_str().trim_start_matches('_');
                                let ref_ = ref_of_param(ctx, text, param.ty());
                                f(&format_args!("${{{}:{}{}}}", index + offset, ref_, text))
                            }
                            None => f(&format_args!("${{{}:_}}", index + offset,)),
                        },
                    );
                    match self_param {
                        Some(self_param) => {
                            format!(
                                "{}(${{1:{}}}{}{})$0",
                                name,
                                self_param.display(ctx.db),
                                if params.is_empty() { "" } else { ", " },
                                function_params_snippet
                            )
                        }
                        None => {
                            format!("{}({})$0", name, function_params_snippet)
                        }
                    }
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
