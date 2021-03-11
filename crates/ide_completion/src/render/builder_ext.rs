//! Extensions for `Builder` structure required for item rendering.

use itertools::Itertools;

use crate::{item::Builder, CompletionContext};

#[derive(Debug)]
pub(super) enum Params {
    Named(Vec<String>),
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
        if ctx.use_item_syntax.is_some() {
            cov_mark::hit!(no_parens_in_use_item);
            return false;
        }
        if ctx.is_pattern_call {
            return false;
        }
        if ctx.is_call {
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
                    let function_params_snippet =
                        params.iter().enumerate().format_with(", ", |(index, param_name), f| {
                            f(&format_args!("${{{}:{}}}", index + 1, param_name))
                        });
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
