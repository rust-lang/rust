use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Expr, Path, Token};

use crate::diagnostics::error::span_err;
use crate::diagnostics::utils::SetOnce;

pub(crate) struct DiagAttribute {
    /// Slug is a mandatory part of the struct attribute as corresponds to the Fluent message that
    /// has the actual diagnostic message.
    pub slug: Path,

    /// Error codes are a optional part of the struct attribute - this is only set to detect
    /// multiple specifications.
    pub code: Option<Expr>,
}

impl Parse for DiagAttribute {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let slug = input.parse::<Path>()?;
        let mut code = None;

        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            } // Allow trailing comma
            let arg_path = input.parse::<Path>()?;
            input.parse::<Token![=]>()?;
            if arg_path.is_ident("code") {
                let code_val = input.parse::<Expr>()?;
                code.set_once(code_val, arg_path.span().unwrap());
            } else {
                span_err(arg_path.span().unwrap(), "unknown argument")
                    .note("only the `code` parameter is valid after the slug")
                    .emit();
            }
        }

        Ok(Self { slug, code: code.map(|(code, _span)| code) })
    }
}

pub(crate) struct Mismatch {
    pub(crate) slug_name: String,
    pub(crate) crate_name: String,
    pub(crate) slug_prefix: String,
}

impl Mismatch {
    /// Checks whether the slug starts with the crate name it's in.
    pub(crate) fn check(slug: &syn::Path) -> Option<Mismatch> {
        // If this is missing we're probably in a test, so bail.
        let crate_name = std::env::var("CARGO_CRATE_NAME").ok()?;

        // If we're not in a "rustc_" crate, bail.
        let Some(("rustc", slug_prefix)) = crate_name.split_once('_') else { return None };

        let slug_name = slug.segments.first()?.ident.to_string();
        if slug_name.starts_with(slug_prefix) {
            return None;
        }

        Some(Mismatch { slug_name, slug_prefix: slug_prefix.to_string(), crate_name })
    }
}
