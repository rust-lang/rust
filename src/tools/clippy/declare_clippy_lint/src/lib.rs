#![feature(let_chains)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Attribute, Error, Expr, ExprLit, Ident, Lit, LitStr, Meta, Result, Token};

fn parse_attr<const LEN: usize>(path: [&'static str; LEN], attr: &Attribute) -> Option<LitStr> {
    if let Meta::NameValue(name_value) = &attr.meta {
        let path_idents = name_value.path.segments.iter().map(|segment| &segment.ident);

        if itertools::equal(path_idents, path)
            && let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = &name_value.value
        {
            return Some(s.clone());
        }
    }

    None
}

struct ClippyLint {
    attrs: Vec<Attribute>,
    explanation: String,
    name: Ident,
    category: Ident,
    description: LitStr,
}

impl Parse for ClippyLint {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;

        let mut in_code = false;
        let mut explanation = String::new();
        let mut version = None;
        for attr in &attrs {
            if let Some(lit) = parse_attr(["doc"], attr) {
                let value = lit.value();
                let line = value.strip_prefix(' ').unwrap_or(&value);

                if line.starts_with("```") {
                    explanation += "```\n";
                    in_code = !in_code;
                } else if !(in_code && line.starts_with("# ")) {
                    explanation += line;
                    explanation.push('\n');
                }
            } else if let Some(lit) = parse_attr(["clippy", "version"], attr) {
                if let Some(duplicate) = version.replace(lit) {
                    return Err(Error::new_spanned(duplicate, "duplicate clippy::version"));
                }
            } else {
                return Err(Error::new_spanned(attr, "unexpected attribute"));
            }
        }

        input.parse::<Token![pub]>()?;
        let name = input.parse()?;
        input.parse::<Token![,]>()?;

        let category = input.parse()?;
        input.parse::<Token![,]>()?;

        let description = input.parse()?;

        Ok(Self {
            attrs,
            explanation,
            name,
            category,
            description,
        })
    }
}

/// Macro used to declare a Clippy lint.
///
/// Every lint declaration consists of 4 parts:
///
/// 1. The documentation, which is used for the website and `cargo clippy --explain`
/// 2. The `LINT_NAME`. See [lint naming][lint_naming] on lint naming conventions.
/// 3. The `lint_level`, which is a mapping from *one* of our lint groups to `Allow`, `Warn` or
///    `Deny`. The lint level here has nothing to do with what lint groups the lint is a part of.
/// 4. The `description` that contains a short explanation on what's wrong with code where the lint
///    is triggered.
///
/// Currently the categories `style`, `correctness`, `suspicious`, `complexity` and `perf` are
/// enabled by default. As said in the README.md of this repository, if the lint level mapping
/// changes, please update README.md.
///
/// # Example
///
/// ```
/// use rustc_session::declare_tool_lint;
///
/// declare_clippy_lint! {
///     /// ### What it does
///     /// Checks for ... (describe what the lint matches).
///     ///
///     /// ### Why is this bad?
///     /// Supply the reason for linting the code.
///     ///
///     /// ### Example
///     /// ```rust
///     /// Insert a short example of code that triggers the lint
///     /// ```
///     ///
///     /// Use instead:
///     /// ```rust
///     /// Insert a short example of improved code that doesn't trigger the lint
///     /// ```
///     #[clippy::version = "1.65.0"]
///     pub LINT_NAME,
///     pedantic,
///     "description"
/// }
/// ```
/// [lint_naming]: https://rust-lang.github.io/rfcs/0344-conventions-galore.html#lints
#[proc_macro]
pub fn declare_clippy_lint(input: TokenStream) -> TokenStream {
    let ClippyLint {
        attrs,
        explanation,
        name,
        category,
        description,
    } = parse_macro_input!(input as ClippyLint);

    let mut category = category.to_string();

    let level = format_ident!(
        "{}",
        match category.as_str() {
            "correctness" => "Deny",
            "style" | "suspicious" | "complexity" | "perf" | "internal_warn" => "Warn",
            "pedantic" | "restriction" | "cargo" | "nursery" | "internal" => "Allow",
            _ => panic!("unknown category {category}"),
        },
    );

    let info = if category == "internal_warn" {
        None
    } else {
        let info_name = format_ident!("{name}_INFO");

        (&mut category[0..1]).make_ascii_uppercase();
        let category_variant = format_ident!("{category}");

        Some(quote! {
            pub(crate) static #info_name: &'static crate::LintInfo = &crate::LintInfo {
                lint: &#name,
                category: crate::LintCategory::#category_variant,
                explanation: #explanation,
            };
        })
    };

    let output = quote! {
        declare_tool_lint! {
            #(#attrs)*
            pub clippy::#name,
            #level,
            #description,
            report_in_external_macro: true
        }

        #info
    };

    TokenStream::from(output)
}
