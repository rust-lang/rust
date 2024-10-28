use std::collections::BTreeMap;

use proc_macro2::Span;
use quote::ToTokens;
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token::Comma;
use syn::{Arm, Attribute, Expr, ExprMatch, Ident, Meta, Token, bracketed};

/// The input to our macro; just a list of `field: value` items.
#[derive(Debug)]
pub struct Invocation {
    fields: Punctuated<Mapping, Comma>,
}

impl Parse for Invocation {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self { fields: input.parse_terminated(Mapping::parse, Token![,])? })
    }
}

/// A `key: expression` mapping with nothing else. Basically a simplified `syn::Field`.
#[derive(Debug)]
struct Mapping {
    name: Ident,
    _sep: Token![:],
    expr: Expr,
}

impl Parse for Mapping {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self { name: input.parse()?, _sep: input.parse()?, expr: input.parse()? })
    }
}

/// The input provided to our proc macro, after parsing into the form we expect.
#[derive(Debug)]
pub struct StructuredInput {
    /// Macro to invoke once per function
    pub callback: Ident,
    /// Skip these functions
    pub skip: Vec<Ident>,
    /// Invoke only for these functions
    pub only: Option<Vec<Ident>>,
    /// Attributes that get applied to specific functions
    pub attributes: Option<Vec<AttributeMap>>,
    /// Extra expressions to pass to all invocations of the macro
    pub extra: Option<Expr>,
    /// Per-function extra expressions to pass to the macro
    pub fn_extra: Option<BTreeMap<Ident, Expr>>,
    // For diagnostics
    pub only_span: Option<Span>,
    pub fn_extra_span: Option<Span>,
}

impl StructuredInput {
    pub fn from_fields(input: Invocation) -> syn::Result<Self> {
        let mut map: Vec<_> = input.fields.into_iter().collect();
        let cb_expr = expect_field(&mut map, "callback")?;
        let skip_expr = expect_field(&mut map, "skip").ok();
        let only_expr = expect_field(&mut map, "only").ok();
        let attr_expr = expect_field(&mut map, "attributes").ok();
        let extra = expect_field(&mut map, "extra").ok();
        let fn_extra = expect_field(&mut map, "fn_extra").ok();

        if !map.is_empty() {
            Err(syn::Error::new(
                map.first().unwrap().name.span(),
                format!("unexpected fields {map:?}"),
            ))?;
        }

        let skip = match skip_expr {
            Some(expr) => Parser::parse2(parse_ident_array, expr.into_token_stream())?,
            None => Vec::new(),
        };

        let only_span = only_expr.as_ref().map(|expr| expr.span());
        let only = match only_expr {
            Some(expr) => Some(Parser::parse2(parse_ident_array, expr.into_token_stream())?),
            None => None,
        };

        let attributes = match attr_expr {
            Some(expr) => {
                let mut attributes = Vec::new();
                let attr_exprs = Parser::parse2(parse_expr_array, expr.into_token_stream())?;

                for attr in attr_exprs {
                    attributes.push(syn::parse2(attr.into_token_stream())?);
                }
                Some(attributes)
            }
            None => None,
        };

        let fn_extra_span = fn_extra.as_ref().map(|expr| expr.span());
        let fn_extra = match fn_extra {
            Some(expr) => Some(extract_fn_extra_field(expr)?),
            None => None,
        };

        Ok(Self {
            callback: expect_ident(cb_expr)?,
            skip,
            only,
            only_span,
            attributes,
            extra,
            fn_extra,
            fn_extra_span,
        })
    }
}

fn extract_fn_extra_field(expr: Expr) -> syn::Result<BTreeMap<Ident, Expr>> {
    let Expr::Match(mexpr) = expr else {
        let e = syn::Error::new(expr.span(), "`fn_extra` expects a match expression");
        return Err(e);
    };

    let ExprMatch { attrs, match_token: _, expr, brace_token: _, arms } = mexpr;

    expect_empty_attrs(&attrs)?;

    let match_on = expect_ident(*expr)?;
    if match_on != "MACRO_FN_NAME" {
        let e = syn::Error::new(match_on.span(), "only allowed to match on `MACRO_FN_NAME`");
        return Err(e);
    }

    let mut res = BTreeMap::new();

    for arm in arms {
        let Arm { attrs, pat, guard, fat_arrow_token: _, body, comma: _ } = arm;

        expect_empty_attrs(&attrs)?;

        let keys = match pat {
            syn::Pat::Wild(w) => vec![Ident::new("_", w.span())],
            _ => Parser::parse2(parse_ident_pat, pat.into_token_stream())?,
        };

        if let Some(guard) = guard {
            let e = syn::Error::new(guard.0.span(), "no guards allowed in this position");
            return Err(e);
        }

        for key in keys {
            let inserted = res.insert(key.clone(), *body.clone());
            if inserted.is_some() {
                let e = syn::Error::new(key.span(), format!("key `{key}` specified twice"));
                return Err(e);
            }
        }
    }

    Ok(res)
}

fn expect_empty_attrs(attrs: &[Attribute]) -> syn::Result<()> {
    if attrs.is_empty() {
        return Ok(());
    }

    let e =
        syn::Error::new(attrs.first().unwrap().span(), "no attributes allowed in this position");
    Err(e)
}

/// Extract a named field from a map, raising an error if it doesn't exist.
fn expect_field(v: &mut Vec<Mapping>, name: &str) -> syn::Result<Expr> {
    let pos = v.iter().position(|v| v.name == name).ok_or_else(|| {
        syn::Error::new(Span::call_site(), format!("missing expected field `{name}`"))
    })?;

    Ok(v.remove(pos).expr)
}

/// Coerce an expression into a simple identifier.
fn expect_ident(expr: Expr) -> syn::Result<Ident> {
    syn::parse2(expr.into_token_stream())
}

/// Parse an array of expressions.
fn parse_expr_array(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    let _ = bracketed!(content in input);
    let fields = content.parse_terminated(Expr::parse, Token![,])?;
    Ok(fields.into_iter().collect())
}

/// Parse an array of idents, e.g. `[foo, bar, baz]`.
fn parse_ident_array(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    let _ = bracketed!(content in input);
    let fields = content.parse_terminated(Ident::parse, Token![,])?;
    Ok(fields.into_iter().collect())
}

/// Parse an pattern of idents, specifically `(foo | bar | baz)`.
fn parse_ident_pat(input: ParseStream) -> syn::Result<Vec<Ident>> {
    if !input.peek2(Token![|]) {
        return Ok(vec![input.parse()?]);
    }

    let fields = Punctuated::<Ident, Token![|]>::parse_separated_nonempty(input)?;
    Ok(fields.into_iter().collect())
}

/// A mapping of attributes to identifiers (just a simplified `Expr`).
///
/// Expressed as:
///
/// ```ignore
/// #[meta1]
/// #[meta2]
/// [foo, bar, baz]
/// ```
#[derive(Debug)]
pub struct AttributeMap {
    pub meta: Vec<Meta>,
    pub names: Vec<Ident>,
}

impl Parse for AttributeMap {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;

        Ok(Self {
            meta: attrs.into_iter().map(|a| a.meta).collect(),
            names: parse_ident_array(input)?,
        })
    }
}
