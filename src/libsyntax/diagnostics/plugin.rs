use std::collections::BTreeMap;
use std::env;

use crate::ast::{self, Ident, Name};
use crate::source_map;
use crate::ext::base::{ExtCtxt, MacEager, MacResult};
use crate::ext::build::AstBuilder;
use crate::parse::token::{self, Token};
use crate::ptr::P;
use crate::symbol::kw;
use crate::tokenstream::{TokenTree};

use smallvec::smallvec;
use syntax_pos::Span;

use crate::diagnostics::metadata::output_metadata;

pub use errors::*;

// Maximum width of any line in an extended error description (inclusive).
const MAX_DESCRIPTION_WIDTH: usize = 80;

/// Error information type.
pub struct ErrorInfo {
    pub description: Option<Name>,
    pub use_site: Option<Span>
}

/// Mapping from error codes to metadata.
pub type ErrorMap = BTreeMap<Name, ErrorInfo>;

pub fn expand_diagnostic_used<'cx>(ecx: &'cx mut ExtCtxt<'_>,
                                   span: Span,
                                   token_tree: &[TokenTree])
                                   -> Box<dyn MacResult+'cx> {
    let code = match token_tree {
        [
            TokenTree::Token(Token { kind: token::Ident(code, _), .. })
        ] => code,
        _ => unreachable!()
    };

    ecx.parse_sess.registered_diagnostics.with_lock(|diagnostics| {
        match diagnostics.get_mut(&code) {
            // Previously used errors.
            Some(&mut ErrorInfo { description: _, use_site: Some(previous_span) }) => {
                ecx.struct_span_warn(span, &format!(
                    "diagnostic code {} already used", code
                )).span_note(previous_span, "previous invocation")
                  .emit();
            }
            // Newly used errors.
            Some(ref mut info) => {
                info.use_site = Some(span);
            }
            // Unregistered errors.
            None => {
                ecx.span_err(span, &format!(
                    "used diagnostic code {} not registered", code
                ));
            }
        }
    });
    MacEager::expr(ecx.expr_tuple(span, Vec::new()))
}

pub fn expand_register_diagnostic<'cx>(ecx: &'cx mut ExtCtxt<'_>,
                                       span: Span,
                                       token_tree: &[TokenTree])
                                       -> Box<dyn MacResult+'cx> {
    let (code, description) = match  token_tree {
        [
            TokenTree::Token(Token { kind: token::Ident(code, _), .. })
        ] => {
            (*code, None)
        },
        [
            TokenTree::Token(Token { kind: token::Ident(code, _), .. }),
            TokenTree::Token(Token { kind: token::Comma, .. }),
            TokenTree::Token(Token { kind: token::Literal(token::Lit { symbol, .. }), ..})
        ] => {
            (*code, Some(*symbol))
        },
        _ => unreachable!()
    };

    // Check that the description starts and ends with a newline and doesn't
    // overflow the maximum line width.
    description.map(|raw_msg| {
        let msg = raw_msg.as_str();
        if !msg.starts_with("\n") || !msg.ends_with("\n") {
            ecx.span_err(span, &format!(
                "description for error code {} doesn't start and end with a newline",
                code
            ));
        }

        // URLs can be unavoidably longer than the line limit, so we allow them.
        // Allowed format is: `[name]: https://www.rust-lang.org/`
        let is_url = |l: &str| l.starts_with("[") && l.contains("]:") && l.contains("http");

        if msg.lines().any(|line| line.len() > MAX_DESCRIPTION_WIDTH && !is_url(line)) {
            ecx.span_err(span, &format!(
                "description for error code {} contains a line longer than {} characters.\n\
                 if you're inserting a long URL use the footnote style to bypass this check.",
                code, MAX_DESCRIPTION_WIDTH
            ));
        }
    });
    // Add the error to the map.
    ecx.parse_sess.registered_diagnostics.with_lock(|diagnostics| {
        let info = ErrorInfo {
            description,
            use_site: None
        };
        if diagnostics.insert(code, info).is_some() {
            ecx.span_err(span, &format!(
                "diagnostic code {} already registered", code
            ));
        }
    });

    MacEager::items(smallvec![])
}

#[allow(deprecated)]
pub fn expand_build_diagnostic_array<'cx>(ecx: &'cx mut ExtCtxt<'_>,
                                          span: Span,
                                          token_tree: &[TokenTree])
                                          -> Box<dyn MacResult+'cx> {
    assert_eq!(token_tree.len(), 3);
    let (crate_name, ident) = match (&token_tree[0], &token_tree[2]) {
        (
            // Crate name.
            &TokenTree::Token(Token { kind: token::Ident(crate_name, _), .. }),
            // DIAGNOSTICS ident.
            &TokenTree::Token(Token { kind: token::Ident(name, _), span })
        ) => (crate_name, Ident::new(name, span)),
        _ => unreachable!()
    };

    // Output error metadata to `tmp/extended-errors/<target arch>/<crate name>.json`
    if let Ok(target_triple) = env::var("CFG_COMPILER_HOST_TRIPLE") {
        ecx.parse_sess.registered_diagnostics.with_lock(|diagnostics| {
            if let Err(e) = output_metadata(ecx,
                                            &target_triple,
                                            &crate_name.as_str(),
                                            diagnostics) {
                ecx.span_bug(span, &format!(
                    "error writing metadata for triple `{}` and crate `{}`, error: {}, \
                     cause: {:?}",
                    target_triple, crate_name, e.description(), e.cause()
                ));
            }
        });
    } else {
        ecx.span_err(span, &format!(
            "failed to write metadata for crate `{}` because $CFG_COMPILER_HOST_TRIPLE is not set",
            crate_name));
    }

    // Construct the output expression.
    let (count, expr) =
        ecx.parse_sess.registered_diagnostics.with_lock(|diagnostics| {
            let descriptions: Vec<P<ast::Expr>> =
                diagnostics.iter().filter_map(|(&code, info)| {
                    info.description.map(|description| {
                        ecx.expr_tuple(span, vec![
                            ecx.expr_str(span, code),
                            ecx.expr_str(span, description)
                        ])
                    })
                }).collect();
            (descriptions.len(), ecx.expr_vec(span, descriptions))
        });

    let static_ = ecx.lifetime(span, Ident::with_empty_ctxt(kw::StaticLifetime));
    let ty_str = ecx.ty_rptr(
        span,
        ecx.ty_ident(span, ecx.ident_of("str")),
        Some(static_),
        ast::Mutability::Immutable,
    );

    let ty = ecx.ty(
        span,
        ast::TyKind::Array(
            ecx.ty(
                span,
                ast::TyKind::Tup(vec![ty_str.clone(), ty_str])
            ),
            ast::AnonConst {
                id: ast::DUMMY_NODE_ID,
                value: ecx.expr_usize(span, count),
            },
        ),
    );

    MacEager::items(smallvec![
        P(ast::Item {
            ident,
            attrs: Vec::new(),
            id: ast::DUMMY_NODE_ID,
            node: ast::ItemKind::Const(
                ty,
                expr,
            ),
            vis: source_map::respan(span.shrink_to_lo(), ast::VisibilityKind::Public),
            span,
            tokens: None,
        })
    ])
}
