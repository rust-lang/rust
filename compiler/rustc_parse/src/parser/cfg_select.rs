use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::{AttrKind, token};
use rustc_errors::PResult;
use rustc_span::Span;

use crate::exp;
use crate::parser::{AttrWrapper, ForceCollect, Parser, Restrictions, Trailing, UsePreAttrPos};

#[derive(Default)]
pub struct CfgSelectBranchAttrSpans {
    pub attrs: Vec<Span>,
    pub doc_comments: Vec<Span>,
}

impl<'a> Parser<'a> {
    /// Parses the right-hand side of a `cfg_select!` branch,
    /// which can be either a braced block or an expression.
    pub fn parse_cfg_select_branch_rhs(&mut self) -> PResult<'a, TokenStream> {
        if self.token == token::OpenBrace {
            // Strip the outer '{' and '}'.
            match self.parse_token_tree() {
                TokenTree::Token(..) => unreachable!("because the current token is a '{{'"),
                TokenTree::Delimited(.., tts) => {
                    // Optionally end with a comma.
                    let _ = self.eat(exp!(Comma));
                    return Ok(tts);
                }
            }
        }
        let attrs = AttrWrapper::empty(); // FIXME expressions with attributes can be supported here
        let expr = self.collect_tokens(
            None,
            AttrWrapper::empty(),
            ForceCollect::Yes,
            |p, _empty_attrs| {
                p.parse_expr_res_after_attrs(Restrictions::STMT_EXPR, attrs)
                    .map(|(expr, _)| (expr, Trailing::No, UsePreAttrPos::No))
            },
        )?;
        if !classify::expr_is_complete(&expr)
            && self.token != token::CloseBrace
            && self.token != token::Eof
        {
            self.expect(exp!(Comma))?;
        } else {
            let _ = self.eat(exp!(Comma));
        }
        Ok(TokenStream::from_ast(&expr))
    }

    /// Parses outer attributes before a `cfg_select!` branch for recovery.
    pub fn parse_cfg_select_branch_outer_attrs(
        &mut self,
    ) -> PResult<'a, Option<CfgSelectBranchAttrSpans>> {
        let attrs = self.parse_outer_attributes()?;
        if attrs.is_empty() {
            return Ok(None);
        }

        let mut spans = CfgSelectBranchAttrSpans::default();
        for attr in attrs.take_for_recovery(self.psess) {
            match attr.kind {
                AttrKind::Normal(..) => spans.attrs.push(attr.span),
                AttrKind::Synthetic(..) => unreachable!(),
                // `parse_outer_attributes` already emitted E0753 for inner doc comments before
                // recovering them as outer doc-comment attributes.
                AttrKind::DocComment(comment_kind, _)
                    if self.span_to_snippet(attr.span).ok().is_some_and(
                        |snippet| match comment_kind {
                            token::CommentKind::Line => snippet.starts_with("//!"),
                            token::CommentKind::Block => snippet.starts_with("/*!"),
                        },
                    ) => {}
                AttrKind::DocComment(..) => spans.doc_comments.push(attr.span),
            }
        }

        Ok(Some(spans))
    }
}
