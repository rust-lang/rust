use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::{AttrKind, AttrStyle, token};
use rustc_errors::PResult;
use rustc_span::Span;

use crate::exp;
use crate::parser::attr::InnerAttrPolicy;
use crate::parser::{AttrWrapper, ForceCollect, Parser, Restrictions, Trailing, UsePreAttrPos};

#[derive(Default)]
pub struct CfgSelectBranchAttrSpans {
    pub attrs: Vec<Span>,
    pub doc_comments: Vec<Span>,
}

impl<'a> Parser<'a> {
    /// Parses a `TokenTree` consisting either of `{ /* ... */ }` optionally followed by a comma
    /// (and strip the braces and the optional comma) or an expression followed by a comma
    /// (and strip the comma).
    pub fn parse_delimited_token_tree(&mut self) -> PResult<'a, TokenStream> {
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
        let expr = self.collect_tokens(None, AttrWrapper::empty(), ForceCollect::Yes, |p, _| {
            p.parse_expr_res(Restrictions::STMT_EXPR, AttrWrapper::empty())
                .map(|(expr, _)| (expr, Trailing::No, UsePreAttrPos::No))
        })?;
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
        // `parse_outer_attributes` recovers inner doc comments as outer ones, so collect their
        // spans first and suppress the follow-up `cfg_select` branch diagnostic for them.
        let mut inner_doc_comment_spans = Vec::new();
        let mut snapshot = self.create_snapshot_for_diagnostic();
        loop {
            if snapshot.check(exp!(Pound)) {
                if let Err(err) = snapshot.parse_attribute(InnerAttrPolicy::Permitted) {
                    err.cancel();
                    break;
                }
            } else if let token::DocComment(_, attr_style, _) = snapshot.token.kind {
                if attr_style == AttrStyle::Inner {
                    inner_doc_comment_spans.push(snapshot.token.span);
                }
                snapshot.bump();
            } else {
                break;
            }
        }

        let attrs = self.parse_outer_attributes()?;
        if attrs.is_empty() {
            return Ok(None);
        }

        let mut spans = CfgSelectBranchAttrSpans::default();
        for attr in attrs.take_for_recovery(self.psess) {
            match attr.kind {
                AttrKind::Normal(..) => spans.attrs.push(attr.span),
                // `parse_outer_attributes` already emitted E0753 for each inner doc comment
                // before recovering it as an outer doc-comment attribute.
                AttrKind::DocComment(..) if inner_doc_comment_spans.contains(&attr.span) => {}
                AttrKind::DocComment(..) => spans.doc_comments.push(attr.span),
            }
        }

        Ok(Some(spans))
    }
}
