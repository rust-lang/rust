#![allow(unused_imports, unused_variables)]

use rustc_ast::token;
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_errors::ErrorGuaranteed;
use rustc_expand::base::{AttrProcMacro, ExtCtxt};
use rustc_span::Span;
use rustc_span::symbol::{Ident, Symbol, kw, sym};

pub(crate) struct ExpandRequires;

pub(crate) struct ExpandEnsures;

impl AttrProcMacro for ExpandRequires {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        expand_requires_tts(ecx, span, annotation, annotated)
    }
}

impl AttrProcMacro for ExpandEnsures {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        expand_ensures_tts(ecx, span, annotation, annotated)
    }
}

fn expand_injecting_circa_where_clause(
    _ecx: &mut ExtCtxt<'_>,
    attr_span: Span,
    annotated: TokenStream,
    inject: impl FnOnce(&mut Vec<TokenTree>) -> Result<(), ErrorGuaranteed>,
) -> Result<TokenStream, ErrorGuaranteed> {
    let mut new_tts = Vec::with_capacity(annotated.len());
    let mut cursor = annotated.into_trees();

    // Find the `fn name<G,...>(x:X,...)` and inject the AST contract forms right after
    // the formal parameters (and return type if any).
    while let Some(tt) = cursor.next_ref() {
        new_tts.push(tt.clone());
        if let TokenTree::Token(tok, _) = tt
            && tok.is_ident_named(kw::Fn)
        {
            break;
        }
    }

    // Found the `fn` keyword, now find the formal parameters.
    //
    // FIXME: can this fail if you have parentheticals in a generics list, like `fn foo<F: Fn(X) -> Y>` ?
    while let Some(tt) = cursor.next_ref() {
        new_tts.push(tt.clone());

        if let TokenTree::Delimited(_, _, token::Delimiter::Parenthesis, _) = tt {
            break;
        }
        if let TokenTree::Token(token::Token { kind: token::TokenKind::Semi, .. }, _) = tt {
            panic!("contract attribute applied to fn without parameter list.");
        }
    }

    // There *might* be a return type declaration (and figuring out where that ends would require
    // parsing an arbitrary type expression, e.g. `-> Foo<args ...>`
    //
    // Instead of trying to figure that out, scan ahead and look for the first occurence of a
    // `where`, a `{ ... }`, or a `;`.
    //
    // FIXME: this might still fall into a trap for something like `-> Ctor<T, const { 0 }>`. I
    // *think* such cases must be under a Delimited (e.g. `[T; { N }]` or have the braced form
    // prefixed by e.g. `const`, so we should still be able to filter them out without having to
    // parse the type expression itself. But rather than try to fix things with hacks like that,
    // time might be better spent extending the attribute expander to suport tt-annotation atop
    // ast-annotated, which would be an elegant way to sidestep all of this.
    let mut opt_next_tt = cursor.next_ref();
    while let Some(next_tt) = opt_next_tt {
        if let TokenTree::Token(tok, _) = next_tt
            && tok.is_ident_named(kw::Where)
        {
            break;
        }
        if let TokenTree::Delimited(_, _, token::Delimiter::Brace, _) = next_tt {
            break;
        }
        if let TokenTree::Token(token::Token { kind: token::TokenKind::Semi, .. }, _) = next_tt {
            break;
        }

        // for anything else, transcribe the tt and keep looking.
        new_tts.push(next_tt.clone());
        opt_next_tt = cursor.next_ref();
        continue;
    }

    // At this point, we've transcribed everything from the `fn` through the formal parameter list
    // and return type declaration, (if any), but `tt` itself has *not* been transcribed.
    //
    // Now inject the AST contract form.
    //
    // FIXME: this kind of manual token tree munging does not have significant precedent among
    // rustc builtin macros, probably because most builtin macros use direct AST manipulation to
    // accomplish similar goals. But since our attributes need to take arbitrary expressions, and
    // our attribute infrastructure does not yet support mixing a token-tree annotation with an AST
    // annotated, we end up doing token tree manipulation.
    inject(&mut new_tts)?;

    // Above we injected the internal AST requires/ensures contruct. Now copy over all the other
    // token trees.
    if let Some(tt) = opt_next_tt {
        new_tts.push(tt.clone());
    }
    while let Some(tt) = cursor.next_ref() {
        new_tts.push(tt.clone());
    }

    Ok(TokenStream::new(new_tts))
}

fn expand_requires_tts(
    _ecx: &mut ExtCtxt<'_>,
    attr_span: Span,
    annotation: TokenStream,
    annotated: TokenStream,
) -> Result<TokenStream, ErrorGuaranteed> {
    expand_injecting_circa_where_clause(_ecx, attr_span, annotated, |new_tts| {
        new_tts.push(TokenTree::Token(
            token::Token::from_ast_ident(Ident::new(kw::RustcContractRequires, attr_span)),
            Spacing::Joint,
        ));
        new_tts.push(TokenTree::Token(
            token::Token::new(token::TokenKind::OrOr, attr_span),
            Spacing::Alone,
        ));
        new_tts.push(TokenTree::Delimited(
            DelimSpan::from_single(attr_span),
            DelimSpacing::new(Spacing::JointHidden, Spacing::JointHidden),
            token::Delimiter::Parenthesis,
            annotation,
        ));
        Ok(())
    })
}

fn expand_ensures_tts(
    _ecx: &mut ExtCtxt<'_>,
    attr_span: Span,
    annotation: TokenStream,
    annotated: TokenStream,
) -> Result<TokenStream, ErrorGuaranteed> {
    expand_injecting_circa_where_clause(_ecx, attr_span, annotated, |new_tts| {
        new_tts.push(TokenTree::Token(
            token::Token::from_ast_ident(Ident::new(kw::RustcContractEnsures, attr_span)),
            Spacing::Joint,
        ));
        new_tts.push(TokenTree::Delimited(
            DelimSpan::from_single(attr_span),
            DelimSpacing::new(Spacing::JointHidden, Spacing::JointHidden),
            token::Delimiter::Parenthesis,
            annotation,
        ));
        Ok(())
    })
}
