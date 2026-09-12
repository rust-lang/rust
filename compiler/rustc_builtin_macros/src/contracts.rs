use rustc_ast::token;
use rustc_ast::token::Delimiter;
use rustc_ast::tokenarena::{
    ArenaTokenStream, ArenaTokenStreamBuilder, ArenaTokenTree, DelimitedData, PerTreeOp,
};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing};
use rustc_errors::ErrorGuaranteed;
use rustc_expand::base::{AttrProcMacro, ExtCtxt};
use rustc_span::Span;
use rustc_span::symbol::{Ident, Symbol, kw};

pub(crate) struct ExpandRequires;

pub(crate) struct ExpandEnsures;

impl AttrProcMacro for ExpandRequires {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: ArenaTokenStream,
        annotated: ArenaTokenStream,
    ) -> Result<ArenaTokenStream, ErrorGuaranteed> {
        expand_contract_clause_tts(ecx, span, annotation, annotated, kw::ContractRequires)
    }
}

impl AttrProcMacro for ExpandEnsures {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: ArenaTokenStream,
        annotated: ArenaTokenStream,
    ) -> Result<ArenaTokenStream, ErrorGuaranteed> {
        expand_contract_clause_tts(ecx, span, annotation, annotated, kw::ContractEnsures)
    }
}

/// Expand the function signature to include the contract clause.
///
/// The contracts clause will be injected before the function body and the optional where clause.
/// For that, we search for the body / where token, and invoke the `inject` callback to generate the
/// contract clause in the right place.
///
// FIXME: this kind of manual token tree munging does not have significant precedent among
// rustc builtin macros, probably because most builtin macros use direct AST manipulation to
// accomplish similar goals. But since our attributes need to take arbitrary expressions, and
// our attribute infrastructure does not yet support mixing a token-tree annotation with an AST
// annotated, we end up doing token tree manipulation.
fn expand_contract_clause(
    ecx: &mut ExtCtxt<'_>,
    attr_span: Span,
    annotated: ArenaTokenStream,
    inject: impl FnOnce(&mut ArenaTokenStreamBuilder) -> Result<(), ErrorGuaranteed>,
) -> Result<ArenaTokenStream, ErrorGuaranteed> {
    let mut builder = ArenaTokenStreamBuilder::with_capacity(annotated.length());
    let mut cursor = annotated.iter_top_level_trees();

    let is_kw = |tt: &ArenaTokenTree, sym: Symbol| {
        if let ArenaTokenTree::Token(token, _) = tt { token.is_ident_named(sym) } else { false }
    };

    // Find the `fn` keyword to check if this is a function.
    if cursor
        .find(|tt| {
            builder.push_token_tree(&tt.to_token_tree(&annotated));
            is_kw(tt, kw::Fn)
        })
        .is_none()
    {
        return Err(ecx
            .sess
            .dcx()
            .span_err(attr_span, "contract annotations can only be used on functions"));
    }

    // Contracts are not yet supported on async/gen functions
    if builder.tokens().iter().any(|tt| is_kw(tt, kw::Async) || is_kw(tt, kw::Gen)) {
        return Err(ecx.sess.dcx().span_err(
            attr_span,
            "contract annotations are not yet supported on async or gen functions",
        ));
    }

    // Found the `fn` keyword, now find either the `where` token or the function body.
    let next_tt = loop {
        let Some(tt) = cursor.next() else {
            return Err(ecx.sess.dcx().span_err(
                attr_span,
                "contract annotations is only supported in functions with bodies",
            ));
        };
        // If `tt` is the last element. Check if it is the function body.
        if cursor.peek().is_none() {
            if let ArenaTokenTree::DelimitedStart(
                _,
                DelimitedData { delimiter: token::Delimiter::Brace, .. },
            ) = tt
            {
                break tt;
            } else {
                return Err(ecx.sess.dcx().span_err(
                    attr_span,
                    "contract annotations is only supported in functions with bodies",
                ));
            }
        }

        if is_kw(tt, kw::Where) {
            break tt;
        }
        builder.push_token_tree(&tt.to_token_tree(&annotated));
    };

    // At this point, we've transcribed everything from the `fn` through the formal parameter list
    // and return type declaration, (if any), but `tt` itself has *not* been transcribed.
    //
    // Now inject the AST contract form.
    //
    inject(&mut builder)?;

    // Above we injected the internal AST requires/ensures construct. Now copy over all the other
    // token trees.
    builder.push_token_tree(&next_tt.to_token_tree(&annotated));
    while let Some(tt) = cursor.next() {
        builder.push_token_tree(&tt.to_token_tree(&annotated));
        if cursor.peek().is_none()
            && !matches!(
                tt,
                ArenaTokenTree::DelimitedStart(
                    _,
                    DelimitedData { delimiter: token::Delimiter::Brace, .. }
                )
            )
        {
            return Err(ecx.sess.dcx().span_err(
                attr_span,
                "contract annotations is only supported in functions with bodies",
            ));
        }
    }

    Ok(builder.finish())
}

fn expand_contract_clause_tts(
    ecx: &mut ExtCtxt<'_>,
    attr_span: Span,
    annotation: ArenaTokenStream,
    annotated: ArenaTokenStream,
    clause_keyword: rustc_span::Symbol,
) -> Result<ArenaTokenStream, ErrorGuaranteed> {
    if annotation.is_empty() {
        let (name, example) = if clause_keyword == kw::ContractRequires {
            ("requires", "condition")
        } else {
            ("ensures", "|result: &T| condition")
        };
        ecx.sess.dcx().span_err(
            attr_span,
            format!("`{name}` attribute requires an argument, e.g., `#[{name}({example})]`"),
        );
        // Returning `Err` would replace it with a dummy fragment and cause cascading name-resolution errors.
        // Instead, we return the original token stream so that there is no later noises.
        return Ok(annotated);
    }

    let feature_span = ecx.with_def_site_ctxt(attr_span);
    expand_contract_clause(ecx, attr_span, annotated, |builder| {
        builder.push_token(
            token::Token::from_ast_ident(Ident::new(clause_keyword, feature_span)),
            Spacing::Joint,
        );
        let start = builder.start_delimited();
        builder.build_from_stream(&annotation, |_, _| PerTreeOp::Continue);
        builder.close_delimited(
            start,
            DelimitedData {
                span: DelimSpan::from_single(attr_span),
                spacing: DelimSpacing::new(Spacing::JointHidden, Spacing::JointHidden),
                delimiter: Delimiter::Brace,
            },
        );
        Ok(())
    })
}
