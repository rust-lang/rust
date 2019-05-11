/// Module-level assembly support.
///
/// The macro defined here allows you to specify "top-level",
/// "file-scoped", or "module-level" assembly. These synonyms
/// all correspond to LLVM's module-level inline assembly instruction.
///
/// For example, `global_asm!("some assembly here")` codegens to
/// LLVM's `module asm "some assembly here"`. All of LLVM's caveats
/// therefore apply.

use errors::DiagnosticBuilder;

use syntax::ast;
use syntax::source_map::respan;
use syntax::ext::base::{self, *};
use syntax::feature_gate;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::symbol::{Symbol, sym};
use syntax_pos::Span;
use syntax::tokenstream;
use smallvec::smallvec;

pub const MACRO: Symbol = sym::global_asm;

pub fn expand_global_asm<'cx>(cx: &'cx mut ExtCtxt<'_>,
                              sp: Span,
                              tts: &[tokenstream::TokenTree]) -> Box<dyn base::MacResult + 'cx> {
    if !cx.ecfg.enable_global_asm() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       MACRO,
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_GLOBAL_ASM);
    }

    match parse_global_asm(cx, sp, tts) {
        Ok(Some(global_asm)) => {
            MacEager::items(smallvec![P(ast::Item {
                ident: ast::Ident::invalid(),
                attrs: Vec::new(),
                id: ast::DUMMY_NODE_ID,
                node: ast::ItemKind::GlobalAsm(P(global_asm)),
                vis: respan(sp.shrink_to_lo(), ast::VisibilityKind::Inherited),
                span: sp,
                tokens: None,
            })])
        }
        Ok(None) => DummyResult::any(sp),
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}

fn parse_global_asm<'a>(
    cx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: &[tokenstream::TokenTree]
) -> Result<Option<ast::GlobalAsm>, DiagnosticBuilder<'a>> {
    let mut p = cx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        let mut err = cx.struct_span_err(sp, "macro requires a string literal as an argument");
        err.span_label(sp, "string literal required");
        return Err(err);
    }

    let expr = p.parse_expr()?;
    let (asm, _) = match expr_to_string(cx, expr, "inline assembly must be a string literal") {
        Some((s, st)) => (s, st),
        None => return Ok(None),
    };

    Ok(Some(ast::GlobalAsm {
        asm,
        ctxt: cx.backtrace(),
    }))
}
