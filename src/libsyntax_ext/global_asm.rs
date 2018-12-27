/// Module-level assembly support.
///
/// The macro defined here allows you to specify "top-level",
/// "file-scoped", or "module-level" assembly. These synonyms
/// all correspond to LLVM's module-level inline assembly instruction.
///
/// For example, `global_asm!("some assembly here")` codegens to
/// LLVM's `module asm "some assembly here"`. All of LLVM's caveats
/// therefore apply.

use syntax::ast;
use syntax::source_map::respan;
use syntax::ext::base;
use syntax::ext::base::*;
use syntax::feature_gate;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;
use syntax::tokenstream;

pub const MACRO: &str = "global_asm";

pub fn expand_global_asm<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[tokenstream::TokenTree]) -> Box<dyn base::MacResult + 'cx> {
    if !cx.ecfg.enable_global_asm() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       MACRO,
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_GLOBAL_ASM);
        return DummyResult::any(sp);
    }

    let mut p = cx.new_parser_from_tts(tts);
    let (asm, _) = match expr_to_string(cx,
                                        panictry!(p.parse_expr()),
                                        "inline assembly must be a string literal") {
        Some((s, st)) => (s, st),
        None => return DummyResult::any(sp),
    };

    MacEager::items(smallvec![P(ast::Item {
        ident: ast::Ident::with_empty_ctxt(Symbol::intern("")),
        attrs: Vec::new(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemKind::GlobalAsm(P(ast::GlobalAsm {
            asm,
            ctxt: cx.backtrace(),
        })),
        vis: respan(sp.shrink_to_lo(), ast::VisibilityKind::Inherited),
        span: sp,
        tokens: None,
    })])
}
