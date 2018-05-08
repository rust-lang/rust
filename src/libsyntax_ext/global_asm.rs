// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
use syntax::codemap::respan;
use syntax::ext::base;
use syntax::ext::base::*;
use syntax::feature_gate;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;
use syntax::tokenstream;

use syntax::util::small_vector::SmallVector;

pub const MACRO: &'static str = "global_asm";

pub fn expand_global_asm<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[tokenstream::TokenTree]) -> Box<base::MacResult + 'cx> {
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

    MacEager::items(SmallVector::one(P(ast::Item {
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
    })))
}
