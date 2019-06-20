// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc;
extern crate rustc_plugin;

use std::borrow::ToOwned;
use syntax::ast;
use syntax::ext::build::AstBuilder;
use syntax::ext::base::{SyntaxExtension, SyntaxExtensionKind};
use syntax::ext::base::{TTMacroExpander, ExtCtxt, MacResult, MacEager};
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax_pos::Span;
use syntax::tokenstream::TokenStream;
use rustc_plugin::Registry;

struct Expander {
    args: Vec<ast::NestedMetaItem>,
}

impl TTMacroExpander for Expander {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   sp: Span,
                   _: TokenStream,
                   _: Option<Span>) -> Box<dyn MacResult+'cx> {
        let args = self.args.iter().map(|i| pprust::meta_list_item_to_string(i))
            .collect::<Vec<_>>().join(", ");
        MacEager::expr(ecx.expr_str(sp, Symbol::intern(&args)))
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    let args = reg.args().to_owned();
    reg.register_syntax_extension(Symbol::intern("plugin_args"), SyntaxExtension::default(
        SyntaxExtensionKind::LegacyBang(Box::new(Expander { args })), reg.sess.edition()
    ));
}
