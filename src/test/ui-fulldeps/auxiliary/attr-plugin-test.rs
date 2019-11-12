// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc_driver;
extern crate syntax;
extern crate syntax_expand;

use rustc_driver::plugin::Registry;
use syntax_expand::base::SyntaxExtension;
use syntax::feature_gate::AttributeType;
use syntax::symbol::Symbol;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(
        Symbol::intern("mac"), SyntaxExtension::dummy_bang(reg.sess.edition())
    );
}
