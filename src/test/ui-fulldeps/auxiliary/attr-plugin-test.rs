// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate syntax;

extern crate rustc;
extern crate rustc_plugin;

use syntax::symbol::Symbol;
use syntax::feature_gate::AttributeType;
use rustc_plugin::Registry;


#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_attribute(Symbol::intern("foo"), AttributeType::Normal);
    reg.register_attribute(Symbol::intern("bar"), AttributeType::CrateLevel);
    reg.register_attribute(Symbol::intern("baz"), AttributeType::Whitelisted);
}
