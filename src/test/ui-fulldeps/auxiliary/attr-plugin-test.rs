// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate syntax;

extern crate rustc;
extern crate rustc_plugin;

use syntax::feature_gate::AttributeType;
use rustc_plugin::Registry;



#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_attribute("foo".to_owned(), AttributeType::Normal);
    reg.register_attribute("bar".to_owned(), AttributeType::CrateLevel);
    reg.register_attribute("baz".to_owned(), AttributeType::Whitelisted);
}
