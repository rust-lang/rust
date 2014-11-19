#![feature(globs, phase, plugin_registrar)] 


#[phase(plugin,link)]
extern crate syntax;
#[phase(plugin, link)]
extern crate rustc;



use rustc::plugin::Registry;
use rustc::lint::LintPassObject;

pub mod types;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    //reg.register_syntax_extension(intern("jstraceable"), base::ItemDecorator(box expand_jstraceable));
    //reg.register_macro("factorial", expand)
    reg.register_lint_pass(box types::TypePass as LintPassObject);
}