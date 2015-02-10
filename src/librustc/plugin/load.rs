// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Used by `rustc` when loading a plugin, or a crate with exported macros.

use session::Session;
use metadata::creader::{CrateOrString, CrateReader};
use plugin::registry::Registry;

use std::mem;
use std::env;
use std::dynamic_lib::DynamicLibrary;
use std::collections::HashSet;
use std::borrow::ToOwned;
use syntax::ast;
use syntax::attr;
use syntax::codemap::{Span, COMMAND_LINE_SP};
use syntax::parse::token;
use syntax::ptr::P;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::attr::AttrMetaMethods;

/// Pointer to a registrar function.
pub type PluginRegistrarFun =
    fn(&mut Registry);

pub struct PluginRegistrar {
    pub fun: PluginRegistrarFun,
    pub args: Vec<P<ast::MetaItem>>,
}

/// Information about loaded plugins.
pub struct Plugins {
    /// Imported macros.
    pub macros: Vec<ast::MacroDef>,
    /// Registrars, as function pointers.
    pub registrars: Vec<PluginRegistrar>,
}

pub struct PluginLoader<'a> {
    sess: &'a Session,
    span_whitelist: HashSet<Span>,
    reader: CrateReader<'a>,
    pub plugins: Plugins,
}

impl<'a> PluginLoader<'a> {
    fn new(sess: &'a Session) -> PluginLoader<'a> {
        PluginLoader {
            sess: sess,
            reader: CrateReader::new(sess),
            span_whitelist: HashSet::new(),
            plugins: Plugins {
                macros: vec!(),
                registrars: vec!(),
            },
        }
    }
}

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(sess: &Session, krate: &ast::Crate,
                    addl_plugins: Option<Vec<String>>) -> Plugins {
    let mut loader = PluginLoader::new(sess);

    // We need to error on `#[macro_use] extern crate` when it isn't at the
    // crate root, because `$crate` won't work properly. Identify these by
    // spans, because the crate map isn't set up yet.
    for item in &krate.module.items {
        if let ast::ItemExternCrate(_) = item.node {
            loader.span_whitelist.insert(item.span);
        }
    }

    visit::walk_crate(&mut loader, krate);

    for attr in &krate.attrs {
        if !attr.check_name("plugin") {
            continue;
        }

        let plugins = match attr.meta_item_list() {
            Some(xs) => xs,
            None => {
                sess.span_err(attr.span, "malformed plugin attribute");
                continue;
            }
        };

        for plugin in plugins {
            if plugin.value_str().is_some() {
                sess.span_err(attr.span, "malformed plugin attribute");
                continue;
            }

            let args = plugin.meta_item_list().map(ToOwned::to_owned).unwrap_or_default();
            loader.load_plugin(CrateOrString::Str(plugin.span, &*plugin.name()),
                               args);
        }
    }

    if let Some(plugins) = addl_plugins {
        for plugin in plugins {
            loader.load_plugin(CrateOrString::Str(COMMAND_LINE_SP, &plugin), vec![]);
        }
    }

    return loader.plugins;
}

// note that macros aren't expanded yet, and therefore macros can't add plugins.
impl<'a, 'v> Visitor<'v> for PluginLoader<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        // We're only interested in `extern crate`.
        match item.node {
            ast::ItemExternCrate(_) => {}
            _ => {
                visit::walk_item(self, item);
                return;
            }
        }

        // Parse the attributes relating to macro / plugin loading.
        let mut macro_selection = Some(HashSet::new());  // None => load all
        let mut reexport = HashSet::new();
        for attr in &item.attrs {
            let mut used = true;
            match &attr.name()[] {
                "phase" => {
                    self.sess.span_err(attr.span, "#[phase] is deprecated");
                }
                "plugin" => {
                    self.sess.span_err(attr.span, "#[plugin] on `extern crate` is deprecated");
                    self.sess.span_help(attr.span, &format!("use a crate attribute instead, \
                                                            i.e. #![plugin({})]",
                                                            item.ident.as_str())[]);
                }
                "macro_use" => {
                    let names = attr.meta_item_list();
                    if names.is_none() {
                        // no names => load all
                        macro_selection = None;
                    }
                    if let (Some(sel), Some(names)) = (macro_selection.as_mut(), names) {
                        for name in names {
                            if let ast::MetaWord(ref name) = name.node {
                                sel.insert(name.clone());
                            } else {
                                self.sess.span_err(name.span, "bad macro import");
                            }
                        }
                    }
                }
                "macro_reexport" => {
                    let names = match attr.meta_item_list() {
                        Some(names) => names,
                        None => {
                            self.sess.span_err(attr.span, "bad macro reexport");
                            continue;
                        }
                    };

                    for name in names {
                        if let ast::MetaWord(ref name) = name.node {
                            reexport.insert(name.clone());
                        } else {
                            self.sess.span_err(name.span, "bad macro reexport");
                        }
                    }
                }
                _ => used = false,
            }
            if used {
                attr::mark_used(attr);
            }
        }

        self.load_macros(item, macro_selection, Some(reexport))
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // bummer... can't see plugins inside macros.
        // do nothing.
    }
}

impl<'a> PluginLoader<'a> {
    pub fn load_macros<'b>(&mut self,
                           vi: &ast::Item,
                           macro_selection: Option<HashSet<token::InternedString>>,
                           reexport: Option<HashSet<token::InternedString>>) {
        if let (Some(sel), Some(re)) = (macro_selection.as_ref(), reexport.as_ref()) {
            if sel.is_empty() && re.is_empty() {
                return;
            }
        }

        if !self.span_whitelist.contains(&vi.span) {
            self.sess.span_err(vi.span, "an `extern crate` loading macros must be at \
                                         the crate root");
            return;
        }

        let pmd = self.reader.read_plugin_metadata(CrateOrString::Krate(vi));

        for mut def in pmd.exported_macros() {
            let name = token::get_ident(def.ident);
            def.use_locally = match macro_selection.as_ref() {
                None => true,
                Some(sel) => sel.contains(&name),
            };
            def.export = if let Some(ref re) = reexport {
                re.contains(&name)
            } else {
                false // Don't reexport macros from crates loaded from the command line
            };
            self.plugins.macros.push(def);
        }
    }

    pub fn load_plugin<'b>(&mut self,
                           c: CrateOrString<'b>,
                           args: Vec<P<ast::MetaItem>>) {
        let registrar = {
            let pmd = self.reader.read_plugin_metadata(c);
            pmd.plugin_registrar()
        };

        if let Some((lib, symbol)) = registrar {
            let fun = self.dylink_registrar(c, lib, symbol);
            self.plugins.registrars.push(PluginRegistrar {
                fun: fun,
                args: args,
            });
        }
    }

    // Dynamically link a registrar function into the compiler process.
    fn dylink_registrar<'b>(&mut self,
                        c: CrateOrString<'b>,
                        path: Path,
                        symbol: String) -> PluginRegistrarFun {
        // Make sure the path contains a / or the linker will search for it.
        let path = env::current_dir().unwrap().join(&path);

        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            // this is fatal: there are almost certainly macros we need
            // inside this crate, so continue would spew "macro undefined"
            // errors
            Err(err) => {
                if let CrateOrString::Krate(cr) = c {
                    self.sess.span_fatal(cr.span, &err[])
                } else {
                    self.sess.fatal(&err[])
                }
            }
        };

        unsafe {
            let registrar =
                match lib.symbol(&symbol[]) {
                    Ok(registrar) => {
                        mem::transmute::<*mut u8,PluginRegistrarFun>(registrar)
                    }
                    // again fatal if we can't register macros
                    Err(err) => {
                        if let CrateOrString::Krate(cr) = c {
                            self.sess.span_fatal(cr.span, &err[])
                        } else {
                            self.sess.fatal(&err[])
                        }
                    }
                };

            // Intentionally leak the dynamic library. We can't ever unload it
            // since the library can make things that will live arbitrarily long
            // (e.g. an @-box cycle or a task).
            mem::forget(lib);

            registrar
        }
    }
}
