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
use std::os;
use std::dynamic_lib::DynamicLibrary;
use std::collections::HashSet;
use syntax::ast;
use syntax::attr;
use syntax::codemap::Span;
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
    pub args: P<ast::MetaItem>,
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
    for vi in krate.module.view_items.iter() {
        loader.span_whitelist.insert(vi.span);
    }

    visit::walk_crate(&mut loader, krate);

    if let Some(plugins) = addl_plugins {
        for plugin in plugins.iter() {
            loader.load_plugin(CrateOrString::Str(plugin.as_slice()),
                                                  None, None, None)
        }
    }

    return loader.plugins;
}

// note that macros aren't expanded yet, and therefore macros can't add plugins.
impl<'a, 'v> Visitor<'v> for PluginLoader<'a> {
    fn visit_view_item(&mut self, vi: &ast::ViewItem) {
        // We're only interested in `extern crate`.
        match vi.node {
            ast::ViewItemExternCrate(..) => (),
            _ => return,
        }

        // Parse the attributes relating to macro / plugin loading.
        let mut plugin_attr = None;
        let mut macro_selection = Some(HashSet::new());  // None => load all
        let mut reexport = HashSet::new();
        for attr in vi.attrs.iter() {
            let mut used = true;
            match attr.name().get() {
                "phase" => {
                    self.sess.span_err(attr.span, "#[phase] is deprecated; use \
                                       #[macro_use], #[plugin], and/or #[no_link]");
                }
                "plugin" => {
                    if plugin_attr.is_some() {
                        self.sess.span_err(attr.span, "#[plugin] specified multiple times");
                    }
                    plugin_attr = Some(attr.node.value.clone());
                }
                "macro_use" => {
                    let names = attr.meta_item_list();
                    if names.is_none() {
                        // no names => load all
                        macro_selection = None;
                    }
                    if let (Some(sel), Some(names)) = (macro_selection.as_mut(), names) {
                        for name in names.iter() {
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

                    for name in names.iter() {
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

        self.load_plugin(CrateOrString::Krate(vi), plugin_attr, macro_selection, Some(reexport))
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // bummer... can't see plugins inside macros.
        // do nothing.
    }
}

impl<'a> PluginLoader<'a> {
    pub fn load_plugin<'b>(&mut self,
                           c: CrateOrString<'b>,
                           plugin_attr: Option<P<ast::MetaItem>>,
                           macro_selection: Option<HashSet<token::InternedString>>,
                           reexport: Option<HashSet<token::InternedString>>) {
        let mut macros = vec![];
        let mut registrar = None;

        let load_macros = match (macro_selection.as_ref(), reexport.as_ref()) {
            (Some(sel), Some(re)) => sel.len() != 0 || re.len() != 0,
            _ => true,
        };
        let load_registrar = plugin_attr.is_some();

        if let CrateOrString::Krate(vi) = c {
            if load_macros && !self.span_whitelist.contains(&vi.span) {
                self.sess.span_err(vi.span, "an `extern crate` loading macros must be at \
                                             the crate root");
            }
       }

        if load_macros || load_registrar {
            let pmd = self.reader.read_plugin_metadata(c);
            if load_macros {
                macros = pmd.exported_macros();
            }
            if load_registrar {
                registrar = pmd.plugin_registrar();
            }
        }

        for mut def in macros.into_iter() {
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

        if let Some((lib, symbol)) = registrar {
            let fun = self.dylink_registrar(c, lib, symbol);
            self.plugins.registrars.push(PluginRegistrar {
                fun: fun,
                args: plugin_attr.unwrap(),
            });
        }
    }

    // Dynamically link a registrar function into the compiler process.
    fn dylink_registrar<'b>(&mut self,
                        c: CrateOrString<'b>,
                        path: Path,
                        symbol: String) -> PluginRegistrarFun {
        // Make sure the path contains a / or the linker will search for it.
        let path = os::make_absolute(&path).unwrap();

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
