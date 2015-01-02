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
use metadata::creader::CrateReader;
use plugin::registry::Registry;

use std::mem;
use std::os;
use std::dynamic_lib::DynamicLibrary;
use std::collections::HashSet;
use syntax::ast;
use syntax::attr;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::attr::AttrMetaMethods;

/// Pointer to a registrar function.
pub type PluginRegistrarFun =
    fn(&mut Registry);

/// Information about loaded plugins.
pub struct Plugins {
    /// Imported macros.
    pub macros: Vec<ast::MacroDef>,
    /// Registrars, as function pointers.
    pub registrars: Vec<PluginRegistrarFun>,
}

struct PluginLoader<'a> {
    sess: &'a Session,
    reader: CrateReader<'a>,
    plugins: Plugins,
}

impl<'a> PluginLoader<'a> {
    fn new(sess: &'a Session) -> PluginLoader<'a> {
        PluginLoader {
            sess: sess,
            reader: CrateReader::new(sess),
            plugins: Plugins {
                macros: vec!(),
                registrars: vec!(),
            },
        }
    }
}

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(sess: &Session, krate: &ast::Crate,
                    addl_plugins: Option<Plugins>) -> Plugins {
    let mut loader = PluginLoader::new(sess);
    visit::walk_crate(&mut loader, krate);

    let mut plugins = loader.plugins;

    match addl_plugins {
        Some(addl_plugins) => {
            // Add in the additional plugins requested by the frontend
            let Plugins { macros: addl_macros, registrars: addl_registrars } = addl_plugins;
            plugins.macros.extend(addl_macros.into_iter());
            plugins.registrars.extend(addl_registrars.into_iter());
        }
        None => ()
    }

    return plugins;
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
        let mut load_macros = false;
        let mut load_registrar = false;
        let mut reexport = HashSet::new();
        for attr in vi.attrs.iter() {
            let mut used = true;
            match attr.name().get() {
                "phase" => {
                    self.sess.span_err(attr.span, "#[phase] is deprecated; use \
                                       #[macro_use], #[plugin], and/or #[no_link]");
                }
                "plugin" => load_registrar = true,
                "macro_use" => load_macros = true,
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

        let mut macros = vec![];
        let mut registrar = None;

        if load_macros || load_registrar {
            let pmd = self.reader.read_plugin_metadata(vi);
            if load_macros {
                macros = pmd.exported_macros();
            }
            if load_registrar {
                registrar = pmd.plugin_registrar();
            }
        }

        for mut def in macros.into_iter() {
            if reexport.contains(&token::get_ident(def.ident)) {
                def.export = true;
            }
            self.plugins.macros.push(def);
        }

        if let Some((lib, symbol)) = registrar {
            self.dylink_registrar(vi, lib, symbol);
        }
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // bummer... can't see plugins inside macros.
        // do nothing.
    }
}

impl<'a> PluginLoader<'a> {
    // Dynamically link a registrar function into the compiler process.
    fn dylink_registrar(&mut self, vi: &ast::ViewItem, path: Path, symbol: String) {
        // Make sure the path contains a / or the linker will search for it.
        let path = os::make_absolute(&path).unwrap();

        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            // this is fatal: there are almost certainly macros we need
            // inside this crate, so continue would spew "macro undefined"
            // errors
            Err(err) => self.sess.span_fatal(vi.span, err[])
        };

        unsafe {
            let registrar =
                match lib.symbol(symbol[]) {
                    Ok(registrar) => {
                        mem::transmute::<*mut u8,PluginRegistrarFun>(registrar)
                    }
                    // again fatal if we can't register macros
                    Err(err) => self.sess.span_fatal(vi.span, err[])
                };

            self.plugins.registrars.push(registrar);

            // Intentionally leak the dynamic library. We can't ever unload it
            // since the library can make things that will live arbitrarily long
            // (e.g. an @-box cycle or a task).
            mem::forget(lib);

        }
    }
}
