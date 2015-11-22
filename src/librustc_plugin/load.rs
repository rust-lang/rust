// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Used by `rustc` when loading a plugin.

use rustc::session::Session;
use rustc::metadata::creader::CrateReader;
use rustc::metadata::cstore::CStore;
use registry::Registry;

use std::borrow::ToOwned;
use std::env;
use std::mem;
use std::path::PathBuf;
use syntax::ast;
use syntax::codemap::{Span, COMMAND_LINE_SP};
use syntax::ptr::P;
use syntax::attr::AttrMetaMethods;

/// Pointer to a registrar function.
pub type PluginRegistrarFun =
    fn(&mut Registry);

pub struct PluginRegistrar {
    pub fun: PluginRegistrarFun,
    pub args: Vec<P<ast::MetaItem>>,
}

struct PluginLoader<'a> {
    sess: &'a Session,
    reader: CrateReader<'a>,
    plugins: Vec<PluginRegistrar>,
}

fn call_malformed_plugin_attribute(a: &Session, b: Span) {
    span_err!(a, b, E0498, "malformed plugin attribute");
}

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(sess: &Session, cstore: &CStore, krate: &ast::Crate,
                    addl_plugins: Option<Vec<String>>) -> Vec<PluginRegistrar> {
    let mut loader = PluginLoader::new(sess, cstore);

    for attr in &krate.attrs {
        if !attr.check_name("plugin") {
            continue;
        }

        let plugins = match attr.meta_item_list() {
            Some(xs) => xs,
            None => {
                call_malformed_plugin_attribute(sess, attr.span);
                continue;
            }
        };

        for plugin in plugins {
            if plugin.value_str().is_some() {
                call_malformed_plugin_attribute(sess, attr.span);
                continue;
            }

            let args = plugin.meta_item_list().map(ToOwned::to_owned).unwrap_or_default();
            loader.load_plugin(plugin.span, &*plugin.name(), args);
        }
    }

    if let Some(plugins) = addl_plugins {
        for plugin in plugins {
            loader.load_plugin(COMMAND_LINE_SP, &plugin, vec![]);
        }
    }

    loader.plugins
}

impl<'a> PluginLoader<'a> {
    fn new(sess: &'a Session, cstore: &'a CStore) -> PluginLoader<'a> {
        PluginLoader {
            sess: sess,
            reader: CrateReader::new(sess, cstore),
            plugins: vec![],
        }
    }

    fn load_plugin(&mut self, span: Span, name: &str, args: Vec<P<ast::MetaItem>>) {
        let registrar = self.reader.find_plugin_registrar(span, name);

        if let Some((lib, symbol)) = registrar {
            let fun = self.dylink_registrar(span, lib, symbol);
            self.plugins.push(PluginRegistrar {
                fun: fun,
                args: args,
            });
        }
    }

    // Dynamically link a registrar function into the compiler process.
    #[allow(deprecated)]
    fn dylink_registrar(&mut self,
                        span: Span,
                        path: PathBuf,
                        symbol: String) -> PluginRegistrarFun {
        use std::dynamic_lib::DynamicLibrary;

        // Make sure the path contains a / or the linker will search for it.
        let path = env::current_dir().unwrap().join(&path);

        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            // this is fatal: there are almost certainly macros we need
            // inside this crate, so continue would spew "macro undefined"
            // errors
            Err(err) => {
                self.sess.span_fatal(span, &err[..])
            }
        };

        unsafe {
            let registrar =
                match lib.symbol(&symbol[..]) {
                    Ok(registrar) => {
                        mem::transmute::<*mut u8,PluginRegistrarFun>(registrar)
                    }
                    // again fatal if we can't register macros
                    Err(err) => {
                        self.sess.span_fatal(span, &err[..])
                    }
                };

            // Intentionally leak the dynamic library. We can't ever unload it
            // since the library can make things that will live arbitrarily long
            // (e.g. an @-box cycle or a thread).
            mem::forget(lib);

            registrar
        }
    }
}
