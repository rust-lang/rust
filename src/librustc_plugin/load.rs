//! Used by `rustc` when loading a plugin.

use rustc::middle::cstore::MetadataLoader;
use rustc::session::Session;
use rustc_metadata::locator;
use crate::registry::Registry;

use std::borrow::ToOwned;
use std::env;
use std::mem;
use std::path::PathBuf;
use syntax::ast;
use syntax::struct_span_err;
use syntax::symbol::{Symbol, kw, sym};
use syntax_pos::{Span, DUMMY_SP};

/// Pointer to a registrar function.
pub type PluginRegistrarFun =
    fn(&mut Registry<'_>);

pub struct PluginRegistrar {
    pub fun: PluginRegistrarFun,
    pub args: Vec<ast::NestedMetaItem>,
}

struct PluginLoader<'a> {
    sess: &'a Session,
    metadata_loader: &'a dyn MetadataLoader,
    plugins: Vec<PluginRegistrar>,
}

fn call_malformed_plugin_attribute(sess: &Session, span: Span) {
    struct_span_err!(sess, span, E0498, "malformed `plugin` attribute")
        .span_label(span, "malformed attribute")
        .emit();
}

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(sess: &Session,
                    metadata_loader: &dyn MetadataLoader,
                    krate: &ast::Crate,
                    addl_plugins: Option<Vec<String>>) -> Vec<PluginRegistrar> {
    let mut loader = PluginLoader { sess, metadata_loader, plugins: Vec::new() };

    // do not report any error now. since crate attributes are
    // not touched by expansion, every use of plugin without
    // the feature enabled will result in an error later...
    if sess.features_untracked().plugin {
        for attr in &krate.attrs {
            if !attr.check_name(sym::plugin) {
                continue;
            }

            let plugins = match attr.meta_item_list() {
                Some(xs) => xs,
                None => continue,
            };

            for plugin in plugins {
                // plugins must have a name and can't be key = value
                let name = plugin.name_or_empty();
                if name != kw::Invalid && !plugin.is_value_str() {
                    let args = plugin.meta_item_list().map(ToOwned::to_owned);
                    loader.load_plugin(plugin.span(), name, args.unwrap_or_default());
                } else {
                    call_malformed_plugin_attribute(sess, attr.span);
                }
            }
        }
    }

    if let Some(plugins) = addl_plugins {
        for plugin in plugins {
            loader.load_plugin(DUMMY_SP, Symbol::intern(&plugin), vec![]);
        }
    }

    loader.plugins
}

impl<'a> PluginLoader<'a> {
    fn load_plugin(&mut self, span: Span, name: Symbol, args: Vec<ast::NestedMetaItem>) {
        let registrar = locator::find_plugin_registrar(self.sess, self.metadata_loader, span, name);

        if let Some((lib, disambiguator)) = registrar {
            let symbol = self.sess.generate_plugin_registrar_symbol(disambiguator);
            let fun = self.dylink_registrar(span, lib, symbol);
            self.plugins.push(PluginRegistrar {
                fun,
                args,
            });
        }
    }

    // Dynamically link a registrar function into the compiler process.
    fn dylink_registrar(&mut self,
                        span: Span,
                        path: PathBuf,
                        symbol: String) -> PluginRegistrarFun {
        use rustc_metadata::dynamic_lib::DynamicLibrary;

        // Make sure the path contains a / or the linker will search for it.
        let path = env::current_dir().unwrap().join(&path);

        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            // this is fatal: there are almost certainly macros we need
            // inside this crate, so continue would spew "macro undefined"
            // errors
            Err(err) => {
                self.sess.span_fatal(span, &err)
            }
        };

        unsafe {
            let registrar =
                match lib.symbol(&symbol) {
                    Ok(registrar) => {
                        mem::transmute::<*mut u8,PluginRegistrarFun>(registrar)
                    }
                    // again fatal if we can't register macros
                    Err(err) => {
                        self.sess.span_fatal(span, &err)
                    }
                };

            // Intentionally leak the dynamic library. We can't ever unload it
            // since the library can make things that will live arbitrarily long
            // (e.g., an @-box cycle or a thread).
            mem::forget(lib);

            registrar
        }
    }
}
