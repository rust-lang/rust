//! Used by `rustc` when loading a plugin.

use crate::errors::{LoadPluginError, MalformedPluginAttribute};
use crate::Registry;
use libloading::Library;
use rustc_ast::Attribute;
use rustc_metadata::locator;
use rustc_session::cstore::MetadataLoader;
use rustc_session::Session;
use rustc_span::symbol::{sym, Ident};

use std::env;
use std::mem;
use std::path::PathBuf;

/// Pointer to a registrar function.
type PluginRegistrarFn = fn(&mut Registry<'_>);

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(
    sess: &Session,
    metadata_loader: &dyn MetadataLoader,
    attrs: &[Attribute],
) -> Vec<PluginRegistrarFn> {
    let mut plugins = Vec::new();

    for attr in attrs {
        if !attr.has_name(sym::plugin) {
            continue;
        }

        for plugin in attr.meta_item_list().unwrap_or_default() {
            match plugin.ident() {
                Some(ident) if plugin.is_word() => {
                    load_plugin(&mut plugins, sess, metadata_loader, ident)
                }
                _ => {
                    sess.emit_err(MalformedPluginAttribute { span: plugin.span() });
                }
            }
        }
    }

    plugins
}

fn load_plugin(
    plugins: &mut Vec<PluginRegistrarFn>,
    sess: &Session,
    metadata_loader: &dyn MetadataLoader,
    ident: Ident,
) {
    let lib = locator::find_plugin_registrar(sess, metadata_loader, ident.span, ident.name);
    let fun = dylink_registrar(lib).unwrap_or_else(|err| {
        // This is fatal: there are almost certainly macros we need inside this crate, so
        // continuing would spew "macro undefined" errors.
        sess.emit_fatal(LoadPluginError { span: ident.span, msg: err.to_string() });
    });
    plugins.push(fun);
}

/// Dynamically link a registrar function into the compiler process.
fn dylink_registrar(lib_path: PathBuf) -> Result<PluginRegistrarFn, libloading::Error> {
    // Make sure the path contains a / or the linker will search for it.
    let lib_path = env::current_dir().unwrap().join(&lib_path);

    let lib = unsafe { Library::new(&lib_path) }?;

    let registrar_sym = unsafe { lib.get::<PluginRegistrarFn>(b"__rustc_plugin_registrar") }?;

    // Intentionally leak the dynamic library. We can't ever unload it
    // since the library can make things that will live arbitrarily long
    // (e.g., an Rc cycle or a thread).
    let registrar_sym = unsafe { registrar_sym.into_raw() };
    mem::forget(lib);

    Ok(*registrar_sym)
}
