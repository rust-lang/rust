//! Used by `rustc` when loading a plugin.

use crate::Registry;
use libloading::Library;
use rustc_ast::Crate;
use rustc_errors::struct_span_err;
use rustc_metadata::locator;
use rustc_session::cstore::MetadataLoader;
use rustc_session::Session;
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

use std::borrow::ToOwned;
use std::env;
use std::mem;
use std::path::PathBuf;

/// Pointer to a registrar function.
type PluginRegistrarFn = fn(&mut Registry<'_>);

fn call_malformed_plugin_attribute(sess: &Session, span: Span) {
    struct_span_err!(sess, span, E0498, "malformed `plugin` attribute")
        .span_label(span, "malformed attribute")
        .emit();
}

/// Read plugin metadata and dynamically load registrar functions.
pub fn load_plugins(
    sess: &Session,
    metadata_loader: &dyn MetadataLoader,
    krate: &Crate,
) -> Vec<PluginRegistrarFn> {
    let mut plugins = Vec::new();

    for attr in &krate.attrs {
        if !attr.has_name(sym::plugin) {
            continue;
        }

        for plugin in attr.meta_item_list().unwrap_or_default() {
            match plugin.ident() {
                Some(ident) if plugin.is_word() => {
                    load_plugin(&mut plugins, sess, metadata_loader, ident)
                }
                _ => call_malformed_plugin_attribute(sess, plugin.span()),
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
        sess.span_fatal(ident.span, &err.to_string());
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
