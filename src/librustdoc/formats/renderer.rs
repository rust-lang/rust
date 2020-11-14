use std::sync::Arc;

use rustc_span::edition::Edition;

use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;
use crate::formats::cache::{Cache, CACHE_KEY};

/// Allows for different backends to rustdoc to be used with the `run_format()` function. Each
/// backend renderer has hooks for initialization, documenting an item, entering and exiting a
/// module, and cleanup/finalizing output.
pub trait FormatRenderer: Clone {
    /// Sets up any state required for the renderer. When this is called the cache has already been
    /// populated.
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        render_info: RenderInfo,
        edition: Edition,
        cache: &mut Cache,
    ) -> Result<(Self, clean::Crate), Error>;

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item, cache: &Cache) -> Result<(), Error>;

    /// Renders a module (should not handle recursing into children).
    fn mod_item_in(
        &mut self,
        item: &clean::Item,
        item_name: &str,
        cache: &Cache,
    ) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self, item_name: &str) -> Result<(), Error>;

    /// Post processing hook for cleanup and dumping output to files.
    fn after_krate(&mut self, krate: &clean::Crate, cache: &Cache) -> Result<(), Error>;

    /// Called after everything else to write out errors.
    fn after_run(&mut self, diag: &rustc_errors::Handler) -> Result<(), Error>;
}

/// Main method for rendering a crate.
pub fn run_format<T: FormatRenderer>(
    krate: clean::Crate,
    options: RenderOptions,
    render_info: RenderInfo,
    diag: &rustc_errors::Handler,
    edition: Edition,
) -> Result<(), Error> {
    let (krate, mut cache) = Cache::from_krate(
        render_info.clone(),
        options.document_private,
        &options.extern_html_root_urls,
        &options.output,
        krate,
    );

    let (mut format_renderer, mut krate) =
        T::init(krate, options, render_info, edition, &mut cache)?;

    let cache = Arc::new(cache);
    // Freeze the cache now that the index has been built. Put an Arc into TLS for future
    // parallelization opportunities
    CACHE_KEY.with(|v| *v.borrow_mut() = cache.clone());

    let mut item = match krate.module.take() {
        Some(i) => i,
        None => return Ok(()),
    };

    item.name = Some(krate.name.clone());

    // Render the crate documentation
    let mut work = vec![(format_renderer.clone(), item)];

    while let Some((mut cx, item)) = work.pop() {
        if item.is_mod() {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            let name = item.name.as_ref().unwrap().to_string();
            if name.is_empty() {
                panic!("Unexpected module with empty name");
            }

            cx.mod_item_in(&item, &name, &cache)?;
            let module = match item.kind {
                clean::StrippedItem(box clean::ModuleItem(m)) | clean::ModuleItem(m) => m,
                _ => unreachable!(),
            };
            for it in module.items {
                debug!("Adding {:?} to worklist", it.name);
                work.push((cx.clone(), it));
            }

            cx.mod_item_out(&name)?;
        } else if item.name.is_some() {
            cx.item(item, &cache)?;
        }
    }

    format_renderer.after_krate(&krate, &cache)?;
    format_renderer.after_run(diag)
}
