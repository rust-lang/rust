use rustc_middle::ty::TyCtxt;
use rustc_span::edition::Edition;

use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;
use crate::formats::cache::Cache;

/// Allows for different backends to rustdoc to be used with the `run_format()` function. Each
/// backend renderer has hooks for initialization, documenting an item, entering and exiting a
/// module, and cleanup/finalizing output.
crate trait FormatRenderer<'tcx>: Clone {
    /// Gives a description of the renderer. Used for performance profiling.
    fn descr() -> &'static str;

    /// Sets up any state required for the renderer. When this is called the cache has already been
    /// populated.
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        render_info: RenderInfo,
        edition: Edition,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error>;

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item) -> Result<(), Error>;

    /// Renders a module (should not handle recursing into children).
    fn mod_item_in(&mut self, item: &clean::Item, item_name: &str) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self, item_name: &str) -> Result<(), Error>;

    /// Post processing hook for cleanup and dumping output to files.
    ///
    /// A handler is available if the renderer wants to report errors.
    fn after_krate(
        &mut self,
        krate: &clean::Crate,
        diag: &rustc_errors::Handler,
    ) -> Result<(), Error>;

    fn cache(&self) -> &Cache;
}

/// Main method for rendering a crate.
crate fn run_format<'tcx, T: FormatRenderer<'tcx>>(
    krate: clean::Crate,
    options: RenderOptions,
    render_info: RenderInfo,
    diag: &rustc_errors::Handler,
    edition: Edition,
    tcx: TyCtxt<'tcx>,
) -> Result<(), Error> {
    let (krate, cache) = tcx.sess.time("create_format_cache", || {
        Cache::from_krate(
            render_info.clone(),
            options.document_private,
            &options.extern_html_root_urls,
            &options.output,
            krate,
            tcx,
        )
    });
    let prof = &tcx.sess.prof;

    let (mut format_renderer, mut krate) = prof
        .extra_verbose_generic_activity("create_renderer", T::descr())
        .run(|| T::init(krate, options, render_info, edition, cache, tcx))?;

    let mut item = match krate.module.take() {
        Some(i) => i,
        None => return Ok(()),
    };

    item.name = Some(krate.name);

    // Render the crate documentation
    let mut work = vec![(format_renderer.clone(), item)];

    let unknown = rustc_span::Symbol::intern("<unknown item>");
    while let Some((mut cx, item)) = work.pop() {
        if item.is_mod() {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            let name = item.name.as_ref().unwrap().to_string();
            if name.is_empty() {
                panic!("Unexpected module with empty name");
            }
            let _timer = prof.generic_activity_with_arg("render_mod_item", name.as_str());

            cx.mod_item_in(&item, &name)?;
            let module = match *item.kind {
                clean::StrippedItem(box clean::ModuleItem(m)) | clean::ModuleItem(m) => m,
                _ => unreachable!(),
            };
            for it in module.items {
                debug!("Adding {:?} to worklist", it.name);
                work.push((cx.clone(), it));
            }

            cx.mod_item_out(&name)?;
        } else if item.name.is_some() {
            prof.generic_activity_with_arg("render_item", &*item.name.unwrap_or(unknown).as_str())
                .run(|| cx.item(item))?;
        }
    }
    prof.extra_verbose_generic_activity("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate(&krate, diag))
}
