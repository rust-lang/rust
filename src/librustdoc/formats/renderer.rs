use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

use crate::clean;
use crate::config::RenderOptions;
use crate::error::Error;
use crate::formats::cache::Cache;

/// Allows for different backends to rustdoc to be used with the `run_format()` function. Each
/// backend renderer has hooks for initialization, documenting an item, entering and exiting a
/// module, and cleanup/finalizing output.
pub(crate) trait FormatRenderer<'tcx>: Sized {
    /// Gives a description of the renderer. Used for performance profiling.
    fn descr() -> &'static str;

    /// Whether to call `item` recursively for modules
    ///
    /// This is true for html, and false for json. See #80664
    const RUN_ON_MODULE: bool;

    /// Sets up any state required for the renderer. When this is called the cache has already been
    /// populated.
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error>;

    /// Make a new renderer to render a child of the item currently being rendered.
    fn make_child_renderer(&self) -> Self;

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item) -> Result<(), Error>;

    /// Renders a module (should not handle recursing into children).
    fn mod_item_in(&mut self, item: &clean::Item) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Post processing hook for cleanup and dumping output to files.
    fn after_krate(&mut self) -> Result<(), Error>;

    fn cache(&self) -> &Cache;
}

/// Main method for rendering a crate.
pub(crate) fn run_format<'tcx, T: FormatRenderer<'tcx>>(
    krate: clean::Crate,
    options: RenderOptions,
    cache: Cache,
    tcx: TyCtxt<'tcx>,
) -> Result<(), Error> {
    let prof = &tcx.sess.prof;

    let emit_crate = options.should_emit_crate();
    let (mut format_renderer, krate) = prof
        .verbose_generic_activity_with_arg("create_renderer", T::descr())
        .run(|| T::init(krate, options, cache, tcx))?;

    if !emit_crate {
        return Ok(());
    }

    // Render the crate documentation
    let mut work = vec![(format_renderer.make_child_renderer(), krate.module)];

    let unknown = Symbol::intern("<unknown item>");
    while let Some((mut cx, item)) = work.pop() {
        if item.is_mod() && T::RUN_ON_MODULE {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            let _timer =
                prof.generic_activity_with_arg("render_mod_item", item.name.unwrap().to_string());

            cx.mod_item_in(&item)?;
            let (clean::StrippedItem(box clean::ModuleItem(module)) | clean::ModuleItem(module)) =
                *item.kind
            else {
                unreachable!()
            };
            for it in module.items {
                debug!("Adding {:?} to worklist", it.name);
                work.push((cx.make_child_renderer(), it));
            }

            cx.mod_item_out()?;
        // FIXME: checking `item.name.is_some()` is very implicit and leads to lots of special
        // cases. Use an explicit match instead.
        } else if item.name.is_some() && !item.is_extern_crate() {
            prof.generic_activity_with_arg("render_item", item.name.unwrap_or(unknown).as_str())
                .run(|| cx.item(item))?;
        }
    }
    prof.verbose_generic_activity_with_arg("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate())
}
