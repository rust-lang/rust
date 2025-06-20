use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_middle::ty::TyCtxt;

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

    /// This associated type is the type where the current module information is stored.
    ///
    /// For each module, we go through their items by calling for each item:
    ///
    /// 1. `save_module_data`
    /// 2. `item`
    /// 3. `restore_module_data`
    ///
    /// This is because the `item` method might update information in `self` (for example if the child
    /// is a module). To prevent it from impacting the other children of the current module, we need to
    /// reset the information between each call to `item` by using `restore_module_data`.
    type ModuleData;

    /// Sets up any state required for the renderer. When this is called the cache has already been
    /// populated.
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error>;

    /// This method is called right before call [`Self::item`]. This method returns a type
    /// containing information that needs to be reset after the [`Self::item`] method has been
    /// called with the [`Self::restore_module_data`] method.
    ///
    /// In short it goes like this:
    ///
    /// ```ignore (not valid code)
    /// let reset_data = renderer.save_module_data();
    /// renderer.item(item)?;
    /// renderer.restore_module_data(reset_data);
    /// ```
    fn save_module_data(&mut self) -> Self::ModuleData;
    /// Used to reset current module's information.
    fn restore_module_data(&mut self, info: Self::ModuleData);

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: &clean::Item) -> Result<(), Error>;

    /// Renders a module (should not handle recursing into children).
    fn mod_item_in(&mut self, item: &clean::Item) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Post processing hook for cleanup and dumping output to files.
    fn after_krate(self) -> Result<(), Error>;
}

fn run_format_inner<'tcx, T: FormatRenderer<'tcx>>(
    cx: &mut T,
    item: &clean::Item,
    prof: &SelfProfilerRef,
) -> Result<(), Error> {
    if item.is_mod() && T::RUN_ON_MODULE {
        // modules are special because they add a namespace. We also need to
        // recurse into the items of the module as well.
        let _timer =
            prof.generic_activity_with_arg("render_mod_item", item.name.unwrap().to_string());

        cx.mod_item_in(&item)?;
        let (clean::StrippedItem(box clean::ModuleItem(ref module))
        | clean::ModuleItem(ref module)) = item.inner.kind
        else {
            unreachable!()
        };
        for it in module.items.iter() {
            let info = cx.save_module_data();
            run_format_inner(cx, it, prof)?;
            cx.restore_module_data(info);
        }

        cx.mod_item_out()?;
    // FIXME: checking `item.name.is_some()` is very implicit and leads to lots of special
    // cases. Use an explicit match instead.
    } else if let Some(item_name) = item.name
        && !item.is_extern_crate()
    {
        prof.generic_activity_with_arg("render_item", item_name.as_str()).run(|| cx.item(&item))?;
    }
    Ok(())
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
    run_format_inner(&mut format_renderer, &krate.module, prof)?;

    prof.verbose_generic_activity_with_arg("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate())
}
