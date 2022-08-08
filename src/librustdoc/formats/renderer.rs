use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

use crate::clean::{self, Item};
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

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item) -> Result<(), Error>;

    /// Runs before rendering an item (not a module)
    fn before_item(&mut self, _item: &clean::Item) -> Result<(), Error> {
        Ok(())
    }

    /// Runs after rendering an item (not a module)
    fn after_item(&mut self) -> Result<(), Error> {
        Ok(())
    }

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
        .extra_verbose_generic_activity("create_renderer", T::descr())
        .run(|| T::init(krate, options, cache, tcx))?;

    if !emit_crate {
        return Ok(());
    }

    enum WorkUnit {
        Module { item: Item, current_index: usize },
        Single(Item),
    }

    let mut work_units: Vec<WorkUnit> =
        vec![WorkUnit::Module { item: krate.module, current_index: 0 }];

    let unknown = Symbol::intern("<unknown item>");
    while let Some(work_unit) = work_units.pop() {
        match work_unit {
            WorkUnit::Module { item, current_index } if T::RUN_ON_MODULE => {
                let (clean::StrippedItem(box clean::ModuleItem(module)) | clean::ModuleItem(module)) = item.kind.as_ref()
                else { unreachable!() };

                if current_index == 0 {
                    // just enter the module
                    format_renderer.mod_item_in(&item)?;
                }

                if current_index < module.items.len() {
                    // get the next item
                    let next_item = module.items[current_index].clone();

                    // stay in the module
                    work_units.push(WorkUnit::Module { item, current_index: current_index + 1 });

                    // push the next item
                    if next_item.is_mod() {
                        work_units.push(WorkUnit::Module { item: next_item, current_index: 0 });
                    } else {
                        work_units.push(WorkUnit::Single(next_item));
                    }
                } else {
                    // the last item of the module has been rendered
                    // -> exit the module
                    format_renderer.mod_item_out()?;
                }
            }
            // FIXME: checking `item.name.is_some()` is very implicit and leads to lots of special
            // cases. Use an explicit match instead.
            WorkUnit::Module { item, .. } | WorkUnit::Single(item)
                if item.name.is_some() && !item.is_extern_crate() =>
            {
                // render the item
                prof.generic_activity_with_arg(
                    "render_item",
                    item.name.unwrap_or(unknown).as_str(),
                )
                .run(|| {
                    format_renderer.before_item(&item)?;
                    let result = format_renderer.item(item)?;
                    format_renderer.after_item()?;
                    Ok(result)
                })?;
            }
            _ => {}
        }
    }

    prof.extra_verbose_generic_activity("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate())
}
