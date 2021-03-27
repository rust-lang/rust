use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::ty::TyCtxt;
use rustc_span::{edition::Edition, Symbol};

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::clean;
use crate::config::RenderOptions;
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::html::render::print_item::item_path;

/// Allows for different backends to rustdoc to be used with the `run_format()` function. Each
/// backend renderer has hooks for initialization, documenting an item, entering and exiting a
/// module, and cleanup/finalizing output.
crate trait FormatRenderer<'tcx>: Sized {
    /// Gives a description of the renderer. Used for performance profiling.
    fn descr() -> &'static str;

    /// Whether to call `item` recursivly for modules
    ///
    /// This is true for html, and false for json. See #80664
    const RUN_ON_MODULE: bool;

    /// Sets up any state required for the renderer. When this is called the cache has already been
    /// populated.
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        edition: Edition,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
        case_insensitive_conflicts: Option<FxHashSet<String>>,
    ) -> Result<(Self, clean::Crate), Error>;

    /// Make a new renderer to render a child of the item currently being rendered.
    fn make_child_renderer(&self) -> Self;

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
        crate_name: Symbol,
        diag: &rustc_errors::Handler,
    ) -> Result<(), Error>;

    fn cache(&self) -> &Cache;
}

fn handle_module(dst: &Path, item: &clean::Item, paths_map: &mut FxHashMap<String, usize>) {
    // modules are special because they add a namespace. We also need to
    // recurse into the items of the module as well.
    let name = item.name.as_ref().unwrap().to_string();
    if name.is_empty() {
        panic!("Unexpected module with empty name");
    }
    let module = match *item.kind {
        clean::StrippedItem(box clean::ModuleItem(ref m)) | clean::ModuleItem(ref m) => m,
        _ => unreachable!(),
    };
    let mod_path = dst.join(&name);
    for it in &module.items {
        if it.is_mod() {
            handle_module(&mod_path, it, paths_map);
        } else if it.name.is_some() && !it.is_extern_crate() {
            let name = it.name.as_ref().unwrap();
            let item_type = it.type_();
            let file_name = &item_path(item_type, &name.as_str());
            let insensitive_path = mod_path.join(file_name).display().to_string();

            let entry = paths_map.entry(insensitive_path.to_lowercase()).or_insert(0);
            *entry += 1;
        }
    }
}

fn build_case_insensitive_map(
    krate: &clean::Crate,
    options: &RenderOptions,
) -> Option<FxHashSet<String>> {
    let mut paths_map: FxHashMap<String, usize> = FxHashMap::default();

    handle_module(&options.output, &krate.module, &mut paths_map);
    Some(paths_map.into_iter().filter(|(_, count)| *count > 1).map(|(path, _)| path).collect())
}

fn check_if_case_insensitive(dst: &Path) -> bool {
    fn create_and_write(dst: &Path, content: &str) {
        if let Ok(mut f) = fs::OpenOptions::new().write(true).create(true).truncate(true).open(dst)
        {
            // Ignoring potential errors.
            let _ = f.write(content.as_bytes());
        }
    }
    fn compare_content(dst: &Path, content: &str) -> bool {
        fs::read_to_string(dst).unwrap_or_else(|_| String::new()).as_str() == content
    }

    let path1 = dst.join("___a.tmp");
    let content1 = "a";
    let path2 = dst.join("___A.tmp");
    let content2 = "A";

    create_and_write(&path1, content1);
    create_and_write(&path1, content2);

    let res = compare_content(&path1, content1) && compare_content(&path2, content2);
    // We ignore the errors when removing the files.
    let _ = fs::remove_file(&path1);
    let _ = fs::remove_file(&path2);

    res
}

/// Main method for rendering a crate.
crate fn run_format<'tcx, T: FormatRenderer<'tcx>>(
    krate: clean::Crate,
    options: RenderOptions,
    cache: Cache,
    diag: &rustc_errors::Handler,
    edition: Edition,
    tcx: TyCtxt<'tcx>,
) -> Result<(), Error> {
    let prof = &tcx.sess.prof;

    let case_insensitive_conflicts =
        if options.generate_case_insensitive || check_if_case_insensitive(&options.output) {
            build_case_insensitive_map(&krate, &options)
        } else {
            None
        };

    let (mut format_renderer, krate) = prof
        .extra_verbose_generic_activity("create_renderer", T::descr())
        .run(|| T::init(krate, options, edition, cache, tcx, case_insensitive_conflicts))?;

    // Render the crate documentation
    let crate_name = krate.name;
    let mut work = vec![(format_renderer.make_child_renderer(), krate.module)];

    let unknown = Symbol::intern("<unknown item>");
    while let Some((mut cx, item)) = work.pop() {
        if item.is_mod() && T::RUN_ON_MODULE {
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
                work.push((cx.make_child_renderer(), it));
            }

            cx.mod_item_out(&name)?;
        // FIXME: checking `item.name.is_some()` is very implicit and leads to lots of special
        // cases. Use an explicit match instead.
        } else if item.name.is_some() && !item.is_extern_crate() {
            prof.generic_activity_with_arg("render_item", &*item.name.unwrap_or(unknown).as_str())
                .run(|| cx.item(item))?;
        }
    }
    prof.extra_verbose_generic_activity("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate(crate_name, diag))
}
