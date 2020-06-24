use std::sync::Arc;

use rustc_span::edition::Edition;

use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;
use crate::formats::cache::{Cache, CACHE_KEY};

pub trait FormatRenderer: Clone {
    type Output: FormatRenderer;

    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        renderinfo: RenderInfo,
        edition: Edition,
        cache: &mut Cache,
    ) -> Result<(Self::Output, clean::Crate), Error>;

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item, cache: &Cache) -> Result<(), Error>;

    /// Renders a module (doesn't need to handle recursing into children).
    fn mod_item_in(
        &mut self,
        item: &clean::Item,
        item_name: &str,
        cache: &Cache,
    ) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self, name: &str) -> Result<(), Error>;

    /// Post processing hook for cleanup and dumping output to files.
    fn after_krate(&mut self, krate: &clean::Crate, cache: &Cache) -> Result<(), Error>;

    /// Called after everything else to write out errors.
    fn after_run(&mut self, diag: &rustc_errors::Handler) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct Renderer;

impl Renderer {
    pub fn new() -> Renderer {
        Renderer
    }

    /// Main method for rendering a crate.
    pub fn run<T: FormatRenderer + Clone>(
        self,
        krate: clean::Crate,
        options: RenderOptions,
        renderinfo: RenderInfo,
        diag: &rustc_errors::Handler,
        edition: Edition,
    ) -> Result<(), Error> {
        let (krate, mut cache) = Cache::from_krate(
            renderinfo.clone(),
            options.document_private,
            &options.extern_html_root_urls,
            &options.output,
            krate,
        );

        let (mut renderer, mut krate) = T::init(krate, options, renderinfo, edition, &mut cache)?;

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
        let mut work = vec![(renderer.clone(), item)];

        while let Some((mut cx, item)) = work.pop() {
            if item.is_mod() {
                // modules are special because they add a namespace. We also need to
                // recurse into the items of the module as well.
                let name = item.name.as_ref().unwrap().to_string();
                if name.is_empty() {
                    panic!("Unexpected module with empty name");
                }

                cx.mod_item_in(&item, &name, &cache)?;
                let module = match item.inner {
                    clean::StrippedItem(box clean::ModuleItem(m)) | clean::ModuleItem(m) => m,
                    _ => unreachable!(),
                };
                for it in module.items {
                    info!("Adding {:?} to worklist", it.name);
                    work.push((cx.clone(), it));
                }

                cx.mod_item_out(&name)?;
            } else if item.name.is_some() {
                cx.item(item, &cache)?;
            }
        }

        renderer.after_krate(&krate, &cache)?;
        renderer.after_run(diag)
    }
}
