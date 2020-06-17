use std::cell::RefCell;
use std::rc::Rc;

use rustc_span::edition::Edition;

use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;

pub trait FormatRenderer: Clone {
    type Output: FormatRenderer;

    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        renderinfo: RenderInfo,
        diag: &rustc_errors::Handler,
        edition: Edition,
        parent: Rc<RefCell<Renderer>>,
    ) -> Result<(Self::Output, clean::Crate), Error>;

    /// Renders a single non-module item. This means no recursive sub-item rendering is required.
    fn item(&mut self, item: clean::Item) -> Result<(), Error>;

    /// Renders a module. Doesn't need to handle recursing into children, the driver does that
    /// automatically.
    fn mod_item_in(
        &mut self,
        item: &clean::Item,
        item_name: &str,
        module: &clean::Module,
    ) -> Result<(), Error>;

    /// Runs after recursively rendering all sub-items of a module.
    fn mod_item_out(&mut self) -> Result<(), Error>;

    /// Post processing hook for cleanup and dumping output to files.
    fn after_krate(&mut self, krate: &clean::Crate) -> Result<(), Error>;

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
        let rself = Rc::new(RefCell::new(self));
        let (mut renderer, mut krate) =
            T::init(krate, options, renderinfo, diag, edition, rself.clone())?;
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

                let module = match item.inner {
                    clean::StrippedItem(box clean::ModuleItem(ref m))
                    | clean::ModuleItem(ref m) => m,
                    _ => unreachable!(),
                };
                cx.mod_item_in(&item, &name, module)?;
                let module = match item.inner {
                    clean::StrippedItem(box clean::ModuleItem(m)) | clean::ModuleItem(m) => m,
                    _ => unreachable!(),
                };
                for it in module.items {
                    info!("Adding {:?} to worklist", it.name);
                    work.push((cx.clone(), it));
                }

                cx.mod_item_out()?;
            } else if item.name.is_some() {
                cx.item(item)?;
            }
        }

        renderer.after_krate(&krate)?;
        renderer.after_run(diag)
    }
}
