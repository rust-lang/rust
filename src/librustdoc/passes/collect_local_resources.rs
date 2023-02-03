//! This file will go through all doc comments and retrieve local resources to then store them
//! in the rustdoc output directory.

use pulldown_cmark::{Event, Parser, Tag};

use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::FileName;

use crate::clean::{Crate, Item};
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;
use crate::html::render::root_path;
use crate::html::LOCAL_RESOURCES_FOLDER_NAME;
use crate::passes::Pass;
use crate::visit::DocVisitor;

use std::path::{Path, PathBuf};

pub(crate) const COLLECT_LOCAL_RESOURCES: Pass = Pass {
    name: "collect-local-resources",
    run: collect_local_resources,
    description: "resolves intra-doc links",
};

fn span_file_path(cx: &DocContext<'_>, item: &Item) -> Option<PathBuf> {
    item.span(cx.tcx).and_then(|span| match span.filename(cx.sess()) {
        FileName::Real(ref path) => Some(path.local_path_if_available().into()),
        _ => None,
    })
}

struct ResourcesCollector<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
    /// The depth is used to know how many "../" needs to be generated to get the original file
    /// path.
    depth: usize,
}

fn collect_local_resources(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut collector = ResourcesCollector { cx, depth: 1 };
    collector.visit_crate(&krate);
    krate
}

impl<'a, 'tcx> ResourcesCollector<'a, 'tcx> {
    pub fn handle_event(
        &mut self,
        event: Event<'_>,
        current_path: &mut Option<PathBuf>,
        item: &Item,
    ) {
        if let Event::Start(Tag::Image(_, ref ori_path, _)) = event &&
            !ori_path.starts_with("http://") &&
            !ori_path.starts_with("https://")
        {
            let ori_path = ori_path.to_string();
            if self.cx.cache.local_resources.resources_correspondance
                .get(&self.depth)
                .and_then(|entry| entry.get(&ori_path))
                .is_some()
            {
                // We already have this entry so nothing to be done!
                return;
            }
            if current_path.is_none() {
                *current_path = span_file_path(self.cx, item);
            }
            let Some(current_path) = current_path else { return };

            let path = match current_path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&ori_path)
                .canonicalize()
            {
                Ok(p) => p,
                Err(_) => {
                    self.cx.tcx.sess.struct_span_err(
                        item.attr_span(self.cx.tcx),
                        &format!("`{ori_path}`: No such file"),
                    ).emit();
                    return;
                }
            };

            if !path.is_file() {
                self.cx.tcx.sess.struct_span_err(
                    item.attr_span(self.cx.tcx),
                    &format!("`{ori_path}`: No such file (expanded into `{}`)", path.display()),
                ).emit();
                return;
            }

            // We now enter the file into the `resources_to_copy` in case it's not already in
            // and then generate a path the file that we store into `resources_correspondance`
            // with the `add_entry_at_depth` method.
            let current_nb = self.cx.cache.local_resources.resources_to_copy.len();
            let file_name = self.cx.cache.local_resources.resources_to_copy
                .entry(path.clone())
                .or_insert_with(|| {
                    let extension = path.extension();
                    let (extension, dot) = match extension.and_then(|e| e.to_str()) {
                        Some(e) => (e, "."),
                        None => ("", ""),
                    };
                    format!(
                        "{current_nb}{}{dot}{extension}",
                        self.cx.render_options.resource_suffix,
                    )
                });
            let file = format!(
                "{}{LOCAL_RESOURCES_FOLDER_NAME}/{}/{file_name}",
                root_path(self.depth),
                self.cx.tcx.crate_name(LOCAL_CRATE).as_str(),
            );
            self.cx.cache.local_resources.add_entry_at_depth(self.depth, ori_path, file);
        }
    }
}

impl<'a, 'tcx> DocVisitor for ResourcesCollector<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        if let Some(md) = item.collapsed_doc_value() {
            let mut current_path = None;
            for event in Parser::new_ext(&md, main_body_opts()).into_iter() {
                self.handle_event(event, &mut current_path, item);
            }
        }

        if item.is_mod() && !item.is_crate() {
            self.depth += 1;
            self.visit_item_recur(item);
            self.depth -= 1;
        } else {
            self.visit_item_recur(item)
        }
    }
}
