// ignore-tidy-filelength

//! Rustdoc's HTML rendering module.
//!
//! This modules contains the bulk of the logic necessary for rendering a
//! rustdoc `clean::Crate` instance to a set of static HTML pages. This
//! rendering process is largely driven by the `format!` syntax extension to
//! perform all I/O into files and streams.
//!
//! The rendering process is largely driven by the `Context` and `Cache`
//! structures. The cache is pre-populated by crawling the crate in question,
//! and then it is shared among the various rendering threads. The cache is meant
//! to be a fairly large structure not implementing `Clone` (because it's shared
//! among threads). The context, however, should be a lightweight structure. This
//! is cloned per-thread and contains information about what is currently being
//! rendered.
//!
//! In order to speed up rendering (mostly because of markdown rendering), the
//! rendering process has been parallelized. This parallelization is only
//! exposed through the `crate` method on the context, and then also from the
//! fact that the shared cache is stored in TLS (and must be accessed as such).
//!
//! In addition to rendering the crate itself, this module is also responsible
//! for creating the corresponding search index and source file renderings.
//! These threads are not parallelized (they haven't been a bottleneck yet), and
//! both occur before the crate is rendered.

crate mod cache;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::default::Default;
use std::ffi::OsStr;
use std::fmt::{self, Write};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::str;
use std::string::ToString;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use itertools::Itertools;
use rustc_ast_pretty::pprust;
use rustc_attr::{Deprecation, StabilityLevel};
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::Mutability;
use rustc_middle::middle::stability;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::FileName;
use rustc_span::symbol::{kw, sym, Symbol};
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use crate::clean::{self, AttributesExt, GetDefId, RenderedLink, SelfTy, TypeKind};
use crate::config::{RenderInfo, RenderOptions};
use crate::docfs::{DocFS, PathError};
use crate::doctree;
use crate::error::Error;
use crate::formats::cache::{cache, Cache};
use crate::formats::item_type::ItemType;
use crate::formats::{AssocItemRender, FormatRenderer, Impl, RenderMode};
use crate::html::escape::Escape;
use crate::html::format::fmt_impl_for_trait_page;
use crate::html::format::Function;
use crate::html::format::{href, print_default_space, print_generic_bounds, WhereClause};
use crate::html::format::{print_abi_with_space, Buffer, PrintWithSpace};
use crate::html::markdown::{
    self, plain_text_summary, ErrorCodes, IdMap, Markdown, MarkdownHtml, MarkdownSummaryLine,
};
use crate::html::sources;
use crate::html::{highlight, layout, static_files};
use cache::{build_index, ExternalLocation};

/// A pair of name and its optional document.
crate type NameDoc = (String, Option<String>);

crate fn ensure_trailing_slash(v: &str) -> impl fmt::Display + '_ {
    crate::html::format::display_fn(move |f| {
        if !v.ends_with('/') && !v.is_empty() { write!(f, "{}/", v) } else { write!(f, "{}", v) }
    })
}

/// Major driving force in all rustdoc rendering. This contains information
/// about where in the tree-like hierarchy rendering is occurring and controls
/// how the current page is being rendered.
///
/// It is intended that this context is a lightweight object which can be fairly
/// easily cloned because it is cloned per work-job (about once per item in the
/// rustdoc tree).
#[derive(Clone)]
crate struct Context<'tcx> {
    /// Current hierarchy of components leading down to what's currently being
    /// rendered
    crate current: Vec<String>,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    crate dst: PathBuf,
    /// A flag, which when `true`, will render pages which redirect to the
    /// real location of an item. This is used to allow external links to
    /// publicly reused items to redirect to the right location.
    crate render_redirect_pages: bool,
    /// The map used to ensure all generated 'id=' attributes are unique.
    id_map: Rc<RefCell<IdMap>>,
    crate shared: Arc<SharedContext<'tcx>>,
    all: Rc<RefCell<AllTypes>>,
    /// Storage for the errors produced while generating documentation so they
    /// can be printed together at the end.
    crate errors: Rc<Receiver<String>>,
}

crate struct SharedContext<'tcx> {
    crate tcx: TyCtxt<'tcx>,
    /// The path to the crate root source minus the file name.
    /// Used for simplifying paths to the highlighted source code files.
    crate src_root: PathBuf,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon and added html).
    crate layout: layout::Layout,
    /// This flag indicates whether `[src]` links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    crate include_sources: bool,
    /// The local file sources we've emitted and their respective url-paths.
    crate local_sources: FxHashMap<PathBuf, String>,
    /// Whether the collapsed pass ran
    crate collapsed: bool,
    /// The base-URL of the issue tracker for when an item has been tagged with
    /// an issue number.
    crate issue_tracker_base_url: Option<String>,
    /// The directories that have already been created in this doc run. Used to reduce the number
    /// of spurious `create_dir_all` calls.
    crate created_dirs: RefCell<FxHashSet<PathBuf>>,
    /// This flag indicates whether listings of modules (in the side bar and documentation itself)
    /// should be ordered alphabetically or in order of appearance (in the source code).
    crate sort_modules_alphabetically: bool,
    /// Additional CSS files to be added to the generated docs.
    crate style_files: Vec<StylePath>,
    /// Suffix to be added on resource files (if suffix is "-v2" then "light.css" becomes
    /// "light-v2.css").
    crate resource_suffix: String,
    /// Optional path string to be used to load static files on output pages. If not set, uses
    /// combinations of `../` to reach the documentation root.
    crate static_root_path: Option<String>,
    /// The fs handle we are working with.
    crate fs: DocFS,
    /// The default edition used to parse doctests.
    crate edition: Edition,
    crate codes: ErrorCodes,
    playground: Option<markdown::Playground>,
}

impl<'tcx> Context<'tcx> {
    fn path(&self, filename: &str) -> PathBuf {
        // We use splitn vs Path::extension here because we might get a filename
        // like `style.min.css` and we want to process that into
        // `style-suffix.min.css`.  Path::extension would just return `css`
        // which would result in `style.min-suffix.css` which isn't what we
        // want.
        let (base, ext) = filename.split_once('.').unwrap();
        let filename = format!("{}{}.{}", base, self.shared.resource_suffix, ext);
        self.dst.join(&filename)
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.shared.tcx
    }

    fn sess(&self) -> &Session {
        &self.shared.tcx.sess
    }
}

impl SharedContext<'_> {
    crate fn ensure_dir(&self, dst: &Path) -> Result<(), Error> {
        let mut dirs = self.created_dirs.borrow_mut();
        if !dirs.contains(dst) {
            try_err!(self.fs.create_dir_all(dst), dst);
            dirs.insert(dst.to_path_buf());
        }

        Ok(())
    }

    /// Based on whether the `collapse-docs` pass was run, return either the `doc_value` or the
    /// `collapsed_doc_value` of the given item.
    crate fn maybe_collapsed_doc_value<'a>(&self, item: &'a clean::Item) -> Option<Cow<'a, str>> {
        if self.collapsed {
            item.collapsed_doc_value().map(|s| s.into())
        } else {
            item.doc_value().map(|s| s.into())
        }
    }
}

// Helper structs for rendering items/sidebars and carrying along contextual
// information

/// Struct representing one entry in the JS search index. These are all emitted
/// by hand to a large JS file at the end of cache-creation.
#[derive(Debug)]
crate struct IndexItem {
    crate ty: ItemType,
    crate name: String,
    crate path: String,
    crate desc: String,
    crate parent: Option<DefId>,
    crate parent_idx: Option<usize>,
    crate search_type: Option<IndexItemFunctionType>,
}

impl Serialize for IndexItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        assert_eq!(
            self.parent.is_some(),
            self.parent_idx.is_some(),
            "`{}` is missing idx",
            self.name
        );

        (self.ty, &self.name, &self.path, &self.desc, self.parent_idx, &self.search_type)
            .serialize(serializer)
    }
}

/// A type used for the search index.
#[derive(Debug)]
crate struct RenderType {
    ty: Option<DefId>,
    idx: Option<usize>,
    name: Option<String>,
    generics: Option<Vec<Generic>>,
}

impl Serialize for RenderType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(name) = &self.name {
            let mut seq = serializer.serialize_seq(None)?;
            if let Some(id) = self.idx {
                seq.serialize_element(&id)?;
            } else {
                seq.serialize_element(&name)?;
            }
            if let Some(generics) = &self.generics {
                seq.serialize_element(&generics)?;
            }
            seq.end()
        } else {
            serializer.serialize_none()
        }
    }
}

/// A type used for the search index.
#[derive(Debug)]
crate struct Generic {
    name: String,
    defid: Option<DefId>,
    idx: Option<usize>,
}

impl Serialize for Generic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(id) = self.idx {
            serializer.serialize_some(&id)
        } else {
            serializer.serialize_some(&self.name)
        }
    }
}

/// Full type of functions/methods in the search index.
#[derive(Debug)]
crate struct IndexItemFunctionType {
    inputs: Vec<TypeWithKind>,
    output: Option<Vec<TypeWithKind>>,
}

impl Serialize for IndexItemFunctionType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // If we couldn't figure out a type, just write `null`.
        let mut iter = self.inputs.iter();
        if match self.output {
            Some(ref output) => iter.chain(output.iter()).any(|ref i| i.ty.name.is_none()),
            None => iter.any(|ref i| i.ty.name.is_none()),
        } {
            serializer.serialize_none()
        } else {
            let mut seq = serializer.serialize_seq(None)?;
            seq.serialize_element(&self.inputs)?;
            if let Some(output) = &self.output {
                if output.len() > 1 {
                    seq.serialize_element(&output)?;
                } else {
                    seq.serialize_element(&output[0])?;
                }
            }
            seq.end()
        }
    }
}

#[derive(Debug)]
crate struct TypeWithKind {
    ty: RenderType,
    kind: TypeKind,
}

impl From<(RenderType, TypeKind)> for TypeWithKind {
    fn from(x: (RenderType, TypeKind)) -> TypeWithKind {
        TypeWithKind { ty: x.0, kind: x.1 }
    }
}

impl Serialize for TypeWithKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.ty.name)?;
        let x: ItemType = self.kind.into();
        seq.serialize_element(&x)?;
        seq.end()
    }
}

#[derive(Debug, Clone)]
crate struct StylePath {
    /// The path to the theme
    crate path: PathBuf,
    /// What the `disabled` attribute should be set to in the HTML tag
    crate disabled: bool,
}

thread_local!(crate static CURRENT_DEPTH: Cell<usize> = Cell::new(0));

crate fn initial_ids() -> Vec<String> {
    [
        "main",
        "search",
        "help",
        "TOC",
        "render-detail",
        "associated-types",
        "associated-const",
        "required-methods",
        "provided-methods",
        "implementors",
        "synthetic-implementors",
        "implementors-list",
        "synthetic-implementors-list",
        "methods",
        "deref-methods",
        "implementations",
    ]
    .iter()
    .map(|id| (String::from(*id)))
    .collect()
}

/// Generates the documentation for `crate` into the directory `dst`
impl<'tcx> FormatRenderer<'tcx> for Context<'tcx> {
    fn init(
        mut krate: clean::Crate,
        options: RenderOptions,
        _render_info: RenderInfo,
        edition: Edition,
        cache: &mut Cache,
        tcx: ty::TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error> {
        // need to save a copy of the options for rendering the index page
        let md_opts = options.clone();
        let RenderOptions {
            output,
            external_html,
            id_map,
            playground_url,
            sort_modules_alphabetically,
            themes: style_files,
            default_settings,
            extension_css,
            resource_suffix,
            static_root_path,
            generate_search_filter,
            unstable_features,
            ..
        } = options;

        let src_root = match krate.src {
            FileName::Real(ref p) => match p.local_path().parent() {
                Some(p) => p.to_path_buf(),
                None => PathBuf::new(),
            },
            _ => PathBuf::new(),
        };
        // If user passed in `--playground-url` arg, we fill in crate name here
        let mut playground = None;
        if let Some(url) = playground_url {
            playground =
                Some(markdown::Playground { crate_name: Some(krate.name.to_string()), url });
        }
        let mut layout = layout::Layout {
            logo: String::new(),
            favicon: String::new(),
            external_html,
            default_settings,
            krate: krate.name.to_string(),
            css_file_extension: extension_css,
            generate_search_filter,
        };
        let mut issue_tracker_base_url = None;
        let mut include_sources = true;

        // Crawl the crate attributes looking for attributes which control how we're
        // going to emit HTML
        if let Some(attrs) = krate.module.as_ref().map(|m| &m.attrs) {
            for attr in attrs.lists(sym::doc) {
                match (attr.name_or_empty(), attr.value_str()) {
                    (sym::html_favicon_url, Some(s)) => {
                        layout.favicon = s.to_string();
                    }
                    (sym::html_logo_url, Some(s)) => {
                        layout.logo = s.to_string();
                    }
                    (sym::html_playground_url, Some(s)) => {
                        playground = Some(markdown::Playground {
                            crate_name: Some(krate.name.to_string()),
                            url: s.to_string(),
                        });
                    }
                    (sym::issue_tracker_base_url, Some(s)) => {
                        issue_tracker_base_url = Some(s.to_string());
                    }
                    (sym::html_no_source, None) if attr.is_word() => {
                        include_sources = false;
                    }
                    _ => {}
                }
            }
        }
        let (sender, receiver) = channel();
        let mut scx = SharedContext {
            tcx,
            collapsed: krate.collapsed,
            src_root,
            include_sources,
            local_sources: Default::default(),
            issue_tracker_base_url,
            layout,
            created_dirs: Default::default(),
            sort_modules_alphabetically,
            style_files,
            resource_suffix,
            static_root_path,
            fs: DocFS::new(sender),
            edition,
            codes: ErrorCodes::from(unstable_features.is_nightly_build()),
            playground,
        };

        // Add the default themes to the `Vec` of stylepaths
        //
        // Note that these must be added before `sources::render` is called
        // so that the resulting source pages are styled
        //
        // `light.css` is not disabled because it is the stylesheet that stays loaded
        // by the browser as the theme stylesheet. The theme system (hackily) works by
        // changing the href to this stylesheet. All other themes are disabled to
        // prevent rule conflicts
        scx.style_files.push(StylePath { path: PathBuf::from("light.css"), disabled: false });
        scx.style_files.push(StylePath { path: PathBuf::from("dark.css"), disabled: true });
        scx.style_files.push(StylePath { path: PathBuf::from("ayu.css"), disabled: true });

        let dst = output;
        scx.ensure_dir(&dst)?;
        krate = sources::render(&dst, &mut scx, krate)?;

        // Build our search index
        let index = build_index(&krate, cache);

        let cache = Arc::new(cache);
        let mut cx = Context {
            current: Vec::new(),
            dst,
            render_redirect_pages: false,
            id_map: Rc::new(RefCell::new(id_map)),
            shared: Arc::new(scx),
            all: Rc::new(RefCell::new(AllTypes::new())),
            errors: Rc::new(receiver),
        };

        CURRENT_DEPTH.with(|s| s.set(0));

        // Write shared runs within a flock; disable thread dispatching of IO temporarily.
        Arc::get_mut(&mut cx.shared).unwrap().fs.set_sync_only(true);
        write_shared(&cx, &krate, index, &md_opts, &cache)?;
        Arc::get_mut(&mut cx.shared).unwrap().fs.set_sync_only(false);
        Ok((cx, krate))
    }

    fn after_run(&mut self, diag: &rustc_errors::Handler) -> Result<(), Error> {
        Arc::get_mut(&mut self.shared).unwrap().fs.close();
        let nb_errors = self.errors.iter().map(|err| diag.struct_err(&err).emit()).count();
        if nb_errors > 0 {
            Err(Error::new(io::Error::new(io::ErrorKind::Other, "I/O error"), ""))
        } else {
            Ok(())
        }
    }

    fn after_krate(&mut self, krate: &clean::Crate, cache: &Cache) -> Result<(), Error> {
        let final_file = self.dst.join(&*krate.name.as_str()).join("all.html");
        let settings_file = self.dst.join("settings.html");
        let crate_name = krate.name.clone();

        let mut root_path = self.dst.to_str().expect("invalid path").to_owned();
        if !root_path.ends_with('/') {
            root_path.push('/');
        }
        let mut page = layout::Page {
            title: "List of all items in this crate",
            css_class: "mod",
            root_path: "../",
            static_root_path: self.shared.static_root_path.as_deref(),
            description: "List of all items in this crate",
            keywords: BASIC_KEYWORDS,
            resource_suffix: &self.shared.resource_suffix,
            extra_scripts: &[],
            static_extra_scripts: &[],
        };
        let sidebar = if let Some(ref version) = cache.crate_version {
            format!(
                "<p class=\"location\">Crate {}</p>\
                     <div class=\"block version\">\
                         <p>Version {}</p>\
                     </div>\
                     <a id=\"all-types\" href=\"index.html\"><p>Back to index</p></a>",
                crate_name,
                Escape(version),
            )
        } else {
            String::new()
        };
        let all = self.all.replace(AllTypes::new());
        let v = layout::render(
            &self.shared.layout,
            &page,
            sidebar,
            |buf: &mut Buffer| all.print(buf),
            &self.shared.style_files,
        );
        self.shared.fs.write(&final_file, v.as_bytes())?;

        // Generating settings page.
        page.title = "Rustdoc settings";
        page.description = "Settings of Rustdoc";
        page.root_path = "./";

        let mut style_files = self.shared.style_files.clone();
        let sidebar = "<p class=\"location\">Settings</p><div class=\"sidebar-elems\"></div>";
        style_files.push(StylePath { path: PathBuf::from("settings.css"), disabled: false });
        let v = layout::render(
            &self.shared.layout,
            &page,
            sidebar,
            settings(
                self.shared.static_root_path.as_deref().unwrap_or("./"),
                &self.shared.resource_suffix,
                &self.shared.style_files,
            )?,
            &style_files,
        );
        self.shared.fs.write(&settings_file, v.as_bytes())?;
        Ok(())
    }

    fn mod_item_in(
        &mut self,
        item: &clean::Item,
        item_name: &str,
        cache: &Cache,
    ) -> Result<(), Error> {
        // Stripped modules survive the rustdoc passes (i.e., `strip-private`)
        // if they contain impls for public types. These modules can also
        // contain items such as publicly re-exported structures.
        //
        // External crates will provide links to these structures, so
        // these modules are recursed into, but not rendered normally
        // (a flag on the context).
        if !self.render_redirect_pages {
            self.render_redirect_pages = item.is_stripped();
        }
        let scx = &self.shared;
        self.dst.push(item_name);
        self.current.push(item_name.to_owned());

        info!("Recursing into {}", self.dst.display());

        let buf = self.render_item(item, false, cache);
        // buf will be empty if the module is stripped and there is no redirect for it
        if !buf.is_empty() {
            self.shared.ensure_dir(&self.dst)?;
            let joint_dst = self.dst.join("index.html");
            scx.fs.write(&joint_dst, buf.as_bytes())?;
        }

        // Render sidebar-items.js used throughout this module.
        if !self.render_redirect_pages {
            let module = match *item.kind {
                clean::StrippedItem(box clean::ModuleItem(ref m)) | clean::ModuleItem(ref m) => m,
                _ => unreachable!(),
            };
            let items = self.build_sidebar_items(module);
            let js_dst = self.dst.join("sidebar-items.js");
            let v = format!("initSidebarItems({});", serde_json::to_string(&items).unwrap());
            scx.fs.write(&js_dst, &v)?;
        }
        Ok(())
    }

    fn mod_item_out(&mut self, _item_name: &str) -> Result<(), Error> {
        info!("Recursed; leaving {}", self.dst.display());

        // Go back to where we were at
        self.dst.pop();
        self.current.pop();
        Ok(())
    }

    fn item(&mut self, item: clean::Item, cache: &Cache) -> Result<(), Error> {
        // Stripped modules survive the rustdoc passes (i.e., `strip-private`)
        // if they contain impls for public types. These modules can also
        // contain items such as publicly re-exported structures.
        //
        // External crates will provide links to these structures, so
        // these modules are recursed into, but not rendered normally
        // (a flag on the context).
        if !self.render_redirect_pages {
            self.render_redirect_pages = item.is_stripped();
        }

        let buf = self.render_item(&item, true, cache);
        // buf will be empty if the item is stripped and there is no redirect for it
        if !buf.is_empty() {
            let name = item.name.as_ref().unwrap();
            let item_type = item.type_();
            let file_name = &item_path(item_type, &name.as_str());
            self.shared.ensure_dir(&self.dst)?;
            let joint_dst = self.dst.join(file_name);
            self.shared.fs.write(&joint_dst, buf.as_bytes())?;

            if !self.render_redirect_pages {
                self.all.borrow_mut().append(full_path(self, &item), &item_type);
            }
            // If the item is a macro, redirect from the old macro URL (with !)
            // to the new one (without).
            if item_type == ItemType::Macro {
                let redir_name = format!("{}.{}!.html", item_type, name);
                let redir_dst = self.dst.join(redir_name);
                let v = layout::redirect(file_name);
                self.shared.fs.write(&redir_dst, v.as_bytes())?;
            }
        }
        Ok(())
    }
}

fn write_shared(
    cx: &Context<'_>,
    krate: &clean::Crate,
    search_index: String,
    options: &RenderOptions,
    cache: &Cache,
) -> Result<(), Error> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    let lock_file = cx.dst.join(".lock");
    let _lock = try_err!(flock::Lock::new(&lock_file, true, true, true), &lock_file);

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.

    write_minify(
        &cx.shared.fs,
        cx.path("rustdoc.css"),
        static_files::RUSTDOC_CSS,
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("settings.css"),
        static_files::SETTINGS_CSS,
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("noscript.css"),
        static_files::NOSCRIPT_CSS,
        options.enable_minification,
    )?;

    // To avoid "light.css" to be overwritten, we'll first run over the received themes and only
    // then we'll run over the "official" styles.
    let mut themes: FxHashSet<String> = FxHashSet::default();

    for entry in &cx.shared.style_files {
        let theme = try_none!(try_none!(entry.path.file_stem(), &entry.path).to_str(), &entry.path);
        let extension =
            try_none!(try_none!(entry.path.extension(), &entry.path).to_str(), &entry.path);

        // Handle the official themes
        match theme {
            "light" => write_minify(
                &cx.shared.fs,
                cx.path("light.css"),
                static_files::themes::LIGHT,
                options.enable_minification,
            )?,
            "dark" => write_minify(
                &cx.shared.fs,
                cx.path("dark.css"),
                static_files::themes::DARK,
                options.enable_minification,
            )?,
            "ayu" => write_minify(
                &cx.shared.fs,
                cx.path("ayu.css"),
                static_files::themes::AYU,
                options.enable_minification,
            )?,
            _ => {
                // Handle added third-party themes
                let content = try_err!(fs::read(&entry.path), &entry.path);
                cx.shared
                    .fs
                    .write(cx.path(&format!("{}.{}", theme, extension)), content.as_slice())?;
            }
        };

        themes.insert(theme.to_owned());
    }

    let write = |p, c| cx.shared.fs.write(p, c);
    if (*cx.shared).layout.logo.is_empty() {
        write(cx.path("rust-logo.png"), static_files::RUST_LOGO)?;
    }
    if (*cx.shared).layout.favicon.is_empty() {
        write(cx.path("favicon.svg"), static_files::RUST_FAVICON_SVG)?;
        write(cx.path("favicon-16x16.png"), static_files::RUST_FAVICON_PNG_16)?;
        write(cx.path("favicon-32x32.png"), static_files::RUST_FAVICON_PNG_32)?;
    }
    write(cx.path("brush.svg"), static_files::BRUSH_SVG)?;
    write(cx.path("wheel.svg"), static_files::WHEEL_SVG)?;
    write(cx.path("down-arrow.svg"), static_files::DOWN_ARROW_SVG)?;

    let mut themes: Vec<&String> = themes.iter().collect();
    themes.sort();
    // To avoid theme switch latencies as much as possible, we put everything theme related
    // at the beginning of the html files into another js file.
    let theme_js = format!(
        r#"var themes = document.getElementById("theme-choices");
var themePicker = document.getElementById("theme-picker");

function showThemeButtonState() {{
    themes.style.display = "block";
    themePicker.style.borderBottomRightRadius = "0";
    themePicker.style.borderBottomLeftRadius = "0";
}}

function hideThemeButtonState() {{
    themes.style.display = "none";
    themePicker.style.borderBottomRightRadius = "3px";
    themePicker.style.borderBottomLeftRadius = "3px";
}}

function switchThemeButtonState() {{
    if (themes.style.display === "block") {{
        hideThemeButtonState();
    }} else {{
        showThemeButtonState();
    }}
}};

function handleThemeButtonsBlur(e) {{
    var active = document.activeElement;
    var related = e.relatedTarget;

    if (active.id !== "theme-picker" &&
        (!active.parentNode || active.parentNode.id !== "theme-choices") &&
        (!related ||
         (related.id !== "theme-picker" &&
          (!related.parentNode || related.parentNode.id !== "theme-choices")))) {{
        hideThemeButtonState();
    }}
}}

themePicker.onclick = switchThemeButtonState;
themePicker.onblur = handleThemeButtonsBlur;
{}.forEach(function(item) {{
    var but = document.createElement("button");
    but.textContent = item;
    but.onclick = function(el) {{
        switchTheme(currentTheme, mainTheme, item, true);
        useSystemTheme(false);
    }};
    but.onblur = handleThemeButtonsBlur;
    themes.appendChild(but);
}});"#,
        serde_json::to_string(&themes).unwrap()
    );

    write_minify(&cx.shared.fs, cx.path("theme.js"), &theme_js, options.enable_minification)?;
    write_minify(
        &cx.shared.fs,
        cx.path("main.js"),
        static_files::MAIN_JS,
        options.enable_minification,
    )?;
    write_minify(
        &cx.shared.fs,
        cx.path("settings.js"),
        static_files::SETTINGS_JS,
        options.enable_minification,
    )?;
    if cx.shared.include_sources {
        write_minify(
            &cx.shared.fs,
            cx.path("source-script.js"),
            static_files::sidebar::SOURCE_SCRIPT,
            options.enable_minification,
        )?;
    }

    {
        write_minify(
            &cx.shared.fs,
            cx.path("storage.js"),
            &format!(
                "var resourcesSuffix = \"{}\";{}",
                cx.shared.resource_suffix,
                static_files::STORAGE_JS
            ),
            options.enable_minification,
        )?;
    }

    if let Some(ref css) = cx.shared.layout.css_file_extension {
        let out = cx.path("theme.css");
        let buffer = try_err!(fs::read_to_string(css), css);
        if !options.enable_minification {
            cx.shared.fs.write(&out, &buffer)?;
        } else {
            write_minify(&cx.shared.fs, out, &buffer, options.enable_minification)?;
        }
    }
    write_minify(
        &cx.shared.fs,
        cx.path("normalize.css"),
        static_files::NORMALIZE_CSS,
        options.enable_minification,
    )?;
    write(cx.dst.join("FiraSans-Regular.woff"), static_files::fira_sans::REGULAR)?;
    write(cx.dst.join("FiraSans-Medium.woff"), static_files::fira_sans::MEDIUM)?;
    write(cx.dst.join("FiraSans-LICENSE.txt"), static_files::fira_sans::LICENSE)?;
    write(cx.dst.join("SourceSerifPro-Regular.ttf.woff"), static_files::source_serif_pro::REGULAR)?;
    write(cx.dst.join("SourceSerifPro-Bold.ttf.woff"), static_files::source_serif_pro::BOLD)?;
    write(cx.dst.join("SourceSerifPro-It.ttf.woff"), static_files::source_serif_pro::ITALIC)?;
    write(cx.dst.join("SourceSerifPro-LICENSE.md"), static_files::source_serif_pro::LICENSE)?;
    write(cx.dst.join("SourceCodePro-Regular.woff"), static_files::source_code_pro::REGULAR)?;
    write(cx.dst.join("SourceCodePro-Semibold.woff"), static_files::source_code_pro::SEMIBOLD)?;
    write(cx.dst.join("SourceCodePro-LICENSE.txt"), static_files::source_code_pro::LICENSE)?;
    write(cx.dst.join("LICENSE-MIT.txt"), static_files::LICENSE_MIT)?;
    write(cx.dst.join("LICENSE-APACHE.txt"), static_files::LICENSE_APACHE)?;
    write(cx.dst.join("COPYRIGHT.txt"), static_files::COPYRIGHT)?;

    fn collect(path: &Path, krate: &str, key: &str) -> io::Result<(Vec<String>, Vec<String>)> {
        let mut ret = Vec::new();
        let mut krates = Vec::new();

        if path.exists() {
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if !line.starts_with(key) {
                    continue;
                }
                if line.starts_with(&format!(r#"{}["{}"]"#, key, krate)) {
                    continue;
                }
                ret.push(line.to_string());
                krates.push(
                    line[key.len() + 2..]
                        .split('"')
                        .next()
                        .map(|s| s.to_owned())
                        .unwrap_or_else(String::new),
                );
            }
        }
        Ok((ret, krates))
    }

    fn collect_json(path: &Path, krate: &str) -> io::Result<(Vec<String>, Vec<String>)> {
        let mut ret = Vec::new();
        let mut krates = Vec::new();

        if path.exists() {
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if !line.starts_with('"') {
                    continue;
                }
                if line.starts_with(&format!("\"{}\"", krate)) {
                    continue;
                }
                if line.ends_with(",\\") {
                    ret.push(line[..line.len() - 2].to_string());
                } else {
                    // Ends with "\\" (it's the case for the last added crate line)
                    ret.push(line[..line.len() - 1].to_string());
                }
                krates.push(
                    line.split('"')
                        .find(|s| !s.is_empty())
                        .map(|s| s.to_owned())
                        .unwrap_or_else(String::new),
                );
            }
        }
        Ok((ret, krates))
    }

    use std::ffi::OsString;

    #[derive(Debug)]
    struct Hierarchy {
        elem: OsString,
        children: FxHashMap<OsString, Hierarchy>,
        elems: FxHashSet<OsString>,
    }

    impl Hierarchy {
        fn new(elem: OsString) -> Hierarchy {
            Hierarchy { elem, children: FxHashMap::default(), elems: FxHashSet::default() }
        }

        fn to_json_string(&self) -> String {
            let mut subs: Vec<&Hierarchy> = self.children.values().collect();
            subs.sort_unstable_by(|a, b| a.elem.cmp(&b.elem));
            let mut files = self
                .elems
                .iter()
                .map(|s| format!("\"{}\"", s.to_str().expect("invalid osstring conversion")))
                .collect::<Vec<_>>();
            files.sort_unstable_by(|a, b| a.cmp(b));
            let subs = subs.iter().map(|s| s.to_json_string()).collect::<Vec<_>>().join(",");
            let dirs =
                if subs.is_empty() { String::new() } else { format!(",\"dirs\":[{}]", subs) };
            let files = files.join(",");
            let files =
                if files.is_empty() { String::new() } else { format!(",\"files\":[{}]", files) };
            format!(
                "{{\"name\":\"{name}\"{dirs}{files}}}",
                name = self.elem.to_str().expect("invalid osstring conversion"),
                dirs = dirs,
                files = files
            )
        }
    }

    if cx.shared.include_sources {
        let mut hierarchy = Hierarchy::new(OsString::new());
        for source in cx
            .shared
            .local_sources
            .iter()
            .filter_map(|p| p.0.strip_prefix(&cx.shared.src_root).ok())
        {
            let mut h = &mut hierarchy;
            let mut elems = source
                .components()
                .filter_map(|s| match s {
                    Component::Normal(s) => Some(s.to_owned()),
                    _ => None,
                })
                .peekable();
            loop {
                let cur_elem = elems.next().expect("empty file path");
                if elems.peek().is_none() {
                    h.elems.insert(cur_elem);
                    break;
                } else {
                    let e = cur_elem.clone();
                    h.children.entry(cur_elem.clone()).or_insert_with(|| Hierarchy::new(e));
                    h = h.children.get_mut(&cur_elem).expect("not found child");
                }
            }
        }

        let dst = cx.dst.join(&format!("source-files{}.js", cx.shared.resource_suffix));
        let (mut all_sources, _krates) =
            try_err!(collect(&dst, &krate.name.as_str(), "sourcesIndex"), &dst);
        all_sources.push(format!(
            "sourcesIndex[\"{}\"] = {};",
            &krate.name,
            hierarchy.to_json_string()
        ));
        all_sources.sort();
        let v = format!(
            "var N = null;var sourcesIndex = {{}};\n{}\ncreateSourceSidebar();\n",
            all_sources.join("\n")
        );
        cx.shared.fs.write(&dst, v.as_bytes())?;
    }

    // Update the search index
    let dst = cx.dst.join(&format!("search-index{}.js", cx.shared.resource_suffix));
    let (mut all_indexes, mut krates) = try_err!(collect_json(&dst, &krate.name.as_str()), &dst);
    all_indexes.push(search_index);

    // Sort the indexes by crate so the file will be generated identically even
    // with rustdoc running in parallel.
    all_indexes.sort();
    {
        let mut v = String::from("var searchIndex = JSON.parse('{\\\n");
        v.push_str(&all_indexes.join(",\\\n"));
        // "addSearchOptions" has to be called first so the crate filtering can be set before the
        // search might start (if it's set into the URL for example).
        v.push_str("\\\n}');\naddSearchOptions(searchIndex);initSearch(searchIndex);");
        cx.shared.fs.write(&dst, &v)?;
    }
    if options.enable_index_page {
        if let Some(index_page) = options.index_page.clone() {
            let mut md_opts = options.clone();
            md_opts.output = cx.dst.clone();
            md_opts.external_html = (*cx.shared).layout.external_html.clone();

            crate::markdown::render(&index_page, md_opts, cx.shared.edition)
                .map_err(|e| Error::new(e, &index_page))?;
        } else {
            let dst = cx.dst.join("index.html");
            let page = layout::Page {
                title: "Index of crates",
                css_class: "mod",
                root_path: "./",
                static_root_path: cx.shared.static_root_path.as_deref(),
                description: "List of crates",
                keywords: BASIC_KEYWORDS,
                resource_suffix: &cx.shared.resource_suffix,
                extra_scripts: &[],
                static_extra_scripts: &[],
            };
            krates.push(krate.name.to_string());
            krates.sort();
            krates.dedup();

            let content = format!(
                "<h1 class=\"fqn\">\
                     <span class=\"in-band\">List of all crates</span>\
                </h1><ul class=\"crate mod\">{}</ul>",
                krates
                    .iter()
                    .map(|s| {
                        format!(
                            "<li><a class=\"crate mod\" href=\"{}index.html\">{}</a></li>",
                            ensure_trailing_slash(s),
                            s
                        )
                    })
                    .collect::<String>()
            );
            let v = layout::render(&cx.shared.layout, &page, "", content, &cx.shared.style_files);
            cx.shared.fs.write(&dst, v.as_bytes())?;
        }
    }

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    for (&did, imps) in &cache.implementors {
        // Private modules can leak through to this phase of rustdoc, which
        // could contain implementations for otherwise private types. In some
        // rare cases we could find an implementation for an item which wasn't
        // indexed, so we just skip this step in that case.
        //
        // FIXME: this is a vague explanation for why this can't be a `get`, in
        //        theory it should be...
        let &(ref remote_path, remote_item_type) = match cache.paths.get(&did) {
            Some(p) => p,
            None => match cache.external_paths.get(&did) {
                Some(p) => p,
                None => continue,
            },
        };

        #[derive(Serialize)]
        struct Implementor {
            text: String,
            synthetic: bool,
            types: Vec<String>,
        }

        let implementors = imps
            .iter()
            .filter_map(|imp| {
                // If the trait and implementation are in the same crate, then
                // there's no need to emit information about it (there's inlining
                // going on). If they're in different crates then the crate defining
                // the trait will be interested in our implementation.
                //
                // If the implementation is from another crate then that crate
                // should add it.
                if imp.impl_item.def_id.krate == did.krate || !imp.impl_item.def_id.is_local() {
                    None
                } else {
                    Some(Implementor {
                        text: imp.inner_impl().print().to_string(),
                        synthetic: imp.inner_impl().synthetic,
                        types: collect_paths_for_type(imp.inner_impl().for_.clone()),
                    })
                }
            })
            .collect::<Vec<_>>();

        // Only create a js file if we have impls to add to it. If the trait is
        // documented locally though we always create the file to avoid dead
        // links.
        if implementors.is_empty() && !cache.paths.contains_key(&did) {
            continue;
        }

        let implementors = format!(
            r#"implementors["{}"] = {};"#,
            krate.name,
            serde_json::to_string(&implementors).unwrap()
        );

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part);
        }
        cx.shared.ensure_dir(&mydst)?;
        mydst.push(&format!("{}.{}.js", remote_item_type, remote_path[remote_path.len() - 1]));

        let (mut all_implementors, _) =
            try_err!(collect(&mydst, &krate.name.as_str(), "implementors"), &mydst);
        all_implementors.push(implementors);
        // Sort the implementors by crate so the file will be generated
        // identically even with rustdoc running in parallel.
        all_implementors.sort();

        let mut v = String::from("(function() {var implementors = {};\n");
        for implementor in &all_implementors {
            writeln!(v, "{}", *implementor).unwrap();
        }
        v.push_str(
            "if (window.register_implementors) {\
                 window.register_implementors(implementors);\
             } else {\
                 window.pending_implementors = implementors;\
             }",
        );
        v.push_str("})()");
        cx.shared.fs.write(&mydst, &v)?;
    }
    Ok(())
}

fn write_minify(
    fs: &DocFS,
    dst: PathBuf,
    contents: &str,
    enable_minification: bool,
) -> Result<(), Error> {
    if enable_minification {
        if dst.extension() == Some(&OsStr::new("css")) {
            let res = try_none!(minifier::css::minify(contents).ok(), &dst);
            fs.write(dst, res.as_bytes())
        } else {
            fs.write(dst, minifier::js::minify(contents).as_bytes())
        }
    } else {
        fs.write(dst, contents.as_bytes())
    }
}

fn write_srclink(cx: &Context<'_>, item: &clean::Item, buf: &mut Buffer, cache: &Cache) {
    if let Some(l) = cx.src_href(item, cache) {
        write!(
            buf,
            "<a class=\"srclink\" href=\"{}\" title=\"{}\">[src]</a>",
            l, "goto source code"
        )
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ItemEntry {
    url: String,
    name: String,
}

impl ItemEntry {
    fn new(mut url: String, name: String) -> ItemEntry {
        while url.starts_with('/') {
            url.remove(0);
        }
        ItemEntry { url, name }
    }
}

impl ItemEntry {
    crate fn print(&self) -> impl fmt::Display + '_ {
        crate::html::format::display_fn(move |f| {
            write!(f, "<a href=\"{}\">{}</a>", self.url, Escape(&self.name))
        })
    }
}

impl PartialOrd for ItemEntry {
    fn partial_cmp(&self, other: &ItemEntry) -> Option<::std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ItemEntry {
    fn cmp(&self, other: &ItemEntry) -> ::std::cmp::Ordering {
        self.name.cmp(&other.name)
    }
}

#[derive(Debug)]
struct AllTypes {
    structs: FxHashSet<ItemEntry>,
    enums: FxHashSet<ItemEntry>,
    unions: FxHashSet<ItemEntry>,
    primitives: FxHashSet<ItemEntry>,
    traits: FxHashSet<ItemEntry>,
    macros: FxHashSet<ItemEntry>,
    functions: FxHashSet<ItemEntry>,
    typedefs: FxHashSet<ItemEntry>,
    opaque_tys: FxHashSet<ItemEntry>,
    statics: FxHashSet<ItemEntry>,
    constants: FxHashSet<ItemEntry>,
    keywords: FxHashSet<ItemEntry>,
    attributes: FxHashSet<ItemEntry>,
    derives: FxHashSet<ItemEntry>,
    trait_aliases: FxHashSet<ItemEntry>,
}

impl AllTypes {
    fn new() -> AllTypes {
        let new_set = |cap| FxHashSet::with_capacity_and_hasher(cap, Default::default());
        AllTypes {
            structs: new_set(100),
            enums: new_set(100),
            unions: new_set(100),
            primitives: new_set(26),
            traits: new_set(100),
            macros: new_set(100),
            functions: new_set(100),
            typedefs: new_set(100),
            opaque_tys: new_set(100),
            statics: new_set(100),
            constants: new_set(100),
            keywords: new_set(100),
            attributes: new_set(100),
            derives: new_set(100),
            trait_aliases: new_set(100),
        }
    }

    fn append(&mut self, item_name: String, item_type: &ItemType) {
        let mut url: Vec<_> = item_name.split("::").skip(1).collect();
        if let Some(name) = url.pop() {
            let new_url = format!("{}/{}.{}.html", url.join("/"), item_type, name);
            url.push(name);
            let name = url.join("::");
            match *item_type {
                ItemType::Struct => self.structs.insert(ItemEntry::new(new_url, name)),
                ItemType::Enum => self.enums.insert(ItemEntry::new(new_url, name)),
                ItemType::Union => self.unions.insert(ItemEntry::new(new_url, name)),
                ItemType::Primitive => self.primitives.insert(ItemEntry::new(new_url, name)),
                ItemType::Trait => self.traits.insert(ItemEntry::new(new_url, name)),
                ItemType::Macro => self.macros.insert(ItemEntry::new(new_url, name)),
                ItemType::Function => self.functions.insert(ItemEntry::new(new_url, name)),
                ItemType::Typedef => self.typedefs.insert(ItemEntry::new(new_url, name)),
                ItemType::OpaqueTy => self.opaque_tys.insert(ItemEntry::new(new_url, name)),
                ItemType::Static => self.statics.insert(ItemEntry::new(new_url, name)),
                ItemType::Constant => self.constants.insert(ItemEntry::new(new_url, name)),
                ItemType::ProcAttribute => self.attributes.insert(ItemEntry::new(new_url, name)),
                ItemType::ProcDerive => self.derives.insert(ItemEntry::new(new_url, name)),
                ItemType::TraitAlias => self.trait_aliases.insert(ItemEntry::new(new_url, name)),
                _ => true,
            };
        }
    }
}

fn print_entries(f: &mut Buffer, e: &FxHashSet<ItemEntry>, title: &str, class: &str) {
    if !e.is_empty() {
        let mut e: Vec<&ItemEntry> = e.iter().collect();
        e.sort();
        write!(
            f,
            "<h3 id=\"{}\">{}</h3><ul class=\"{} docblock\">{}</ul>",
            title,
            Escape(title),
            class,
            e.iter().map(|s| format!("<li>{}</li>", s.print())).collect::<String>()
        );
    }
}

impl AllTypes {
    fn print(self, f: &mut Buffer) {
        write!(
            f,
            "<h1 class=\"fqn\">\
                 <span class=\"out-of-band\">\
                     <span id=\"render-detail\">\
                         <a id=\"toggle-all-docs\" href=\"javascript:void(0)\" \
                            title=\"collapse all docs\">\
                             [<span class=\"inner\">&#x2212;</span>]\
                         </a>\
                     </span>
                 </span>
                 <span class=\"in-band\">List of all items</span>\
             </h1>"
        );
        print_entries(f, &self.structs, "Structs", "structs");
        print_entries(f, &self.enums, "Enums", "enums");
        print_entries(f, &self.unions, "Unions", "unions");
        print_entries(f, &self.primitives, "Primitives", "primitives");
        print_entries(f, &self.traits, "Traits", "traits");
        print_entries(f, &self.macros, "Macros", "macros");
        print_entries(f, &self.attributes, "Attribute Macros", "attributes");
        print_entries(f, &self.derives, "Derive Macros", "derives");
        print_entries(f, &self.functions, "Functions", "functions");
        print_entries(f, &self.typedefs, "Typedefs", "typedefs");
        print_entries(f, &self.trait_aliases, "Trait Aliases", "trait-aliases");
        print_entries(f, &self.opaque_tys, "Opaque Types", "opaque-types");
        print_entries(f, &self.statics, "Statics", "statics");
        print_entries(f, &self.constants, "Constants", "constants")
    }
}

#[derive(Debug)]
enum Setting {
    Section {
        description: &'static str,
        sub_settings: Vec<Setting>,
    },
    Toggle {
        js_data_name: &'static str,
        description: &'static str,
        default_value: bool,
    },
    Select {
        js_data_name: &'static str,
        description: &'static str,
        default_value: &'static str,
        options: Vec<(String, String)>,
    },
}

impl Setting {
    fn display(&self, root_path: &str, suffix: &str) -> String {
        match *self {
            Setting::Section { description, ref sub_settings } => format!(
                "<div class=\"setting-line\">\
                     <div class=\"title\">{}</div>\
                     <div class=\"sub-settings\">{}</div>
                 </div>",
                description,
                sub_settings.iter().map(|s| s.display(root_path, suffix)).collect::<String>()
            ),
            Setting::Toggle { js_data_name, description, default_value } => format!(
                "<div class=\"setting-line\">\
                     <label class=\"toggle\">\
                     <input type=\"checkbox\" id=\"{}\" {}>\
                     <span class=\"slider\"></span>\
                     </label>\
                     <div>{}</div>\
                 </div>",
                js_data_name,
                if default_value { " checked" } else { "" },
                description,
            ),
            Setting::Select { js_data_name, description, default_value, ref options } => format!(
                "<div class=\"setting-line\">\
                     <div>{}</div>\
                     <label class=\"select-wrapper\">\
                         <select id=\"{}\" autocomplete=\"off\">{}</select>\
                         <img src=\"{}down-arrow{}.svg\" alt=\"Select item\">\
                     </label>\
                 </div>",
                description,
                js_data_name,
                options
                    .iter()
                    .map(|opt| format!(
                        "<option value=\"{}\" {}>{}</option>",
                        opt.0,
                        if &opt.0 == default_value { "selected" } else { "" },
                        opt.1,
                    ))
                    .collect::<String>(),
                root_path,
                suffix,
            ),
        }
    }
}

impl From<(&'static str, &'static str, bool)> for Setting {
    fn from(values: (&'static str, &'static str, bool)) -> Setting {
        Setting::Toggle { js_data_name: values.0, description: values.1, default_value: values.2 }
    }
}

impl<T: Into<Setting>> From<(&'static str, Vec<T>)> for Setting {
    fn from(values: (&'static str, Vec<T>)) -> Setting {
        Setting::Section {
            description: values.0,
            sub_settings: values.1.into_iter().map(|v| v.into()).collect::<Vec<_>>(),
        }
    }
}

fn settings(root_path: &str, suffix: &str, themes: &[StylePath]) -> Result<String, Error> {
    let theme_names: Vec<(String, String)> = themes
        .iter()
        .map(|entry| {
            let theme =
                try_none!(try_none!(entry.path.file_stem(), &entry.path).to_str(), &entry.path)
                    .to_string();

            Ok((theme.clone(), theme))
        })
        .collect::<Result<_, Error>>()?;

    // (id, explanation, default value)
    let settings: &[Setting] = &[
        (
            "Theme preferences",
            vec![
                Setting::from(("use-system-theme", "Use system theme", true)),
                Setting::Select {
                    js_data_name: "preferred-dark-theme",
                    description: "Preferred dark theme",
                    default_value: "dark",
                    options: theme_names.clone(),
                },
                Setting::Select {
                    js_data_name: "preferred-light-theme",
                    description: "Preferred light theme",
                    default_value: "light",
                    options: theme_names,
                },
            ],
        )
            .into(),
        (
            "Auto-hide item declarations",
            vec![
                ("auto-hide-struct", "Auto-hide structs declaration", true),
                ("auto-hide-enum", "Auto-hide enums declaration", false),
                ("auto-hide-union", "Auto-hide unions declaration", true),
                ("auto-hide-trait", "Auto-hide traits declaration", true),
                ("auto-hide-macro", "Auto-hide macros declaration", false),
            ],
        )
            .into(),
        ("auto-hide-attributes", "Auto-hide item attributes.", true).into(),
        ("auto-hide-method-docs", "Auto-hide item methods' documentation", false).into(),
        ("auto-hide-trait-implementations", "Auto-hide trait implementation documentation", true)
            .into(),
        ("auto-collapse-implementors", "Auto-hide implementors of a trait", true).into(),
        ("go-to-only-result", "Directly go to item in search if there is only one result", false)
            .into(),
        ("line-numbers", "Show line numbers on code examples", false).into(),
        ("disable-shortcuts", "Disable keyboard shortcuts", false).into(),
    ];

    Ok(format!(
        "<h1 class=\"fqn\">\
            <span class=\"in-band\">Rustdoc settings</span>\
        </h1>\
        <div class=\"settings\">{}</div>\
        <script src=\"{}settings{}.js\"></script>",
        settings.iter().map(|s| s.display(root_path, suffix)).collect::<String>(),
        root_path,
        suffix
    ))
}

impl Context<'_> {
    fn derive_id(&self, id: String) -> String {
        let mut map = self.id_map.borrow_mut();
        map.derive(id)
    }

    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    fn root_path(&self) -> String {
        "../".repeat(self.current.len())
    }

    fn render_item(&self, it: &clean::Item, pushname: bool, cache: &Cache) -> String {
        // A little unfortunate that this is done like this, but it sure
        // does make formatting *a lot* nicer.
        CURRENT_DEPTH.with(|slot| {
            slot.set(self.current.len());
        });

        let mut title = if it.is_primitive() || it.is_keyword() {
            // No need to include the namespace for primitive types and keywords
            String::new()
        } else {
            self.current.join("::")
        };
        if pushname {
            if !title.is_empty() {
                title.push_str("::");
            }
            title.push_str(&it.name.unwrap().as_str());
        }
        title.push_str(" - Rust");
        let tyname = it.type_();
        let desc = if it.is_crate() {
            format!("API documentation for the Rust `{}` crate.", self.shared.layout.krate)
        } else {
            format!(
                "API documentation for the Rust `{}` {} in crate `{}`.",
                it.name.as_ref().unwrap(),
                tyname,
                self.shared.layout.krate
            )
        };
        let keywords = make_item_keywords(it);
        let page = layout::Page {
            css_class: tyname.as_str(),
            root_path: &self.root_path(),
            static_root_path: self.shared.static_root_path.as_deref(),
            title: &title,
            description: &desc,
            keywords: &keywords,
            resource_suffix: &self.shared.resource_suffix,
            extra_scripts: &[],
            static_extra_scripts: &[],
        };

        {
            self.id_map.borrow_mut().reset();
            self.id_map.borrow_mut().populate(initial_ids());
        }

        if !self.render_redirect_pages {
            layout::render(
                &self.shared.layout,
                &page,
                |buf: &mut _| print_sidebar(self, it, buf, cache),
                |buf: &mut _| print_item(self, it, buf, cache),
                &self.shared.style_files,
            )
        } else {
            let mut url = self.root_path();
            if let Some(&(ref names, ty)) = cache.paths.get(&it.def_id) {
                for name in &names[..names.len() - 1] {
                    url.push_str(name);
                    url.push_str("/");
                }
                url.push_str(&item_path(ty, names.last().unwrap()));
                layout::redirect(&url)
            } else {
                String::new()
            }
        }
    }

    /// Construct a map of items shown in the sidebar to a plain-text summary of their docs.
    fn build_sidebar_items(&self, m: &clean::Module) -> BTreeMap<String, Vec<NameDoc>> {
        // BTreeMap instead of HashMap to get a sorted output
        let mut map: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for item in &m.items {
            if item.is_stripped() {
                continue;
            }

            let short = item.type_();
            let myname = match item.name {
                None => continue,
                Some(ref s) => s.to_string(),
            };
            let short = short.to_string();
            map.entry(short).or_default().push((
                myname,
                Some(item.doc_value().map_or_else(|| String::new(), plain_text_summary)),
            ));
        }

        if self.shared.sort_modules_alphabetically {
            for items in map.values_mut() {
                items.sort();
            }
        }
        map
    }

    /// Generates a url appropriate for an `href` attribute back to the source of
    /// this item.
    ///
    /// The url generated, when clicked, will redirect the browser back to the
    /// original source code.
    ///
    /// If `None` is returned, then a source link couldn't be generated. This
    /// may happen, for example, with externally inlined items where the source
    /// of their crate documentation isn't known.
    fn src_href(&self, item: &clean::Item, cache: &Cache) -> Option<String> {
        let mut root = self.root_path();
        let mut path = String::new();
        let cnum = item.source.cnum(self.sess());

        // We can safely ignore synthetic `SourceFile`s.
        let file = match item.source.filename(self.sess()) {
            FileName::Real(ref path) => path.local_path().to_path_buf(),
            _ => return None,
        };
        let file = &file;

        let symbol;
        let (krate, path) = if cnum == LOCAL_CRATE {
            if let Some(path) = self.shared.local_sources.get(file) {
                (self.shared.layout.krate.as_str(), path)
            } else {
                return None;
            }
        } else {
            let (krate, src_root) = match *cache.extern_locations.get(&cnum)? {
                (name, ref src, ExternalLocation::Local) => (name, src),
                (name, ref src, ExternalLocation::Remote(ref s)) => {
                    root = s.to_string();
                    (name, src)
                }
                (_, _, ExternalLocation::Unknown) => return None,
            };

            sources::clean_path(&src_root, file, false, |component| {
                path.push_str(&component.to_string_lossy());
                path.push('/');
            });
            let mut fname = file.file_name().expect("source has no filename").to_os_string();
            fname.push(".html");
            path.push_str(&fname.to_string_lossy());
            symbol = krate.as_str();
            (&*symbol, &path)
        };

        let loline = item.source.lo(self.sess()).line;
        let hiline = item.source.hi(self.sess()).line;
        let lines =
            if loline == hiline { loline.to_string() } else { format!("{}-{}", loline, hiline) };
        Some(format!(
            "{root}src/{krate}/{path}#{lines}",
            root = Escape(&root),
            krate = krate,
            path = path,
            lines = lines
        ))
    }
}

fn wrap_into_docblock<F>(w: &mut Buffer, f: F)
where
    F: FnOnce(&mut Buffer),
{
    write!(w, "<div class=\"docblock type-decl hidden-by-usual-hider\">");
    f(w);
    write!(w, "</div>")
}

fn print_item(cx: &Context<'_>, item: &clean::Item, buf: &mut Buffer, cache: &Cache) {
    debug_assert!(!item.is_stripped());
    // Write the breadcrumb trail header for the top
    write!(buf, "<h1 class=\"fqn\"><span class=\"out-of-band\">");
    render_stability_since_raw(
        buf,
        item.stable_since(cx.tcx()).as_deref(),
        item.const_stable_since(cx.tcx()).as_deref(),
        None,
        None,
    );
    write!(
        buf,
        "<span id=\"render-detail\">\
                <a id=\"toggle-all-docs\" href=\"javascript:void(0)\" \
                    title=\"collapse all docs\">\
                    [<span class=\"inner\">&#x2212;</span>]\
                </a>\
            </span>"
    );

    // Write `src` tag
    //
    // When this item is part of a `crate use` in a downstream crate, the
    // [src] link in the downstream documentation will actually come back to
    // this page, and this link will be auto-clicked. The `id` attribute is
    // used to find the link to auto-click.
    if cx.shared.include_sources && !item.is_primitive() {
        write_srclink(cx, item, buf, cache);
    }

    write!(buf, "</span>"); // out-of-band
    write!(buf, "<span class=\"in-band\">");
    let name = match *item.kind {
        clean::ModuleItem(ref m) => {
            if m.is_crate {
                "Crate "
            } else {
                "Module "
            }
        }
        clean::FunctionItem(..) | clean::ForeignFunctionItem(..) => "Function ",
        clean::TraitItem(..) => "Trait ",
        clean::StructItem(..) => "Struct ",
        clean::UnionItem(..) => "Union ",
        clean::EnumItem(..) => "Enum ",
        clean::TypedefItem(..) => "Type Definition ",
        clean::MacroItem(..) => "Macro ",
        clean::ProcMacroItem(ref mac) => match mac.kind {
            MacroKind::Bang => "Macro ",
            MacroKind::Attr => "Attribute Macro ",
            MacroKind::Derive => "Derive Macro ",
        },
        clean::PrimitiveItem(..) => "Primitive Type ",
        clean::StaticItem(..) | clean::ForeignStaticItem(..) => "Static ",
        clean::ConstantItem(..) => "Constant ",
        clean::ForeignTypeItem => "Foreign Type ",
        clean::KeywordItem(..) => "Keyword ",
        clean::OpaqueTyItem(..) => "Opaque Type ",
        clean::TraitAliasItem(..) => "Trait Alias ",
        _ => {
            // We don't generate pages for any other type.
            unreachable!();
        }
    };
    buf.write_str(name);
    if !item.is_primitive() && !item.is_keyword() {
        let cur = &cx.current;
        let amt = if item.is_mod() { cur.len() - 1 } else { cur.len() };
        for (i, component) in cur.iter().enumerate().take(amt) {
            write!(
                buf,
                "<a href=\"{}index.html\">{}</a>::<wbr>",
                "../".repeat(cur.len() - i - 1),
                component
            );
        }
    }
    write!(buf, "<a class=\"{}\" href=\"\">{}</a>", item.type_(), item.name.as_ref().unwrap());

    write!(buf, "</span></h1>"); // in-band

    match *item.kind {
        clean::ModuleItem(ref m) => item_module(buf, cx, item, &m.items),
        clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) => {
            item_function(buf, cx, item, f)
        }
        clean::TraitItem(ref t) => item_trait(buf, cx, item, t, cache),
        clean::StructItem(ref s) => item_struct(buf, cx, item, s, cache),
        clean::UnionItem(ref s) => item_union(buf, cx, item, s, cache),
        clean::EnumItem(ref e) => item_enum(buf, cx, item, e, cache),
        clean::TypedefItem(ref t, _) => item_typedef(buf, cx, item, t, cache),
        clean::MacroItem(ref m) => item_macro(buf, cx, item, m),
        clean::ProcMacroItem(ref m) => item_proc_macro(buf, cx, item, m),
        clean::PrimitiveItem(_) => item_primitive(buf, cx, item, cache),
        clean::StaticItem(ref i) | clean::ForeignStaticItem(ref i) => item_static(buf, cx, item, i),
        clean::ConstantItem(ref c) => item_constant(buf, cx, item, c),
        clean::ForeignTypeItem => item_foreign_type(buf, cx, item, cache),
        clean::KeywordItem(_) => item_keyword(buf, cx, item),
        clean::OpaqueTyItem(ref e) => item_opaque_ty(buf, cx, item, e, cache),
        clean::TraitAliasItem(ref ta) => item_trait_alias(buf, cx, item, ta, cache),
        _ => {
            // We don't generate pages for any other type.
            unreachable!();
        }
    }
}

fn item_path(ty: ItemType, name: &str) -> String {
    match ty {
        ItemType::Module => format!("{}index.html", ensure_trailing_slash(name)),
        _ => format!("{}.{}.html", ty, name),
    }
}

fn full_path(cx: &Context<'_>, item: &clean::Item) -> String {
    let mut s = cx.current.join("::");
    s.push_str("::");
    s.push_str(&item.name.unwrap().as_str());
    s
}

fn document(w: &mut Buffer, cx: &Context<'_>, item: &clean::Item, parent: Option<&clean::Item>) {
    if let Some(ref name) = item.name {
        info!("Documenting {}", name);
    }
    document_item_info(w, cx, item, false, parent);
    document_full(w, item, cx, "", false);
}

/// Render md_text as markdown.
fn render_markdown(
    w: &mut Buffer,
    cx: &Context<'_>,
    md_text: &str,
    links: Vec<RenderedLink>,
    prefix: &str,
    is_hidden: bool,
) {
    let mut ids = cx.id_map.borrow_mut();
    write!(
        w,
        "<div class=\"docblock{}\">{}{}</div>",
        if is_hidden { " hidden" } else { "" },
        prefix,
        Markdown(
            md_text,
            &links,
            &mut ids,
            cx.shared.codes,
            cx.shared.edition,
            &cx.shared.playground
        )
        .into_string()
    )
}

/// Writes a documentation block containing only the first paragraph of the documentation. If the
/// docs are longer, a "Read more" link is appended to the end.
fn document_short(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &Context<'_>,
    link: AssocItemLink<'_>,
    prefix: &str,
    is_hidden: bool,
    parent: Option<&clean::Item>,
    show_def_docs: bool,
) {
    document_item_info(w, cx, item, is_hidden, parent);
    if !show_def_docs {
        return;
    }
    if let Some(s) = item.doc_value() {
        let mut summary_html = MarkdownSummaryLine(s, &item.links()).into_string();

        if s.contains('\n') {
            let link = format!(r#" <a href="{}">Read more</a>"#, naive_assoc_href(item, link));

            if let Some(idx) = summary_html.rfind("</p>") {
                summary_html.insert_str(idx, &link);
            } else {
                summary_html.push_str(&link);
            }
        }

        write!(
            w,
            "<div class='docblock{}'>{}{}</div>",
            if is_hidden { " hidden" } else { "" },
            prefix,
            summary_html,
        );
    } else if !prefix.is_empty() {
        write!(
            w,
            "<div class=\"docblock{}\">{}</div>",
            if is_hidden { " hidden" } else { "" },
            prefix
        );
    }
}

fn document_full(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &Context<'_>,
    prefix: &str,
    is_hidden: bool,
) {
    if let Some(s) = cx.shared.maybe_collapsed_doc_value(item) {
        debug!("Doc block: =====\n{}\n=====", s);
        render_markdown(w, cx, &*s, item.links(), prefix, is_hidden);
    } else if !prefix.is_empty() {
        write!(
            w,
            "<div class=\"docblock{}\">{}</div>",
            if is_hidden { " hidden" } else { "" },
            prefix
        );
    }
}

/// Add extra information about an item such as:
///
/// * Stability
/// * Deprecated
/// * Required features (through the `doc_cfg` feature)
fn document_item_info(
    w: &mut Buffer,
    cx: &Context<'_>,
    item: &clean::Item,
    is_hidden: bool,
    parent: Option<&clean::Item>,
) {
    let item_infos = short_item_info(item, cx, parent);
    if !item_infos.is_empty() {
        write!(w, "<div class=\"item-info{}\">", if is_hidden { " hidden" } else { "" });
        for info in item_infos {
            write!(w, "{}", info);
        }
        write!(w, "</div>");
    }
}

fn document_non_exhaustive_header(item: &clean::Item) -> &str {
    if item.is_non_exhaustive() { " (Non-exhaustive)" } else { "" }
}

fn document_non_exhaustive(w: &mut Buffer, item: &clean::Item) {
    if item.is_non_exhaustive() {
        write!(w, "<div class=\"docblock non-exhaustive non-exhaustive-{}\">", {
            if item.is_struct() {
                "struct"
            } else if item.is_enum() {
                "enum"
            } else if item.is_variant() {
                "variant"
            } else {
                "type"
            }
        });

        if item.is_struct() {
            write!(
                w,
                "Non-exhaustive structs could have additional fields added in future. \
                 Therefore, non-exhaustive structs cannot be constructed in external crates \
                 using the traditional <code>Struct {{ .. }}</code> syntax; cannot be \
                 matched against without a wildcard <code>..</code>; and \
                 struct update syntax will not work."
            );
        } else if item.is_enum() {
            write!(
                w,
                "Non-exhaustive enums could have additional variants added in future. \
                 Therefore, when matching against variants of non-exhaustive enums, an \
                 extra wildcard arm must be added to account for any future variants."
            );
        } else if item.is_variant() {
            write!(
                w,
                "Non-exhaustive enum variants could have additional fields added in future. \
                 Therefore, non-exhaustive enum variants cannot be constructed in external \
                 crates and cannot be matched against."
            );
        } else {
            write!(
                w,
                "This type will require a wildcard arm in any match statements or constructors."
            );
        }

        write!(w, "</div>");
    }
}

/// Compare two strings treating multi-digit numbers as single units (i.e. natural sort order).
crate fn compare_names(mut lhs: &str, mut rhs: &str) -> Ordering {
    /// Takes a non-numeric and a numeric part from the given &str.
    fn take_parts<'a>(s: &mut &'a str) -> (&'a str, &'a str) {
        let i = s.find(|c: char| c.is_ascii_digit());
        let (a, b) = s.split_at(i.unwrap_or(s.len()));
        let i = b.find(|c: char| !c.is_ascii_digit());
        let (b, c) = b.split_at(i.unwrap_or(b.len()));
        *s = c;
        (a, b)
    }

    while !lhs.is_empty() || !rhs.is_empty() {
        let (la, lb) = take_parts(&mut lhs);
        let (ra, rb) = take_parts(&mut rhs);
        // First process the non-numeric part.
        match la.cmp(ra) {
            Ordering::Equal => (),
            x => return x,
        }
        // Then process the numeric part, if both sides have one (and they fit in a u64).
        if let (Ok(ln), Ok(rn)) = (lb.parse::<u64>(), rb.parse::<u64>()) {
            match ln.cmp(&rn) {
                Ordering::Equal => (),
                x => return x,
            }
        }
        // Then process the numeric part again, but this time as strings.
        match lb.cmp(rb) {
            Ordering::Equal => (),
            x => return x,
        }
    }

    Ordering::Equal
}

fn item_module(w: &mut Buffer, cx: &Context<'_>, item: &clean::Item, items: &[clean::Item]) {
    document(w, cx, item, None);

    let mut indices = (0..items.len()).filter(|i| !items[*i].is_stripped()).collect::<Vec<usize>>();

    // the order of item types in the listing
    fn reorder(ty: ItemType) -> u8 {
        match ty {
            ItemType::ExternCrate => 0,
            ItemType::Import => 1,
            ItemType::Primitive => 2,
            ItemType::Module => 3,
            ItemType::Macro => 4,
            ItemType::Struct => 5,
            ItemType::Enum => 6,
            ItemType::Constant => 7,
            ItemType::Static => 8,
            ItemType::Trait => 9,
            ItemType::Function => 10,
            ItemType::Typedef => 12,
            ItemType::Union => 13,
            _ => 14 + ty as u8,
        }
    }

    fn cmp(
        i1: &clean::Item,
        i2: &clean::Item,
        idx1: usize,
        idx2: usize,
        tcx: TyCtxt<'_>,
    ) -> Ordering {
        let ty1 = i1.type_();
        let ty2 = i2.type_();
        if ty1 != ty2 {
            return (reorder(ty1), idx1).cmp(&(reorder(ty2), idx2));
        }
        let s1 = i1.stability(tcx).as_ref().map(|s| s.level);
        let s2 = i2.stability(tcx).as_ref().map(|s| s.level);
        if let (Some(a), Some(b)) = (s1, s2) {
            match (a.is_stable(), b.is_stable()) {
                (true, true) | (false, false) => {}
                (false, true) => return Ordering::Less,
                (true, false) => return Ordering::Greater,
            }
        }
        let lhs = i1.name.unwrap_or(kw::Invalid).as_str();
        let rhs = i2.name.unwrap_or(kw::Invalid).as_str();
        compare_names(&lhs, &rhs)
    }

    if cx.shared.sort_modules_alphabetically {
        indices.sort_by(|&i1, &i2| cmp(&items[i1], &items[i2], i1, i2, cx.tcx()));
    }
    // This call is to remove re-export duplicates in cases such as:
    //
    // ```
    // crate mod foo {
    //     crate mod bar {
    //         crate trait Double { fn foo(); }
    //     }
    // }
    //
    // crate use foo::bar::*;
    // crate use foo::*;
    // ```
    //
    // `Double` will appear twice in the generated docs.
    //
    // FIXME: This code is quite ugly and could be improved. Small issue: DefId
    // can be identical even if the elements are different (mostly in imports).
    // So in case this is an import, we keep everything by adding a "unique id"
    // (which is the position in the vector).
    indices.dedup_by_key(|i| {
        (
            items[*i].def_id,
            if items[*i].name.as_ref().is_some() { Some(full_path(cx, &items[*i])) } else { None },
            items[*i].type_(),
            if items[*i].is_import() { *i } else { 0 },
        )
    });

    debug!("{:?}", indices);
    let mut curty = None;
    for &idx in &indices {
        let myitem = &items[idx];
        if myitem.is_stripped() {
            continue;
        }

        let myty = Some(myitem.type_());
        if curty == Some(ItemType::ExternCrate) && myty == Some(ItemType::Import) {
            // Put `extern crate` and `use` re-exports in the same section.
            curty = myty;
        } else if myty != curty {
            if curty.is_some() {
                write!(w, "</table>");
            }
            curty = myty;
            let (short, name) = item_ty_to_strs(&myty.unwrap());
            write!(
                w,
                "<h2 id=\"{id}\" class=\"section-header\">\
                       <a href=\"#{id}\">{name}</a></h2>\n<table>",
                id = cx.derive_id(short.to_owned()),
                name = name
            );
        }

        match *myitem.kind {
            clean::ExternCrateItem(ref name, ref src) => {
                use crate::html::format::anchor;

                match *src {
                    Some(ref src) => write!(
                        w,
                        "<tr><td><code>{}extern crate {} as {};",
                        myitem.visibility.print_with_space(cx.tcx()),
                        anchor(myitem.def_id, &*src.as_str()),
                        name
                    ),
                    None => write!(
                        w,
                        "<tr><td><code>{}extern crate {};",
                        myitem.visibility.print_with_space(cx.tcx()),
                        anchor(myitem.def_id, &*name.as_str())
                    ),
                }
                write!(w, "</code></td></tr>");
            }

            clean::ImportItem(ref import) => {
                write!(
                    w,
                    "<tr><td><code>{}{}</code></td></tr>",
                    myitem.visibility.print_with_space(cx.tcx()),
                    import.print()
                );
            }

            _ => {
                if myitem.name.is_none() {
                    continue;
                }

                let unsafety_flag = match *myitem.kind {
                    clean::FunctionItem(ref func) | clean::ForeignFunctionItem(ref func)
                        if func.header.unsafety == hir::Unsafety::Unsafe =>
                    {
                        "<a title=\"unsafe function\" href=\"#\"><sup>⚠</sup></a>"
                    }
                    _ => "",
                };

                let stab = myitem.stability_class(cx.tcx());
                let add = if stab.is_some() { " " } else { "" };

                let doc_value = myitem.doc_value().unwrap_or("");
                write!(
                    w,
                    "<tr class=\"{stab}{add}module-item\">\
                         <td><a class=\"{class}\" href=\"{href}\" \
                             title=\"{title}\">{name}</a>{unsafety_flag}</td>\
                         <td class=\"docblock-short\">{stab_tags}{docs}</td>\
                     </tr>",
                    name = *myitem.name.as_ref().unwrap(),
                    stab_tags = extra_info_tags(myitem, item, cx.tcx()),
                    docs = MarkdownSummaryLine(doc_value, &myitem.links()).into_string(),
                    class = myitem.type_(),
                    add = add,
                    stab = stab.unwrap_or_else(String::new),
                    unsafety_flag = unsafety_flag,
                    href = item_path(myitem.type_(), &myitem.name.unwrap().as_str()),
                    title = [full_path(cx, myitem), myitem.type_().to_string()]
                        .iter()
                        .filter_map(|s| if !s.is_empty() { Some(s.as_str()) } else { None })
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }
    }

    if curty.is_some() {
        write!(w, "</table>");
    }
}

/// Render the stability, deprecation and portability tags that are displayed in the item's summary
/// at the module level.
fn extra_info_tags(item: &clean::Item, parent: &clean::Item, tcx: TyCtxt<'_>) -> String {
    let mut tags = String::new();

    fn tag_html(class: &str, title: &str, contents: &str) -> String {
        format!(r#"<span class="stab {}" title="{}">{}</span>"#, class, Escape(title), contents)
    }

    // The trailing space after each tag is to space it properly against the rest of the docs.
    if let Some(depr) = &item.deprecation(tcx) {
        let mut message = "Deprecated";
        if !stability::deprecation_in_effect(
            depr.is_since_rustc_version,
            depr.since.map(|s| s.as_str()).as_deref(),
        ) {
            message = "Deprecation planned";
        }
        tags += &tag_html("deprecated", "", message);
    }

    // The "rustc_private" crates are permanently unstable so it makes no sense
    // to render "unstable" everywhere.
    if item
        .stability(tcx)
        .as_ref()
        .map(|s| s.level.is_unstable() && s.feature != sym::rustc_private)
        == Some(true)
    {
        tags += &tag_html("unstable", "", "Experimental");
    }

    let cfg = match (&item.attrs.cfg, parent.attrs.cfg.as_ref()) {
        (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
        (cfg, _) => cfg.as_deref().cloned(),
    };

    debug!("Portability {:?} - {:?} = {:?}", item.attrs.cfg, parent.attrs.cfg, cfg);
    if let Some(ref cfg) = cfg {
        tags += &tag_html("portability", &cfg.render_long_plain(), &cfg.render_short_html());
    }

    tags
}

fn portability(item: &clean::Item, parent: Option<&clean::Item>) -> Option<String> {
    let cfg = match (&item.attrs.cfg, parent.and_then(|p| p.attrs.cfg.as_ref())) {
        (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
        (cfg, _) => cfg.as_deref().cloned(),
    };

    debug!(
        "Portability {:?} - {:?} = {:?}",
        item.attrs.cfg,
        parent.and_then(|p| p.attrs.cfg.as_ref()),
        cfg
    );

    Some(format!("<div class=\"stab portability\">{}</div>", cfg?.render_long_html()))
}

/// Render the stability, deprecation and portability information that is displayed at the top of
/// the item's documentation.
fn short_item_info(
    item: &clean::Item,
    cx: &Context<'_>,
    parent: Option<&clean::Item>,
) -> Vec<String> {
    let mut extra_info = vec![];
    let error_codes = cx.shared.codes;

    if let Some(Deprecation { note, since, is_since_rustc_version, suggestion: _ }) =
        item.deprecation(cx.tcx())
    {
        // We display deprecation messages for #[deprecated] and #[rustc_deprecated]
        // but only display the future-deprecation messages for #[rustc_deprecated].
        let mut message = if let Some(since) = since {
            let since = &since.as_str();
            if !stability::deprecation_in_effect(is_since_rustc_version, Some(since)) {
                if *since == "TBD" {
                    format!("Deprecating in a future Rust version")
                } else {
                    format!("Deprecating in {}", Escape(since))
                }
            } else {
                format!("Deprecated since {}", Escape(since))
            }
        } else {
            String::from("Deprecated")
        };

        if let Some(note) = note {
            let note = note.as_str();
            let mut ids = cx.id_map.borrow_mut();
            let html = MarkdownHtml(
                &note,
                &mut ids,
                error_codes,
                cx.shared.edition,
                &cx.shared.playground,
            );
            message.push_str(&format!(": {}", html.into_string()));
        }
        extra_info.push(format!(
            "<div class=\"stab deprecated\"><span class=\"emoji\">👎</span> {}</div>",
            message,
        ));
    }

    // Render unstable items. But don't render "rustc_private" crates (internal compiler crates).
    // Those crates are permanently unstable so it makes no sense to render "unstable" everywhere.
    if let Some((StabilityLevel::Unstable { reason, issue, .. }, feature)) = item
        .stability(cx.tcx())
        .as_ref()
        .filter(|stab| stab.feature != sym::rustc_private)
        .map(|stab| (stab.level, stab.feature))
    {
        let mut message =
            "<span class=\"emoji\">🔬</span> This is a nightly-only experimental API.".to_owned();

        let mut feature = format!("<code>{}</code>", Escape(&feature.as_str()));
        if let (Some(url), Some(issue)) = (&cx.shared.issue_tracker_base_url, issue) {
            feature.push_str(&format!(
                "&nbsp;<a href=\"{url}{issue}\">#{issue}</a>",
                url = url,
                issue = issue
            ));
        }

        message.push_str(&format!(" ({})", feature));

        if let Some(unstable_reason) = reason {
            let mut ids = cx.id_map.borrow_mut();
            message = format!(
                "<details><summary>{}</summary>{}</details>",
                message,
                MarkdownHtml(
                    &unstable_reason.as_str(),
                    &mut ids,
                    error_codes,
                    cx.shared.edition,
                    &cx.shared.playground,
                )
                .into_string()
            );
        }

        extra_info.push(format!("<div class=\"stab unstable\">{}</div>", message));
    }

    if let Some(portability) = portability(item, parent) {
        extra_info.push(portability);
    }

    extra_info
}

fn item_constant(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, c: &clean::Constant) {
    write!(w, "<pre class=\"rust const\">");
    render_attributes(w, it, false);

    write!(
        w,
        "{vis}const {name}: {typ}",
        vis = it.visibility.print_with_space(cx.tcx()),
        name = it.name.as_ref().unwrap(),
        typ = c.type_.print(),
    );

    if c.value.is_some() || c.is_literal {
        write!(w, " = {expr};", expr = Escape(&c.expr));
    } else {
        write!(w, ";");
    }

    if let Some(value) = &c.value {
        if !c.is_literal {
            let value_lowercase = value.to_lowercase();
            let expr_lowercase = c.expr.to_lowercase();

            if value_lowercase != expr_lowercase
                && value_lowercase.trim_end_matches("i32") != expr_lowercase
            {
                write!(w, " // {value}", value = Escape(value));
            }
        }
    }

    write!(w, "</pre>");
    document(w, cx, it, None)
}

fn item_static(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, s: &clean::Static) {
    write!(w, "<pre class=\"rust static\">");
    render_attributes(w, it, false);
    write!(
        w,
        "{vis}static {mutability}{name}: {typ}</pre>",
        vis = it.visibility.print_with_space(cx.tcx()),
        mutability = s.mutability.print_with_space(),
        name = it.name.as_ref().unwrap(),
        typ = s.type_.print()
    );
    document(w, cx, it, None)
}

fn item_function(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, f: &clean::Function) {
    let header_len = format!(
        "{}{}{}{}{:#}fn {}{:#}",
        it.visibility.print_with_space(cx.tcx()),
        f.header.constness.print_with_space(),
        f.header.asyncness.print_with_space(),
        f.header.unsafety.print_with_space(),
        print_abi_with_space(f.header.abi),
        it.name.as_ref().unwrap(),
        f.generics.print()
    )
    .len();
    write!(w, "<pre class=\"rust fn\">");
    render_attributes(w, it, false);
    write!(
        w,
        "{vis}{constness}{asyncness}{unsafety}{abi}fn \
         {name}{generics}{decl}{spotlight}{where_clause}</pre>",
        vis = it.visibility.print_with_space(cx.tcx()),
        constness = f.header.constness.print_with_space(),
        asyncness = f.header.asyncness.print_with_space(),
        unsafety = f.header.unsafety.print_with_space(),
        abi = print_abi_with_space(f.header.abi),
        name = it.name.as_ref().unwrap(),
        generics = f.generics.print(),
        where_clause = WhereClause { gens: &f.generics, indent: 0, end_newline: true },
        decl = Function { decl: &f.decl, header_len, indent: 0, asyncness: f.header.asyncness }
            .print(),
        spotlight = spotlight_decl(&f.decl),
    );
    document(w, cx, it, None)
}

fn render_implementor(
    cx: &Context<'_>,
    implementor: &Impl,
    parent: &clean::Item,
    w: &mut Buffer,
    implementor_dups: &FxHashMap<Symbol, (DefId, bool)>,
    aliases: &[String],
    cache: &Cache,
) {
    // If there's already another implementor that has the same abbridged name, use the
    // full path, for example in `std::iter::ExactSizeIterator`
    let use_absolute = match implementor.inner_impl().for_ {
        clean::ResolvedPath { ref path, is_generic: false, .. }
        | clean::BorrowedRef {
            type_: box clean::ResolvedPath { ref path, is_generic: false, .. },
            ..
        } => implementor_dups[&path.last()].1,
        _ => false,
    };
    render_impl(
        w,
        cx,
        implementor,
        parent,
        AssocItemLink::Anchor(None),
        RenderMode::Normal,
        implementor.impl_item.stable_since(cx.tcx()).as_deref(),
        implementor.impl_item.const_stable_since(cx.tcx()).as_deref(),
        false,
        Some(use_absolute),
        false,
        false,
        aliases,
        cache,
    );
}

fn render_impls(
    cx: &Context<'_>,
    w: &mut Buffer,
    traits: &[&&Impl],
    containing_item: &clean::Item,
    cache: &Cache,
) {
    let mut impls = traits
        .iter()
        .map(|i| {
            let did = i.trait_did().unwrap();
            let assoc_link = AssocItemLink::GotoSource(did, &i.inner_impl().provided_trait_methods);
            let mut buffer = if w.is_for_html() { Buffer::html() } else { Buffer::new() };
            render_impl(
                &mut buffer,
                cx,
                i,
                containing_item,
                assoc_link,
                RenderMode::Normal,
                containing_item.stable_since(cx.tcx()).as_deref(),
                containing_item.const_stable_since(cx.tcx()).as_deref(),
                true,
                None,
                false,
                true,
                &[],
                cache,
            );
            buffer.into_inner()
        })
        .collect::<Vec<_>>();
    impls.sort();
    w.write_str(&impls.join(""));
}

fn bounds(t_bounds: &[clean::GenericBound], trait_alias: bool) -> String {
    let mut bounds = String::new();
    if !t_bounds.is_empty() {
        if !trait_alias {
            bounds.push_str(": ");
        }
        for (i, p) in t_bounds.iter().enumerate() {
            if i > 0 {
                bounds.push_str(" + ");
            }
            bounds.push_str(&p.print().to_string());
        }
    }
    bounds
}

fn compare_impl<'a, 'b>(lhs: &'a &&Impl, rhs: &'b &&Impl) -> Ordering {
    let lhs = format!("{}", lhs.inner_impl().print());
    let rhs = format!("{}", rhs.inner_impl().print());

    // lhs and rhs are formatted as HTML, which may be unnecessary
    compare_names(&lhs, &rhs)
}

fn item_trait(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, t: &clean::Trait, cache: &Cache) {
    let bounds = bounds(&t.bounds, false);
    let types = t.items.iter().filter(|m| m.is_associated_type()).collect::<Vec<_>>();
    let consts = t.items.iter().filter(|m| m.is_associated_const()).collect::<Vec<_>>();
    let required = t.items.iter().filter(|m| m.is_ty_method()).collect::<Vec<_>>();
    let provided = t.items.iter().filter(|m| m.is_method()).collect::<Vec<_>>();

    // Output the trait definition
    wrap_into_docblock(w, |w| {
        write!(w, "<pre class=\"rust trait\">");
        render_attributes(w, it, true);
        write!(
            w,
            "{}{}{}trait {}{}{}",
            it.visibility.print_with_space(cx.tcx()),
            t.unsafety.print_with_space(),
            if t.is_auto { "auto " } else { "" },
            it.name.as_ref().unwrap(),
            t.generics.print(),
            bounds
        );

        if !t.generics.where_predicates.is_empty() {
            write!(w, "{}", WhereClause { gens: &t.generics, indent: 0, end_newline: true });
        } else {
            write!(w, " ");
        }

        if t.items.is_empty() {
            write!(w, "{{ }}");
        } else {
            // FIXME: we should be using a derived_id for the Anchors here
            write!(w, "{{\n");
            for t in &types {
                render_assoc_item(w, t, AssocItemLink::Anchor(None), ItemType::Trait, cx);
                write!(w, ";\n");
            }
            if !types.is_empty() && !consts.is_empty() {
                w.write_str("\n");
            }
            for t in &consts {
                render_assoc_item(w, t, AssocItemLink::Anchor(None), ItemType::Trait, cx);
                write!(w, ";\n");
            }
            if !consts.is_empty() && !required.is_empty() {
                w.write_str("\n");
            }
            for (pos, m) in required.iter().enumerate() {
                render_assoc_item(w, m, AssocItemLink::Anchor(None), ItemType::Trait, cx);
                write!(w, ";\n");

                if pos < required.len() - 1 {
                    write!(w, "<div class=\"item-spacer\"></div>");
                }
            }
            if !required.is_empty() && !provided.is_empty() {
                w.write_str("\n");
            }
            for (pos, m) in provided.iter().enumerate() {
                render_assoc_item(w, m, AssocItemLink::Anchor(None), ItemType::Trait, cx);
                match *m.kind {
                    clean::MethodItem(ref inner, _)
                        if !inner.generics.where_predicates.is_empty() =>
                    {
                        write!(w, ",\n    {{ ... }}\n");
                    }
                    _ => {
                        write!(w, " {{ ... }}\n");
                    }
                }
                if pos < provided.len() - 1 {
                    write!(w, "<div class=\"item-spacer\"></div>");
                }
            }
            write!(w, "}}");
        }
        write!(w, "</pre>")
    });

    // Trait documentation
    document(w, cx, it, None);

    fn write_small_section_header(w: &mut Buffer, id: &str, title: &str, extra_content: &str) {
        write!(
            w,
            "<h2 id=\"{0}\" class=\"small-section-header\">\
                {1}<a href=\"#{0}\" class=\"anchor\"></a>\
             </h2>{2}",
            id, title, extra_content
        )
    }

    fn write_loading_content(w: &mut Buffer, extra_content: &str) {
        write!(w, "{}<span class=\"loading-content\">Loading content...</span>", extra_content)
    }

    fn trait_item(
        w: &mut Buffer,
        cx: &Context<'_>,
        m: &clean::Item,
        t: &clean::Item,
        cache: &Cache,
    ) {
        let name = m.name.as_ref().unwrap();
        info!("Documenting {} on {:?}", name, t.name);
        let item_type = m.type_();
        let id = cx.derive_id(format!("{}.{}", item_type, name));
        write!(w, "<h3 id=\"{id}\" class=\"method\"><code>", id = id,);
        render_assoc_item(w, m, AssocItemLink::Anchor(Some(&id)), ItemType::Impl, cx);
        write!(w, "</code>");
        render_stability_since(w, m, t, cx.tcx());
        write_srclink(cx, m, w, cache);
        write!(w, "</h3>");
        document(w, cx, m, Some(t));
    }

    if !types.is_empty() {
        write_small_section_header(
            w,
            "associated-types",
            "Associated Types",
            "<div class=\"methods\">",
        );
        for t in types {
            trait_item(w, cx, t, it, cache);
        }
        write_loading_content(w, "</div>");
    }

    if !consts.is_empty() {
        write_small_section_header(
            w,
            "associated-const",
            "Associated Constants",
            "<div class=\"methods\">",
        );
        for t in consts {
            trait_item(w, cx, t, it, cache);
        }
        write_loading_content(w, "</div>");
    }

    // Output the documentation for each function individually
    if !required.is_empty() {
        write_small_section_header(
            w,
            "required-methods",
            "Required methods",
            "<div class=\"methods\">",
        );
        for m in required {
            trait_item(w, cx, m, it, cache);
        }
        write_loading_content(w, "</div>");
    }
    if !provided.is_empty() {
        write_small_section_header(
            w,
            "provided-methods",
            "Provided methods",
            "<div class=\"methods\">",
        );
        for m in provided {
            trait_item(w, cx, m, it, cache);
        }
        write_loading_content(w, "</div>");
    }

    // If there are methods directly on this trait object, render them here.
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache);

    if let Some(implementors) = cache.implementors.get(&it.def_id) {
        // The DefId is for the first Type found with that name. The bool is
        // if any Types with the same name but different DefId have been found.
        let mut implementor_dups: FxHashMap<Symbol, (DefId, bool)> = FxHashMap::default();
        for implementor in implementors {
            match implementor.inner_impl().for_ {
                clean::ResolvedPath { ref path, did, is_generic: false, .. }
                | clean::BorrowedRef {
                    type_: box clean::ResolvedPath { ref path, did, is_generic: false, .. },
                    ..
                } => {
                    let &mut (prev_did, ref mut has_duplicates) =
                        implementor_dups.entry(path.last()).or_insert((did, false));
                    if prev_did != did {
                        *has_duplicates = true;
                    }
                }
                _ => {}
            }
        }

        let (local, foreign) = implementors.iter().partition::<Vec<_>, _>(|i| {
            i.inner_impl().for_.def_id().map_or(true, |d| cache.paths.contains_key(&d))
        });

        let (mut synthetic, mut concrete): (Vec<&&Impl>, Vec<&&Impl>) =
            local.iter().partition(|i| i.inner_impl().synthetic);

        synthetic.sort_by(compare_impl);
        concrete.sort_by(compare_impl);

        if !foreign.is_empty() {
            write_small_section_header(w, "foreign-impls", "Implementations on Foreign Types", "");

            for implementor in foreign {
                let assoc_link = AssocItemLink::GotoSource(
                    implementor.impl_item.def_id,
                    &implementor.inner_impl().provided_trait_methods,
                );
                render_impl(
                    w,
                    cx,
                    &implementor,
                    it,
                    assoc_link,
                    RenderMode::Normal,
                    implementor.impl_item.stable_since(cx.tcx()).as_deref(),
                    implementor.impl_item.const_stable_since(cx.tcx()).as_deref(),
                    false,
                    None,
                    true,
                    false,
                    &[],
                    cache,
                );
            }
            write_loading_content(w, "");
        }

        write_small_section_header(
            w,
            "implementors",
            "Implementors",
            "<div class=\"item-list\" id=\"implementors-list\">",
        );
        for implementor in concrete {
            render_implementor(cx, implementor, it, w, &implementor_dups, &[], cache);
        }
        write_loading_content(w, "</div>");

        if t.is_auto {
            write_small_section_header(
                w,
                "synthetic-implementors",
                "Auto implementors",
                "<div class=\"item-list\" id=\"synthetic-implementors-list\">",
            );
            for implementor in synthetic {
                render_implementor(
                    cx,
                    implementor,
                    it,
                    w,
                    &implementor_dups,
                    &collect_paths_for_type(implementor.inner_impl().for_.clone()),
                    cache,
                );
            }
            write_loading_content(w, "</div>");
        }
    } else {
        // even without any implementations to write in, we still want the heading and list, so the
        // implementors javascript file pulled in below has somewhere to write the impls into
        write_small_section_header(
            w,
            "implementors",
            "Implementors",
            "<div class=\"item-list\" id=\"implementors-list\">",
        );
        write_loading_content(w, "</div>");

        if t.is_auto {
            write_small_section_header(
                w,
                "synthetic-implementors",
                "Auto implementors",
                "<div class=\"item-list\" id=\"synthetic-implementors-list\">",
            );
            write_loading_content(w, "</div>");
        }
    }

    write!(
        w,
        "<script type=\"text/javascript\" \
                 src=\"{root_path}/implementors/{path}/{ty}.{name}.js\" async>\
         </script>",
        root_path = vec![".."; cx.current.len()].join("/"),
        path = if it.def_id.is_local() {
            cx.current.join("/")
        } else {
            let (ref path, _) = cache.external_paths[&it.def_id];
            path[..path.len() - 1].join("/")
        },
        ty = it.type_(),
        name = *it.name.as_ref().unwrap()
    );
}

fn naive_assoc_href(it: &clean::Item, link: AssocItemLink<'_>) -> String {
    use crate::formats::item_type::ItemType::*;

    let name = it.name.as_ref().unwrap();
    let ty = match it.type_() {
        Typedef | AssocType => AssocType,
        s => s,
    };

    let anchor = format!("#{}.{}", ty, name);
    match link {
        AssocItemLink::Anchor(Some(ref id)) => format!("#{}", id),
        AssocItemLink::Anchor(None) => anchor,
        AssocItemLink::GotoSource(did, _) => {
            href(did).map(|p| format!("{}{}", p.0, anchor)).unwrap_or(anchor)
        }
    }
}

fn assoc_const(
    w: &mut Buffer,
    it: &clean::Item,
    ty: &clean::Type,
    _default: Option<&String>,
    link: AssocItemLink<'_>,
    extra: &str,
    cx: &Context<'_>,
) {
    write!(
        w,
        "{}{}const <a href=\"{}\" class=\"constant\"><b>{}</b></a>: {}",
        extra,
        it.visibility.print_with_space(cx.tcx()),
        naive_assoc_href(it, link),
        it.name.as_ref().unwrap(),
        ty.print()
    );
}

fn assoc_type(
    w: &mut Buffer,
    it: &clean::Item,
    bounds: &[clean::GenericBound],
    default: Option<&clean::Type>,
    link: AssocItemLink<'_>,
    extra: &str,
) {
    write!(
        w,
        "{}type <a href=\"{}\" class=\"type\">{}</a>",
        extra,
        naive_assoc_href(it, link),
        it.name.as_ref().unwrap()
    );
    if !bounds.is_empty() {
        write!(w, ": {}", print_generic_bounds(bounds))
    }
    if let Some(default) = default {
        write!(w, " = {}", default.print())
    }
}

fn render_stability_since_raw(
    w: &mut Buffer,
    ver: Option<&str>,
    const_ver: Option<&str>,
    containing_ver: Option<&str>,
    containing_const_ver: Option<&str>,
) {
    let ver = ver.and_then(|inner| if !inner.is_empty() { Some(inner) } else { None });

    let const_ver = const_ver.and_then(|inner| if !inner.is_empty() { Some(inner) } else { None });

    if let Some(v) = ver {
        if let Some(cv) = const_ver {
            if const_ver != containing_const_ver {
                write!(
                    w,
                    "<span class=\"since\" title=\"Stable since Rust version {0}, const since {1}\">{0} (const: {1})</span>",
                    v, cv
                );
            } else if ver != containing_ver {
                write!(
                    w,
                    "<span class=\"since\" title=\"Stable since Rust version {0}\">{0}</span>",
                    v
                );
            }
        } else {
            if ver != containing_ver {
                write!(
                    w,
                    "<span class=\"since\" title=\"Stable since Rust version {0}\">{0}</span>",
                    v
                );
            }
        }
    }
}

fn render_stability_since(
    w: &mut Buffer,
    item: &clean::Item,
    containing_item: &clean::Item,
    tcx: TyCtxt<'_>,
) {
    render_stability_since_raw(
        w,
        item.stable_since(tcx).as_deref(),
        item.const_stable_since(tcx).as_deref(),
        containing_item.stable_since(tcx).as_deref(),
        containing_item.const_stable_since(tcx).as_deref(),
    )
}

fn render_assoc_item(
    w: &mut Buffer,
    item: &clean::Item,
    link: AssocItemLink<'_>,
    parent: ItemType,
    cx: &Context<'_>,
) {
    fn method(
        w: &mut Buffer,
        meth: &clean::Item,
        header: hir::FnHeader,
        g: &clean::Generics,
        d: &clean::FnDecl,
        link: AssocItemLink<'_>,
        parent: ItemType,
        cx: &Context<'_>,
    ) {
        let name = meth.name.as_ref().unwrap();
        let anchor = format!("#{}.{}", meth.type_(), name);
        let href = match link {
            AssocItemLink::Anchor(Some(ref id)) => format!("#{}", id),
            AssocItemLink::Anchor(None) => anchor,
            AssocItemLink::GotoSource(did, provided_methods) => {
                // We're creating a link from an impl-item to the corresponding
                // trait-item and need to map the anchored type accordingly.
                let ty = if provided_methods.contains(&name) {
                    ItemType::Method
                } else {
                    ItemType::TyMethod
                };

                href(did).map(|p| format!("{}#{}.{}", p.0, ty, name)).unwrap_or(anchor)
            }
        };
        let mut header_len = format!(
            "{}{}{}{}{}{:#}fn {}{:#}",
            meth.visibility.print_with_space(cx.tcx()),
            header.constness.print_with_space(),
            header.asyncness.print_with_space(),
            header.unsafety.print_with_space(),
            print_default_space(meth.is_default()),
            print_abi_with_space(header.abi),
            name,
            g.print()
        )
        .len();
        let (indent, end_newline) = if parent == ItemType::Trait {
            header_len += 4;
            (4, false)
        } else {
            (0, true)
        };
        render_attributes(w, meth, false);
        write!(
            w,
            "{}{}{}{}{}{}{}fn <a href=\"{href}\" class=\"fnname\">{name}</a>\
             {generics}{decl}{spotlight}{where_clause}",
            if parent == ItemType::Trait { "    " } else { "" },
            meth.visibility.print_with_space(cx.tcx()),
            header.constness.print_with_space(),
            header.asyncness.print_with_space(),
            header.unsafety.print_with_space(),
            print_default_space(meth.is_default()),
            print_abi_with_space(header.abi),
            href = href,
            name = name,
            generics = g.print(),
            decl = Function { decl: d, header_len, indent, asyncness: header.asyncness }.print(),
            spotlight = spotlight_decl(&d),
            where_clause = WhereClause { gens: g, indent, end_newline }
        )
    }
    match *item.kind {
        clean::StrippedItem(..) => {}
        clean::TyMethodItem(ref m) => {
            method(w, item, m.header, &m.generics, &m.decl, link, parent, cx)
        }
        clean::MethodItem(ref m, _) => {
            method(w, item, m.header, &m.generics, &m.decl, link, parent, cx)
        }
        clean::AssocConstItem(ref ty, ref default) => assoc_const(
            w,
            item,
            ty,
            default.as_ref(),
            link,
            if parent == ItemType::Trait { "    " } else { "" },
            cx,
        ),
        clean::AssocTypeItem(ref bounds, ref default) => assoc_type(
            w,
            item,
            bounds,
            default.as_ref(),
            link,
            if parent == ItemType::Trait { "    " } else { "" },
        ),
        _ => panic!("render_assoc_item called on non-associated-item"),
    }
}

fn item_struct(
    w: &mut Buffer,
    cx: &Context<'_>,
    it: &clean::Item,
    s: &clean::Struct,
    cache: &Cache,
) {
    wrap_into_docblock(w, |w| {
        write!(w, "<pre class=\"rust struct\">");
        render_attributes(w, it, true);
        render_struct(w, it, Some(&s.generics), s.struct_type, &s.fields, "", true, cx);
        write!(w, "</pre>")
    });

    document(w, cx, it, None);
    let mut fields = s
        .fields
        .iter()
        .filter_map(|f| match *f.kind {
            clean::StructFieldItem(ref ty) => Some((f, ty)),
            _ => None,
        })
        .peekable();
    if let doctree::Plain = s.struct_type {
        if fields.peek().is_some() {
            write!(
                w,
                "<h2 id=\"fields\" class=\"fields small-section-header\">
                       Fields{}<a href=\"#fields\" class=\"anchor\"></a></h2>",
                document_non_exhaustive_header(it)
            );
            document_non_exhaustive(w, it);
            for (field, ty) in fields {
                let id = cx.derive_id(format!(
                    "{}.{}",
                    ItemType::StructField,
                    field.name.as_ref().unwrap()
                ));
                write!(
                    w,
                    "<span id=\"{id}\" class=\"{item_type} small-section-header\">\
                         <a href=\"#{id}\" class=\"anchor field\"></a>\
                         <code>{name}: {ty}</code>\
                     </span>",
                    item_type = ItemType::StructField,
                    id = id,
                    name = field.name.as_ref().unwrap(),
                    ty = ty.print()
                );
                document(w, cx, field, Some(it));
            }
        }
    }
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_union(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, s: &clean::Union, cache: &Cache) {
    wrap_into_docblock(w, |w| {
        write!(w, "<pre class=\"rust union\">");
        render_attributes(w, it, true);
        render_union(w, it, Some(&s.generics), &s.fields, "", true, cx);
        write!(w, "</pre>")
    });

    document(w, cx, it, None);
    let mut fields = s
        .fields
        .iter()
        .filter_map(|f| match *f.kind {
            clean::StructFieldItem(ref ty) => Some((f, ty)),
            _ => None,
        })
        .peekable();
    if fields.peek().is_some() {
        write!(
            w,
            "<h2 id=\"fields\" class=\"fields small-section-header\">
                   Fields<a href=\"#fields\" class=\"anchor\"></a></h2>"
        );
        for (field, ty) in fields {
            let name = field.name.as_ref().expect("union field name");
            let id = format!("{}.{}", ItemType::StructField, name);
            write!(
                w,
                "<span id=\"{id}\" class=\"{shortty} small-section-header\">\
                     <a href=\"#{id}\" class=\"anchor field\"></a>\
                     <code>{name}: {ty}</code>\
                 </span>",
                id = id,
                name = name,
                shortty = ItemType::StructField,
                ty = ty.print()
            );
            if let Some(stability_class) = field.stability_class(cx.tcx()) {
                write!(w, "<span class=\"stab {stab}\"></span>", stab = stability_class);
            }
            document(w, cx, field, Some(it));
        }
    }
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_enum(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, e: &clean::Enum, cache: &Cache) {
    wrap_into_docblock(w, |w| {
        write!(w, "<pre class=\"rust enum\">");
        render_attributes(w, it, true);
        write!(
            w,
            "{}enum {}{}{}",
            it.visibility.print_with_space(cx.tcx()),
            it.name.as_ref().unwrap(),
            e.generics.print(),
            WhereClause { gens: &e.generics, indent: 0, end_newline: true }
        );
        if e.variants.is_empty() && !e.variants_stripped {
            write!(w, " {{}}");
        } else {
            write!(w, " {{\n");
            for v in &e.variants {
                write!(w, "    ");
                let name = v.name.as_ref().unwrap();
                match *v.kind {
                    clean::VariantItem(ref var) => match var.kind {
                        clean::VariantKind::CLike => write!(w, "{}", name),
                        clean::VariantKind::Tuple(ref tys) => {
                            write!(w, "{}(", name);
                            for (i, ty) in tys.iter().enumerate() {
                                if i > 0 {
                                    write!(w, ",&nbsp;")
                                }
                                write!(w, "{}", ty.print());
                            }
                            write!(w, ")");
                        }
                        clean::VariantKind::Struct(ref s) => {
                            render_struct(w, v, None, s.struct_type, &s.fields, "    ", false, cx);
                        }
                    },
                    _ => unreachable!(),
                }
                write!(w, ",\n");
            }

            if e.variants_stripped {
                write!(w, "    // some variants omitted\n");
            }
            write!(w, "}}");
        }
        write!(w, "</pre>")
    });

    document(w, cx, it, None);
    if !e.variants.is_empty() {
        write!(
            w,
            "<h2 id=\"variants\" class=\"variants small-section-header\">
                   Variants{}<a href=\"#variants\" class=\"anchor\"></a></h2>\n",
            document_non_exhaustive_header(it)
        );
        document_non_exhaustive(w, it);
        for variant in &e.variants {
            let id =
                cx.derive_id(format!("{}.{}", ItemType::Variant, variant.name.as_ref().unwrap()));
            write!(
                w,
                "<div id=\"{id}\" class=\"variant small-section-header\">\
                    <a href=\"#{id}\" class=\"anchor field\"></a>\
                    <code>{name}",
                id = id,
                name = variant.name.as_ref().unwrap()
            );
            if let clean::VariantItem(ref var) = *variant.kind {
                if let clean::VariantKind::Tuple(ref tys) = var.kind {
                    write!(w, "(");
                    for (i, ty) in tys.iter().enumerate() {
                        if i > 0 {
                            write!(w, ",&nbsp;");
                        }
                        write!(w, "{}", ty.print());
                    }
                    write!(w, ")");
                }
            }
            write!(w, "</code></div>");
            document(w, cx, variant, Some(it));
            document_non_exhaustive(w, variant);

            use crate::clean::{Variant, VariantKind};
            if let clean::VariantItem(Variant { kind: VariantKind::Struct(ref s) }) = *variant.kind
            {
                let variant_id = cx.derive_id(format!(
                    "{}.{}.fields",
                    ItemType::Variant,
                    variant.name.as_ref().unwrap()
                ));
                write!(w, "<div class=\"autohide sub-variant\" id=\"{id}\">", id = variant_id);
                write!(
                    w,
                    "<h3>Fields of <b>{name}</b></h3><div>",
                    name = variant.name.as_ref().unwrap()
                );
                for field in &s.fields {
                    use crate::clean::StructFieldItem;
                    if let StructFieldItem(ref ty) = *field.kind {
                        let id = cx.derive_id(format!(
                            "variant.{}.field.{}",
                            variant.name.as_ref().unwrap(),
                            field.name.as_ref().unwrap()
                        ));
                        write!(
                            w,
                            "<span id=\"{id}\" class=\"variant small-section-header\">\
                                 <a href=\"#{id}\" class=\"anchor field\"></a>\
                                 <code>{f}:&nbsp;{t}</code>\
                             </span>",
                            id = id,
                            f = field.name.as_ref().unwrap(),
                            t = ty.print()
                        );
                        document(w, cx, field, Some(variant));
                    }
                }
                write!(w, "</div></div>");
            }
            render_stability_since(w, variant, it, cx.tcx());
        }
    }
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

const ALLOWED_ATTRIBUTES: &[Symbol] = &[
    sym::export_name,
    sym::lang,
    sym::link_section,
    sym::must_use,
    sym::no_mangle,
    sym::repr,
    sym::non_exhaustive,
];

// The `top` parameter is used when generating the item declaration to ensure it doesn't have a
// left padding. For example:
//
// #[foo] <----- "top" attribute
// struct Foo {
//     #[bar] <---- not "top" attribute
//     bar: usize,
// }
fn render_attributes(w: &mut Buffer, it: &clean::Item, top: bool) {
    let attrs = it
        .attrs
        .other_attrs
        .iter()
        .filter_map(|attr| {
            if ALLOWED_ATTRIBUTES.contains(&attr.name_or_empty()) {
                Some(pprust::attribute_to_string(&attr))
            } else {
                None
            }
        })
        .join("\n");

    if !attrs.is_empty() {
        write!(
            w,
            "<span class=\"docblock attributes{}\">{}</span>",
            if top { " top-attr" } else { "" },
            &attrs
        );
    }
}

fn render_struct(
    w: &mut Buffer,
    it: &clean::Item,
    g: Option<&clean::Generics>,
    ty: doctree::StructType,
    fields: &[clean::Item],
    tab: &str,
    structhead: bool,
    cx: &Context<'_>,
) {
    write!(
        w,
        "{}{}{}",
        it.visibility.print_with_space(cx.tcx()),
        if structhead { "struct " } else { "" },
        it.name.as_ref().unwrap()
    );
    if let Some(g) = g {
        write!(w, "{}", g.print())
    }
    match ty {
        doctree::Plain => {
            if let Some(g) = g {
                write!(w, "{}", WhereClause { gens: g, indent: 0, end_newline: true })
            }
            let mut has_visible_fields = false;
            write!(w, " {{");
            for field in fields {
                if let clean::StructFieldItem(ref ty) = *field.kind {
                    write!(
                        w,
                        "\n{}    {}{}: {},",
                        tab,
                        field.visibility.print_with_space(cx.tcx()),
                        field.name.as_ref().unwrap(),
                        ty.print()
                    );
                    has_visible_fields = true;
                }
            }

            if has_visible_fields {
                if it.has_stripped_fields().unwrap() {
                    write!(w, "\n{}    // some fields omitted", tab);
                }
                write!(w, "\n{}", tab);
            } else if it.has_stripped_fields().unwrap() {
                // If there are no visible fields we can just display
                // `{ /* fields omitted */ }` to save space.
                write!(w, " /* fields omitted */ ");
            }
            write!(w, "}}");
        }
        doctree::Tuple => {
            write!(w, "(");
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    write!(w, ", ");
                }
                match *field.kind {
                    clean::StrippedItem(box clean::StructFieldItem(..)) => write!(w, "_"),
                    clean::StructFieldItem(ref ty) => {
                        write!(w, "{}{}", field.visibility.print_with_space(cx.tcx()), ty.print())
                    }
                    _ => unreachable!(),
                }
            }
            write!(w, ")");
            if let Some(g) = g {
                write!(w, "{}", WhereClause { gens: g, indent: 0, end_newline: false })
            }
            write!(w, ";");
        }
        doctree::Unit => {
            // Needed for PhantomData.
            if let Some(g) = g {
                write!(w, "{}", WhereClause { gens: g, indent: 0, end_newline: false })
            }
            write!(w, ";");
        }
    }
}

fn render_union(
    w: &mut Buffer,
    it: &clean::Item,
    g: Option<&clean::Generics>,
    fields: &[clean::Item],
    tab: &str,
    structhead: bool,
    cx: &Context<'_>,
) {
    write!(
        w,
        "{}{}{}",
        it.visibility.print_with_space(cx.tcx()),
        if structhead { "union " } else { "" },
        it.name.as_ref().unwrap()
    );
    if let Some(g) = g {
        write!(w, "{}", g.print());
        write!(w, "{}", WhereClause { gens: g, indent: 0, end_newline: true });
    }

    write!(w, " {{\n{}", tab);
    for field in fields {
        if let clean::StructFieldItem(ref ty) = *field.kind {
            write!(
                w,
                "    {}{}: {},\n{}",
                field.visibility.print_with_space(cx.tcx()),
                field.name.as_ref().unwrap(),
                ty.print(),
                tab
            );
        }
    }

    if it.has_stripped_fields().unwrap() {
        write!(w, "    // some fields omitted\n{}", tab);
    }
    write!(w, "}}");
}

#[derive(Copy, Clone)]
enum AssocItemLink<'a> {
    Anchor(Option<&'a str>),
    GotoSource(DefId, &'a FxHashSet<Symbol>),
}

impl<'a> AssocItemLink<'a> {
    fn anchor(&self, id: &'a String) -> Self {
        match *self {
            AssocItemLink::Anchor(_) => AssocItemLink::Anchor(Some(&id)),
            ref other => *other,
        }
    }
}

fn render_assoc_items(
    w: &mut Buffer,
    cx: &Context<'_>,
    containing_item: &clean::Item,
    it: DefId,
    what: AssocItemRender<'_>,
    cache: &Cache,
) {
    info!("Documenting associated items of {:?}", containing_item.name);
    let v = match cache.impls.get(&it) {
        Some(v) => v,
        None => return,
    };
    let (non_trait, traits): (Vec<_>, _) = v.iter().partition(|i| i.inner_impl().trait_.is_none());
    if !non_trait.is_empty() {
        let render_mode = match what {
            AssocItemRender::All => {
                write!(
                    w,
                    "<h2 id=\"implementations\" class=\"small-section-header\">\
                         Implementations<a href=\"#implementations\" class=\"anchor\"></a>\
                    </h2>"
                );
                RenderMode::Normal
            }
            AssocItemRender::DerefFor { trait_, type_, deref_mut_ } => {
                write!(
                    w,
                    "<h2 id=\"deref-methods\" class=\"small-section-header\">\
                         Methods from {}&lt;Target = {}&gt;\
                         <a href=\"#deref-methods\" class=\"anchor\"></a>\
                     </h2>",
                    trait_.print(),
                    type_.print()
                );
                RenderMode::ForDeref { mut_: deref_mut_ }
            }
        };
        for i in &non_trait {
            render_impl(
                w,
                cx,
                i,
                containing_item,
                AssocItemLink::Anchor(None),
                render_mode,
                containing_item.stable_since(cx.tcx()).as_deref(),
                containing_item.const_stable_since(cx.tcx()).as_deref(),
                true,
                None,
                false,
                true,
                &[],
                cache,
            );
        }
    }
    if let AssocItemRender::DerefFor { .. } = what {
        return;
    }
    if !traits.is_empty() {
        let deref_impl =
            traits.iter().find(|t| t.inner_impl().trait_.def_id() == cache.deref_trait_did);
        if let Some(impl_) = deref_impl {
            let has_deref_mut =
                traits.iter().any(|t| t.inner_impl().trait_.def_id() == cache.deref_mut_trait_did);
            render_deref_methods(w, cx, impl_, containing_item, has_deref_mut, cache);
        }

        let (synthetic, concrete): (Vec<&&Impl>, Vec<&&Impl>) =
            traits.iter().partition(|t| t.inner_impl().synthetic);
        let (blanket_impl, concrete): (Vec<&&Impl>, _) =
            concrete.into_iter().partition(|t| t.inner_impl().blanket_impl.is_some());

        let mut impls = Buffer::empty_from(&w);
        render_impls(cx, &mut impls, &concrete, containing_item, cache);
        let impls = impls.into_inner();
        if !impls.is_empty() {
            write!(
                w,
                "<h2 id=\"trait-implementations\" class=\"small-section-header\">\
                     Trait Implementations<a href=\"#trait-implementations\" class=\"anchor\"></a>\
                 </h2>\
                 <div id=\"trait-implementations-list\">{}</div>",
                impls
            );
        }

        if !synthetic.is_empty() {
            write!(
                w,
                "<h2 id=\"synthetic-implementations\" class=\"small-section-header\">\
                     Auto Trait Implementations\
                     <a href=\"#synthetic-implementations\" class=\"anchor\"></a>\
                 </h2>\
                 <div id=\"synthetic-implementations-list\">"
            );
            render_impls(cx, w, &synthetic, containing_item, cache);
            write!(w, "</div>");
        }

        if !blanket_impl.is_empty() {
            write!(
                w,
                "<h2 id=\"blanket-implementations\" class=\"small-section-header\">\
                     Blanket Implementations\
                     <a href=\"#blanket-implementations\" class=\"anchor\"></a>\
                 </h2>\
                 <div id=\"blanket-implementations-list\">"
            );
            render_impls(cx, w, &blanket_impl, containing_item, cache);
            write!(w, "</div>");
        }
    }
}

fn render_deref_methods(
    w: &mut Buffer,
    cx: &Context<'_>,
    impl_: &Impl,
    container_item: &clean::Item,
    deref_mut: bool,
    cache: &Cache,
) {
    let deref_type = impl_.inner_impl().trait_.as_ref().unwrap();
    let (target, real_target) = impl_
        .inner_impl()
        .items
        .iter()
        .find_map(|item| match *item.kind {
            clean::TypedefItem(ref t, true) => Some(match *t {
                clean::Typedef { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
        .expect("Expected associated type binding");
    let what =
        AssocItemRender::DerefFor { trait_: deref_type, type_: real_target, deref_mut_: deref_mut };
    if let Some(did) = target.def_id() {
        render_assoc_items(w, cx, container_item, did, what, cache);
    } else {
        if let Some(prim) = target.primitive_type() {
            if let Some(&did) = cache.primitive_locations.get(&prim) {
                render_assoc_items(w, cx, container_item, did, what, cache);
            }
        }
    }
}

fn should_render_item(item: &clean::Item, deref_mut_: bool) -> bool {
    let self_type_opt = match *item.kind {
        clean::MethodItem(ref method, _) => method.decl.self_type(),
        clean::TyMethodItem(ref method) => method.decl.self_type(),
        _ => None,
    };

    if let Some(self_ty) = self_type_opt {
        let (by_mut_ref, by_box, by_value) = match self_ty {
            SelfTy::SelfBorrowed(_, mutability)
            | SelfTy::SelfExplicit(clean::BorrowedRef { mutability, .. }) => {
                (mutability == Mutability::Mut, false, false)
            }
            SelfTy::SelfExplicit(clean::ResolvedPath { did, .. }) => {
                (false, Some(did) == cache().owned_box_did, false)
            }
            SelfTy::SelfValue => (false, false, true),
            _ => (false, false, false),
        };

        (deref_mut_ || !by_mut_ref) && !by_box && !by_value
    } else {
        false
    }
}

fn spotlight_decl(decl: &clean::FnDecl) -> String {
    let mut out = Buffer::html();
    let mut trait_ = String::new();

    if let Some(did) = decl.output.def_id() {
        let c = cache();
        if let Some(impls) = c.impls.get(&did) {
            for i in impls {
                let impl_ = i.inner_impl();
                if impl_.trait_.def_id().map_or(false, |d| c.traits[&d].is_spotlight) {
                    if out.is_empty() {
                        out.push_str(&format!(
                            "<h3 class=\"notable\">Notable traits for {}</h3>\
                             <code class=\"content\">",
                            impl_.for_.print()
                        ));
                        trait_.push_str(&impl_.for_.print().to_string());
                    }

                    //use the "where" class here to make it small
                    out.push_str(&format!(
                        "<span class=\"where fmt-newline\">{}</span>",
                        impl_.print()
                    ));
                    let t_did = impl_.trait_.def_id().unwrap();
                    for it in &impl_.items {
                        if let clean::TypedefItem(ref tydef, _) = *it.kind {
                            out.push_str("<span class=\"where fmt-newline\">    ");
                            assoc_type(
                                &mut out,
                                it,
                                &[],
                                Some(&tydef.type_),
                                AssocItemLink::GotoSource(t_did, &FxHashSet::default()),
                                "",
                            );
                            out.push_str(";</span>");
                        }
                    }
                }
            }
        }
    }

    if !out.is_empty() {
        out.insert_str(
            0,
            "<span class=\"notable-traits\"><span class=\"notable-traits-tooltip\">ⓘ\
            <div class=\"notable-traits-tooltiptext\"><span class=\"docblock\">",
        );
        out.push_str("</code></span></div></span></span>");
    }

    out.into_inner()
}

fn render_impl(
    w: &mut Buffer,
    cx: &Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    link: AssocItemLink<'_>,
    render_mode: RenderMode,
    outer_version: Option<&str>,
    outer_const_version: Option<&str>,
    show_def_docs: bool,
    use_absolute: Option<bool>,
    is_on_foreign_type: bool,
    show_default_items: bool,
    // This argument is used to reference same type with different paths to avoid duplication
    // in documentation pages for trait with automatic implementations like "Send" and "Sync".
    aliases: &[String],
    cache: &Cache,
) {
    let traits = &cache.traits;
    let trait_ = i.trait_did().map(|did| &traits[&did]);

    if render_mode == RenderMode::Normal {
        let id = cx.derive_id(match i.inner_impl().trait_ {
            Some(ref t) => {
                if is_on_foreign_type {
                    get_id_for_impl_on_foreign_type(&i.inner_impl().for_, t)
                } else {
                    format!("impl-{}", small_url_encode(&format!("{:#}", t.print())))
                }
            }
            None => "impl".to_string(),
        });
        let aliases = if aliases.is_empty() {
            String::new()
        } else {
            format!(" aliases=\"{}\"", aliases.join(","))
        };
        if let Some(use_absolute) = use_absolute {
            write!(w, "<h3 id=\"{}\" class=\"impl\"{}><code class=\"in-band\">", id, aliases);
            fmt_impl_for_trait_page(&i.inner_impl(), w, use_absolute);
            if show_def_docs {
                for it in &i.inner_impl().items {
                    if let clean::TypedefItem(ref tydef, _) = *it.kind {
                        write!(w, "<span class=\"where fmt-newline\">  ");
                        assoc_type(w, it, &[], Some(&tydef.type_), AssocItemLink::Anchor(None), "");
                        write!(w, ";</span>");
                    }
                }
            }
            write!(w, "</code>");
        } else {
            write!(
                w,
                "<h3 id=\"{}\" class=\"impl\"{}><code class=\"in-band\">{}</code>",
                id,
                aliases,
                i.inner_impl().print()
            );
        }
        write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
        render_stability_since_raw(
            w,
            i.impl_item.stable_since(cx.tcx()).as_deref(),
            i.impl_item.const_stable_since(cx.tcx()).as_deref(),
            outer_version,
            outer_const_version,
        );
        write_srclink(cx, &i.impl_item, w, cache);
        write!(w, "</h3>");

        if trait_.is_some() {
            if let Some(portability) = portability(&i.impl_item, Some(parent)) {
                write!(w, "<div class=\"item-info\">{}</div>", portability);
            }
        }

        if let Some(ref dox) = cx.shared.maybe_collapsed_doc_value(&i.impl_item) {
            let mut ids = cx.id_map.borrow_mut();
            write!(
                w,
                "<div class=\"docblock\">{}</div>",
                Markdown(
                    &*dox,
                    &i.impl_item.links(),
                    &mut ids,
                    cx.shared.codes,
                    cx.shared.edition,
                    &cx.shared.playground
                )
                .into_string()
            );
        }
    }

    fn doc_impl_item(
        w: &mut Buffer,
        cx: &Context<'_>,
        item: &clean::Item,
        parent: &clean::Item,
        link: AssocItemLink<'_>,
        render_mode: RenderMode,
        is_default_item: bool,
        outer_version: Option<&str>,
        outer_const_version: Option<&str>,
        trait_: Option<&clean::Trait>,
        show_def_docs: bool,
        cache: &Cache,
    ) {
        let item_type = item.type_();
        let name = item.name.as_ref().unwrap();

        let render_method_item = match render_mode {
            RenderMode::Normal => true,
            RenderMode::ForDeref { mut_: deref_mut_ } => should_render_item(&item, deref_mut_),
        };

        let (is_hidden, extra_class) =
            if (trait_.is_none() || item.doc_value().is_some() || item.kind.is_type_alias())
                && !is_default_item
            {
                (false, "")
            } else {
                (true, " hidden")
            };
        match *item.kind {
            clean::MethodItem(..) | clean::TyMethodItem(_) => {
                // Only render when the method is not static or we allow static methods
                if render_method_item {
                    let id = cx.derive_id(format!("{}.{}", item_type, name));
                    write!(w, "<h4 id=\"{}\" class=\"{}{}\">", id, item_type, extra_class);
                    write!(w, "<code>");
                    render_assoc_item(w, item, link.anchor(&id), ItemType::Impl, cx);
                    write!(w, "</code>");
                    render_stability_since_raw(
                        w,
                        item.stable_since(cx.tcx()).as_deref(),
                        item.const_stable_since(cx.tcx()).as_deref(),
                        outer_version,
                        outer_const_version,
                    );
                    write_srclink(cx, item, w, cache);
                    write!(w, "</h4>");
                }
            }
            clean::TypedefItem(ref tydef, _) => {
                let id = cx.derive_id(format!("{}.{}", ItemType::AssocType, name));
                write!(w, "<h4 id=\"{}\" class=\"{}{}\"><code>", id, item_type, extra_class);
                assoc_type(w, item, &Vec::new(), Some(&tydef.type_), link.anchor(&id), "");
                write!(w, "</code></h4>");
            }
            clean::AssocConstItem(ref ty, ref default) => {
                let id = cx.derive_id(format!("{}.{}", item_type, name));
                write!(w, "<h4 id=\"{}\" class=\"{}{}\"><code>", id, item_type, extra_class);
                assoc_const(w, item, ty, default.as_ref(), link.anchor(&id), "", cx);
                write!(w, "</code>");
                render_stability_since_raw(
                    w,
                    item.stable_since(cx.tcx()).as_deref(),
                    item.const_stable_since(cx.tcx()).as_deref(),
                    outer_version,
                    outer_const_version,
                );
                write_srclink(cx, item, w, cache);
                write!(w, "</h4>");
            }
            clean::AssocTypeItem(ref bounds, ref default) => {
                let id = cx.derive_id(format!("{}.{}", item_type, name));
                write!(w, "<h4 id=\"{}\" class=\"{}{}\"><code>", id, item_type, extra_class);
                assoc_type(w, item, bounds, default.as_ref(), link.anchor(&id), "");
                write!(w, "</code></h4>");
            }
            clean::StrippedItem(..) => return,
            _ => panic!("can't make docs for trait item with name {:?}", item.name),
        }

        if render_method_item {
            if !is_default_item {
                if let Some(t) = trait_ {
                    // The trait item may have been stripped so we might not
                    // find any documentation or stability for it.
                    if let Some(it) = t.items.iter().find(|i| i.name == item.name) {
                        // We need the stability of the item from the trait
                        // because impls can't have a stability.
                        if item.doc_value().is_some() {
                            document_item_info(w, cx, it, is_hidden, Some(parent));
                            document_full(w, item, cx, "", is_hidden);
                        } else {
                            // In case the item isn't documented,
                            // provide short documentation from the trait.
                            document_short(
                                w,
                                it,
                                cx,
                                link,
                                "",
                                is_hidden,
                                Some(parent),
                                show_def_docs,
                            );
                        }
                    }
                } else {
                    document_item_info(w, cx, item, is_hidden, Some(parent));
                    if show_def_docs {
                        document_full(w, item, cx, "", is_hidden);
                    }
                }
            } else {
                document_short(w, item, cx, link, "", is_hidden, Some(parent), show_def_docs);
            }
        }
    }

    write!(w, "<div class=\"impl-items\">");
    for trait_item in &i.inner_impl().items {
        doc_impl_item(
            w,
            cx,
            trait_item,
            if trait_.is_some() { &i.impl_item } else { parent },
            link,
            render_mode,
            false,
            outer_version,
            outer_const_version,
            trait_,
            show_def_docs,
            cache,
        );
    }

    fn render_default_items(
        w: &mut Buffer,
        cx: &Context<'_>,
        t: &clean::Trait,
        i: &clean::Impl,
        parent: &clean::Item,
        render_mode: RenderMode,
        outer_version: Option<&str>,
        outer_const_version: Option<&str>,
        show_def_docs: bool,
        cache: &Cache,
    ) {
        for trait_item in &t.items {
            let n = trait_item.name.clone();
            if i.items.iter().any(|m| m.name == n) {
                continue;
            }
            let did = i.trait_.as_ref().unwrap().def_id().unwrap();
            let assoc_link = AssocItemLink::GotoSource(did, &i.provided_trait_methods);

            doc_impl_item(
                w,
                cx,
                trait_item,
                parent,
                assoc_link,
                render_mode,
                true,
                outer_version,
                outer_const_version,
                None,
                show_def_docs,
                cache,
            );
        }
    }

    // If we've implemented a trait, then also emit documentation for all
    // default items which weren't overridden in the implementation block.
    // We don't emit documentation for default items if they appear in the
    // Implementations on Foreign Types or Implementors sections.
    if show_default_items {
        if let Some(t) = trait_ {
            render_default_items(
                w,
                cx,
                t,
                &i.inner_impl(),
                &i.impl_item,
                render_mode,
                outer_version,
                outer_const_version,
                show_def_docs,
                cache,
            );
        }
    }
    write!(w, "</div>");
}

fn item_opaque_ty(
    w: &mut Buffer,
    cx: &Context<'_>,
    it: &clean::Item,
    t: &clean::OpaqueTy,
    cache: &Cache,
) {
    write!(w, "<pre class=\"rust opaque\">");
    render_attributes(w, it, false);
    write!(
        w,
        "type {}{}{where_clause} = impl {bounds};</pre>",
        it.name.as_ref().unwrap(),
        t.generics.print(),
        where_clause = WhereClause { gens: &t.generics, indent: 0, end_newline: true },
        bounds = bounds(&t.bounds, false)
    );

    document(w, cx, it, None);

    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_trait_alias(
    w: &mut Buffer,
    cx: &Context<'_>,
    it: &clean::Item,
    t: &clean::TraitAlias,
    cache: &Cache,
) {
    write!(w, "<pre class=\"rust trait-alias\">");
    render_attributes(w, it, false);
    write!(
        w,
        "trait {}{}{} = {};</pre>",
        it.name.as_ref().unwrap(),
        t.generics.print(),
        WhereClause { gens: &t.generics, indent: 0, end_newline: true },
        bounds(&t.bounds, true)
    );

    document(w, cx, it, None);

    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_typedef(
    w: &mut Buffer,
    cx: &Context<'_>,
    it: &clean::Item,
    t: &clean::Typedef,
    cache: &Cache,
) {
    write!(w, "<pre class=\"rust typedef\">");
    render_attributes(w, it, false);
    write!(
        w,
        "type {}{}{where_clause} = {type_};</pre>",
        it.name.as_ref().unwrap(),
        t.generics.print(),
        where_clause = WhereClause { gens: &t.generics, indent: 0, end_newline: true },
        type_ = t.type_.print()
    );

    document(w, cx, it, None);

    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_foreign_type(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, cache: &Cache) {
    writeln!(w, "<pre class=\"rust foreigntype\">extern {{");
    render_attributes(w, it, false);
    write!(
        w,
        "    {}type {};\n}}</pre>",
        it.visibility.print_with_space(cx.tcx()),
        it.name.as_ref().unwrap(),
    );

    document(w, cx, it, None);

    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn print_sidebar(cx: &Context<'_>, it: &clean::Item, buffer: &mut Buffer, cache: &Cache) {
    let parentlen = cx.current.len() - if it.is_mod() { 1 } else { 0 };

    if it.is_struct()
        || it.is_trait()
        || it.is_primitive()
        || it.is_union()
        || it.is_enum()
        || it.is_mod()
        || it.is_typedef()
    {
        write!(
            buffer,
            "<p class=\"location\">{}{}</p>",
            match *it.kind {
                clean::StructItem(..) => "Struct ",
                clean::TraitItem(..) => "Trait ",
                clean::PrimitiveItem(..) => "Primitive Type ",
                clean::UnionItem(..) => "Union ",
                clean::EnumItem(..) => "Enum ",
                clean::TypedefItem(..) => "Type Definition ",
                clean::ForeignTypeItem => "Foreign Type ",
                clean::ModuleItem(..) =>
                    if it.is_crate() {
                        "Crate "
                    } else {
                        "Module "
                    },
                _ => "",
            },
            it.name.as_ref().unwrap()
        );
    }

    if it.is_crate() {
        if let Some(ref version) = cache.crate_version {
            write!(
                buffer,
                "<div class=\"block version\">\
                     <p>Version {}</p>\
                 </div>",
                Escape(version)
            );
        }
    }

    write!(buffer, "<div class=\"sidebar-elems\">");
    if it.is_crate() {
        write!(
            buffer,
            "<a id=\"all-types\" href=\"all.html\"><p>See all {}'s items</p></a>",
            it.name.as_ref().expect("crates always have a name")
        );
    }
    match *it.kind {
        clean::StructItem(ref s) => sidebar_struct(buffer, it, s),
        clean::TraitItem(ref t) => sidebar_trait(buffer, it, t),
        clean::PrimitiveItem(_) => sidebar_primitive(buffer, it),
        clean::UnionItem(ref u) => sidebar_union(buffer, it, u),
        clean::EnumItem(ref e) => sidebar_enum(buffer, it, e),
        clean::TypedefItem(_, _) => sidebar_typedef(buffer, it),
        clean::ModuleItem(ref m) => sidebar_module(buffer, &m.items),
        clean::ForeignTypeItem => sidebar_foreign_type(buffer, it),
        _ => (),
    }

    // The sidebar is designed to display sibling functions, modules and
    // other miscellaneous information. since there are lots of sibling
    // items (and that causes quadratic growth in large modules),
    // we refactor common parts into a shared JavaScript file per module.
    // still, we don't move everything into JS because we want to preserve
    // as much HTML as possible in order to allow non-JS-enabled browsers
    // to navigate the documentation (though slightly inefficiently).

    write!(buffer, "<p class=\"location\">");
    for (i, name) in cx.current.iter().take(parentlen).enumerate() {
        if i > 0 {
            write!(buffer, "::<wbr>");
        }
        write!(
            buffer,
            "<a href=\"{}index.html\">{}</a>",
            &cx.root_path()[..(cx.current.len() - i - 1) * 3],
            *name
        );
    }
    write!(buffer, "</p>");

    // Sidebar refers to the enclosing module, not this module.
    let relpath = if it.is_mod() { "../" } else { "" };
    write!(
        buffer,
        "<script>window.sidebarCurrent = {{\
                name: \"{name}\", \
                ty: \"{ty}\", \
                relpath: \"{path}\"\
            }};</script>",
        name = it.name.unwrap_or(kw::Invalid),
        ty = it.type_(),
        path = relpath
    );
    if parentlen == 0 {
        // There is no sidebar-items.js beyond the crate root path
        // FIXME maybe dynamic crate loading can be merged here
    } else {
        write!(buffer, "<script defer src=\"{path}sidebar-items.js\"></script>", path = relpath);
    }
    // Closes sidebar-elems div.
    write!(buffer, "</div>");
}

fn get_next_url(used_links: &mut FxHashSet<String>, url: String) -> String {
    if used_links.insert(url.clone()) {
        return url;
    }
    let mut add = 1;
    while !used_links.insert(format!("{}-{}", url, add)) {
        add += 1;
    }
    format!("{}-{}", url, add)
}

fn get_methods(
    i: &clean::Impl,
    for_deref: bool,
    used_links: &mut FxHashSet<String>,
    deref_mut: bool,
) -> Vec<String> {
    i.items
        .iter()
        .filter_map(|item| match item.name {
            Some(ref name) if !name.is_empty() && item.is_method() => {
                if !for_deref || should_render_item(item, deref_mut) {
                    Some(format!(
                        "<a href=\"#{}\">{}</a>",
                        get_next_url(used_links, format!("method.{}", name)),
                        name
                    ))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

// The point is to url encode any potential character from a type with genericity.
fn small_url_encode(s: &str) -> String {
    s.replace("<", "%3C")
        .replace(">", "%3E")
        .replace(" ", "%20")
        .replace("?", "%3F")
        .replace("'", "%27")
        .replace("&", "%26")
        .replace(",", "%2C")
        .replace(":", "%3A")
        .replace(";", "%3B")
        .replace("[", "%5B")
        .replace("]", "%5D")
        .replace("\"", "%22")
}

fn sidebar_assoc_items(it: &clean::Item) -> String {
    let mut out = String::new();
    let c = cache();
    if let Some(v) = c.impls.get(&it.def_id) {
        let mut used_links = FxHashSet::default();

        {
            let used_links_bor = &mut used_links;
            let mut ret = v
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(move |i| get_methods(i.inner_impl(), false, used_links_bor, false))
                .collect::<Vec<_>>();
            if !ret.is_empty() {
                // We want links' order to be reproducible so we don't use unstable sort.
                ret.sort();
                out.push_str(&format!(
                    "<a class=\"sidebar-title\" href=\"#implementations\">Methods</a>\
                     <div class=\"sidebar-links\">{}</div>",
                    ret.join("")
                ));
            }
        }

        if v.iter().any(|i| i.inner_impl().trait_.is_some()) {
            if let Some(impl_) = v
                .iter()
                .filter(|i| i.inner_impl().trait_.is_some())
                .find(|i| i.inner_impl().trait_.def_id() == c.deref_trait_did)
            {
                if let Some((target, real_target)) =
                    impl_.inner_impl().items.iter().find_map(|item| match *item.kind {
                        clean::TypedefItem(ref t, true) => Some(match *t {
                            clean::Typedef { item_type: Some(ref type_), .. } => (type_, &t.type_),
                            _ => (&t.type_, &t.type_),
                        }),
                        _ => None,
                    })
                {
                    let deref_mut = v
                        .iter()
                        .filter(|i| i.inner_impl().trait_.is_some())
                        .any(|i| i.inner_impl().trait_.def_id() == c.deref_mut_trait_did);
                    let inner_impl = target
                        .def_id()
                        .or(target
                            .primitive_type()
                            .and_then(|prim| c.primitive_locations.get(&prim).cloned()))
                        .and_then(|did| c.impls.get(&did));
                    if let Some(impls) = inner_impl {
                        out.push_str("<a class=\"sidebar-title\" href=\"#deref-methods\">");
                        out.push_str(&format!(
                            "Methods from {}&lt;Target={}&gt;",
                            Escape(&format!(
                                "{:#}",
                                impl_.inner_impl().trait_.as_ref().unwrap().print()
                            )),
                            Escape(&format!("{:#}", real_target.print()))
                        ));
                        out.push_str("</a>");
                        let mut ret = impls
                            .iter()
                            .filter(|i| i.inner_impl().trait_.is_none())
                            .flat_map(|i| {
                                get_methods(i.inner_impl(), true, &mut used_links, deref_mut)
                            })
                            .collect::<Vec<_>>();
                        // We want links' order to be reproducible so we don't use unstable sort.
                        ret.sort();
                        if !ret.is_empty() {
                            out.push_str(&format!(
                                "<div class=\"sidebar-links\">{}</div>",
                                ret.join("")
                            ));
                        }
                    }
                }
            }
            let format_impls = |impls: Vec<&Impl>| {
                let mut links = FxHashSet::default();

                let mut ret = impls
                    .iter()
                    .filter_map(|i| {
                        let is_negative_impl = is_negative_impl(i.inner_impl());
                        if let Some(ref i) = i.inner_impl().trait_ {
                            let i_display = format!("{:#}", i.print());
                            let out = Escape(&i_display);
                            let encoded = small_url_encode(&format!("{:#}", i.print()));
                            let generated = format!(
                                "<a href=\"#impl-{}\">{}{}</a>",
                                encoded,
                                if is_negative_impl { "!" } else { "" },
                                out
                            );
                            if links.insert(generated.clone()) { Some(generated) } else { None }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<String>>();
                ret.sort();
                ret.join("")
            };

            let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
                v.iter().partition::<Vec<_>, _>(|i| i.inner_impl().synthetic);
            let (blanket_impl, concrete): (Vec<&Impl>, Vec<&Impl>) = concrete
                .into_iter()
                .partition::<Vec<_>, _>(|i| i.inner_impl().blanket_impl.is_some());

            let concrete_format = format_impls(concrete);
            let synthetic_format = format_impls(synthetic);
            let blanket_format = format_impls(blanket_impl);

            if !concrete_format.is_empty() {
                out.push_str(
                    "<a class=\"sidebar-title\" href=\"#trait-implementations\">\
                        Trait Implementations</a>",
                );
                out.push_str(&format!("<div class=\"sidebar-links\">{}</div>", concrete_format));
            }

            if !synthetic_format.is_empty() {
                out.push_str(
                    "<a class=\"sidebar-title\" href=\"#synthetic-implementations\">\
                        Auto Trait Implementations</a>",
                );
                out.push_str(&format!("<div class=\"sidebar-links\">{}</div>", synthetic_format));
            }

            if !blanket_format.is_empty() {
                out.push_str(
                    "<a class=\"sidebar-title\" href=\"#blanket-implementations\">\
                        Blanket Implementations</a>",
                );
                out.push_str(&format!("<div class=\"sidebar-links\">{}</div>", blanket_format));
            }
        }
    }

    out
}

fn sidebar_struct(buf: &mut Buffer, it: &clean::Item, s: &clean::Struct) {
    let mut sidebar = String::new();
    let fields = get_struct_fields_name(&s.fields);

    if !fields.is_empty() {
        if let doctree::Plain = s.struct_type {
            sidebar.push_str(&format!(
                "<a class=\"sidebar-title\" href=\"#fields\">Fields</a>\
                 <div class=\"sidebar-links\">{}</div>",
                fields
            ));
        }
    }

    sidebar.push_str(&sidebar_assoc_items(it));

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn get_id_for_impl_on_foreign_type(for_: &clean::Type, trait_: &clean::Type) -> String {
    small_url_encode(&format!("impl-{:#}-for-{:#}", trait_.print(), for_.print()))
}

fn extract_for_impl_name(item: &clean::Item) -> Option<(String, String)> {
    match *item.kind {
        clean::ItemKind::ImplItem(ref i) => {
            if let Some(ref trait_) = i.trait_ {
                Some((
                    format!("{:#}", i.for_.print()),
                    get_id_for_impl_on_foreign_type(&i.for_, trait_),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_negative_impl(i: &clean::Impl) -> bool {
    i.polarity == Some(clean::ImplPolarity::Negative)
}

fn sidebar_trait(buf: &mut Buffer, it: &clean::Item, t: &clean::Trait) {
    let mut sidebar = String::new();

    let mut types = t
        .items
        .iter()
        .filter_map(|m| match m.name {
            Some(ref name) if m.is_associated_type() => {
                Some(format!("<a href=\"#associatedtype.{name}\">{name}</a>", name = name))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut consts = t
        .items
        .iter()
        .filter_map(|m| match m.name {
            Some(ref name) if m.is_associated_const() => {
                Some(format!("<a href=\"#associatedconstant.{name}\">{name}</a>", name = name))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut required = t
        .items
        .iter()
        .filter_map(|m| match m.name {
            Some(ref name) if m.is_ty_method() => {
                Some(format!("<a href=\"#tymethod.{name}\">{name}</a>", name = name))
            }
            _ => None,
        })
        .collect::<Vec<String>>();
    let mut provided = t
        .items
        .iter()
        .filter_map(|m| match m.name {
            Some(ref name) if m.is_method() => {
                Some(format!("<a href=\"#method.{0}\">{0}</a>", name))
            }
            _ => None,
        })
        .collect::<Vec<String>>();

    if !types.is_empty() {
        types.sort();
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#associated-types\">\
                Associated Types</a><div class=\"sidebar-links\">{}</div>",
            types.join("")
        ));
    }
    if !consts.is_empty() {
        consts.sort();
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#associated-const\">\
                Associated Constants</a><div class=\"sidebar-links\">{}</div>",
            consts.join("")
        ));
    }
    if !required.is_empty() {
        required.sort();
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#required-methods\">\
                Required Methods</a><div class=\"sidebar-links\">{}</div>",
            required.join("")
        ));
    }
    if !provided.is_empty() {
        provided.sort();
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#provided-methods\">\
                Provided Methods</a><div class=\"sidebar-links\">{}</div>",
            provided.join("")
        ));
    }

    let c = cache();

    if let Some(implementors) = c.implementors.get(&it.def_id) {
        let mut res = implementors
            .iter()
            .filter(|i| i.inner_impl().for_.def_id().map_or(false, |d| !c.paths.contains_key(&d)))
            .filter_map(|i| extract_for_impl_name(&i.impl_item))
            .collect::<Vec<_>>();

        if !res.is_empty() {
            res.sort();
            sidebar.push_str(&format!(
                "<a class=\"sidebar-title\" href=\"#foreign-impls\">\
                    Implementations on Foreign Types</a>\
                 <div class=\"sidebar-links\">{}</div>",
                res.into_iter()
                    .map(|(name, id)| format!("<a href=\"#{}\">{}</a>", id, Escape(&name)))
                    .collect::<Vec<_>>()
                    .join("")
            ));
        }
    }

    sidebar.push_str(&sidebar_assoc_items(it));

    sidebar.push_str("<a class=\"sidebar-title\" href=\"#implementors\">Implementors</a>");
    if t.is_auto {
        sidebar.push_str(
            "<a class=\"sidebar-title\" \
                href=\"#synthetic-implementors\">Auto Implementors</a>",
        );
    }

    write!(buf, "<div class=\"block items\">{}</div>", sidebar)
}

fn sidebar_primitive(buf: &mut Buffer, it: &clean::Item) {
    let sidebar = sidebar_assoc_items(it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn sidebar_typedef(buf: &mut Buffer, it: &clean::Item) {
    let sidebar = sidebar_assoc_items(it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn get_struct_fields_name(fields: &[clean::Item]) -> String {
    let mut fields = fields
        .iter()
        .filter(|f| matches!(*f.kind, clean::StructFieldItem(..)))
        .filter_map(|f| match f.name {
            Some(ref name) => {
                Some(format!("<a href=\"#structfield.{name}\">{name}</a>", name = name))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    fields.sort();
    fields.join("")
}

fn sidebar_union(buf: &mut Buffer, it: &clean::Item, u: &clean::Union) {
    let mut sidebar = String::new();
    let fields = get_struct_fields_name(&u.fields);

    if !fields.is_empty() {
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#fields\">Fields</a>\
             <div class=\"sidebar-links\">{}</div>",
            fields
        ));
    }

    sidebar.push_str(&sidebar_assoc_items(it));

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn sidebar_enum(buf: &mut Buffer, it: &clean::Item, e: &clean::Enum) {
    let mut sidebar = String::new();

    let mut variants = e
        .variants
        .iter()
        .filter_map(|v| match v.name {
            Some(ref name) => Some(format!("<a href=\"#variant.{name}\">{name}</a>", name = name)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !variants.is_empty() {
        variants.sort_unstable();
        sidebar.push_str(&format!(
            "<a class=\"sidebar-title\" href=\"#variants\">Variants</a>\
             <div class=\"sidebar-links\">{}</div>",
            variants.join(""),
        ));
    }

    sidebar.push_str(&sidebar_assoc_items(it));

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn item_ty_to_strs(ty: &ItemType) -> (&'static str, &'static str) {
    match *ty {
        ItemType::ExternCrate | ItemType::Import => ("reexports", "Re-exports"),
        ItemType::Module => ("modules", "Modules"),
        ItemType::Struct => ("structs", "Structs"),
        ItemType::Union => ("unions", "Unions"),
        ItemType::Enum => ("enums", "Enums"),
        ItemType::Function => ("functions", "Functions"),
        ItemType::Typedef => ("types", "Type Definitions"),
        ItemType::Static => ("statics", "Statics"),
        ItemType::Constant => ("constants", "Constants"),
        ItemType::Trait => ("traits", "Traits"),
        ItemType::Impl => ("impls", "Implementations"),
        ItemType::TyMethod => ("tymethods", "Type Methods"),
        ItemType::Method => ("methods", "Methods"),
        ItemType::StructField => ("fields", "Struct Fields"),
        ItemType::Variant => ("variants", "Variants"),
        ItemType::Macro => ("macros", "Macros"),
        ItemType::Primitive => ("primitives", "Primitive Types"),
        ItemType::AssocType => ("associated-types", "Associated Types"),
        ItemType::AssocConst => ("associated-consts", "Associated Constants"),
        ItemType::ForeignType => ("foreign-types", "Foreign Types"),
        ItemType::Keyword => ("keywords", "Keywords"),
        ItemType::OpaqueTy => ("opaque-types", "Opaque Types"),
        ItemType::ProcAttribute => ("attributes", "Attribute Macros"),
        ItemType::ProcDerive => ("derives", "Derive Macros"),
        ItemType::TraitAlias => ("trait-aliases", "Trait aliases"),
    }
}

fn sidebar_module(buf: &mut Buffer, items: &[clean::Item]) {
    let mut sidebar = String::new();

    if items.iter().any(|it| {
        it.type_() == ItemType::ExternCrate || (it.type_() == ItemType::Import && !it.is_stripped())
    }) {
        sidebar.push_str(&format!(
            "<li><a href=\"#{id}\">{name}</a></li>",
            id = "reexports",
            name = "Re-exports"
        ));
    }

    // ordering taken from item_module, reorder, where it prioritized elements in a certain order
    // to print its headings
    for &myty in &[
        ItemType::Primitive,
        ItemType::Module,
        ItemType::Macro,
        ItemType::Struct,
        ItemType::Enum,
        ItemType::Constant,
        ItemType::Static,
        ItemType::Trait,
        ItemType::Function,
        ItemType::Typedef,
        ItemType::Union,
        ItemType::Impl,
        ItemType::TyMethod,
        ItemType::Method,
        ItemType::StructField,
        ItemType::Variant,
        ItemType::AssocType,
        ItemType::AssocConst,
        ItemType::ForeignType,
        ItemType::Keyword,
    ] {
        if items.iter().any(|it| !it.is_stripped() && it.type_() == myty) {
            let (short, name) = item_ty_to_strs(&myty);
            sidebar.push_str(&format!(
                "<li><a href=\"#{id}\">{name}</a></li>",
                id = short,
                name = name
            ));
        }
    }

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\"><ul>{}</ul></div>", sidebar);
    }
}

fn sidebar_foreign_type(buf: &mut Buffer, it: &clean::Item) {
    let sidebar = sidebar_assoc_items(it);
    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar);
    }
}

fn item_macro(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, t: &clean::Macro) {
    wrap_into_docblock(w, |w| {
        w.write_str(&highlight::render_with_highlighting(
            t.source.clone(),
            Some("macro"),
            None,
            None,
            it.source.span().edition(),
        ))
    });
    document(w, cx, it, None)
}

fn item_proc_macro(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, m: &clean::ProcMacro) {
    let name = it.name.as_ref().expect("proc-macros always have names");
    match m.kind {
        MacroKind::Bang => {
            write!(w, "<pre class=\"rust macro\">");
            write!(w, "{}!() {{ /* proc-macro */ }}", name);
            write!(w, "</pre>");
        }
        MacroKind::Attr => {
            write!(w, "<pre class=\"rust attr\">");
            write!(w, "#[{}]", name);
            write!(w, "</pre>");
        }
        MacroKind::Derive => {
            write!(w, "<pre class=\"rust derive\">");
            write!(w, "#[derive({})]", name);
            if !m.helpers.is_empty() {
                writeln!(w, "\n{{");
                writeln!(w, "    // Attributes available to this derive:");
                for attr in &m.helpers {
                    writeln!(w, "    #[{}]", attr);
                }
                write!(w, "}}");
            }
            write!(w, "</pre>");
        }
    }
    document(w, cx, it, None)
}

fn item_primitive(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, cache: &Cache) {
    document(w, cx, it, None);
    render_assoc_items(w, cx, it, it.def_id, AssocItemRender::All, cache)
}

fn item_keyword(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item) {
    document(w, cx, it, None)
}

crate const BASIC_KEYWORDS: &str = "rust, rustlang, rust-lang";

fn make_item_keywords(it: &clean::Item) -> String {
    format!("{}, {}", BASIC_KEYWORDS, it.name.as_ref().unwrap())
}

/// Returns a list of all paths used in the type.
/// This is used to help deduplicate imported impls
/// for reexported types. If any of the contained
/// types are re-exported, we don't use the corresponding
/// entry from the js file, as inlining will have already
/// picked up the impl
fn collect_paths_for_type(first_ty: clean::Type) -> Vec<String> {
    let mut out = Vec::new();
    let mut visited = FxHashSet::default();
    let mut work = VecDeque::new();
    let cache = cache();

    work.push_back(first_ty);

    while let Some(ty) = work.pop_front() {
        if !visited.insert(ty.clone()) {
            continue;
        }

        match ty {
            clean::Type::ResolvedPath { did, .. } => {
                let get_extern = || cache.external_paths.get(&did).map(|s| s.0.clone());
                let fqp = cache.exact_paths.get(&did).cloned().or_else(get_extern);

                if let Some(path) = fqp {
                    out.push(path.join("::"));
                }
            }
            clean::Type::Tuple(tys) => {
                work.extend(tys.into_iter());
            }
            clean::Type::Slice(ty) => {
                work.push_back(*ty);
            }
            clean::Type::Array(ty, _) => {
                work.push_back(*ty);
            }
            clean::Type::RawPointer(_, ty) => {
                work.push_back(*ty);
            }
            clean::Type::BorrowedRef { type_, .. } => {
                work.push_back(*type_);
            }
            clean::Type::QPath { self_type, trait_, .. } => {
                work.push_back(*self_type);
                work.push_back(*trait_);
            }
            _ => {}
        }
    }
    out
}
