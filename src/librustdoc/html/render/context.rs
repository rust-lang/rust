use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::source_map::FileName;
use rustc_span::symbol::sym;

use super::cache::{build_index, ExternalLocation};
use super::print_item::{full_path, item_path, print_item};
use super::write_shared::write_shared;
use super::{
    print_sidebar, settings, AllTypes, NameDoc, SharedContext, StylePath, BASIC_KEYWORDS,
    CURRENT_DEPTH, INITIAL_IDS,
};

use crate::clean::{self, AttributesExt};
use crate::config::RenderOptions;
use crate::docfs::{DocFS, PathError};
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::formats::FormatRenderer;
use crate::html::escape::Escape;
use crate::html::format::Buffer;
use crate::html::markdown::{self, plain_text_summary, ErrorCodes, IdMap};
use crate::html::{layout, sources};

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
    /// `None` by default, depends on the `generate-redirect-map` option flag. If this field is set
    /// to `Some(...)`, it'll store redirections and then generate a JSON file at the top level of
    /// the crate.
    crate redirections: Option<Rc<RefCell<FxHashMap<String, String>>>>,
    /// The map used to ensure all generated 'id=' attributes are unique.
    pub(super) id_map: Rc<RefCell<IdMap>>,
    /// Tracks section IDs for `Deref` targets so they match in both the main
    /// body and the sidebar.
    pub(super) deref_id_map: Rc<RefCell<FxHashMap<DefId, String>>>,
    crate shared: Arc<SharedContext<'tcx>>,
    all: Rc<RefCell<AllTypes>>,
    /// Storage for the errors produced while generating documentation so they
    /// can be printed together at the end.
    crate errors: Rc<Receiver<String>>,
    crate cache: Rc<Cache>,
}

impl<'tcx> Context<'tcx> {
    pub(super) fn path(&self, filename: &str) -> PathBuf {
        // We use splitn vs Path::extension here because we might get a filename
        // like `style.min.css` and we want to process that into
        // `style-suffix.min.css`.  Path::extension would just return `css`
        // which would result in `style.min-suffix.css` which isn't what we
        // want.
        let (base, ext) = filename.split_once('.').unwrap();
        let filename = format!("{}{}.{}", base, self.shared.resource_suffix, ext);
        self.dst.join(&filename)
    }

    pub(super) fn tcx(&self) -> TyCtxt<'tcx> {
        self.shared.tcx
    }

    fn sess(&self) -> &'tcx Session {
        &self.shared.tcx.sess
    }

    pub(super) fn derive_id(&self, id: String) -> String {
        let mut map = self.id_map.borrow_mut();
        map.derive(id)
    }

    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    pub(super) fn root_path(&self) -> String {
        "../".repeat(self.current.len())
    }

    fn render_item(&self, it: &clean::Item, pushname: bool) -> String {
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
        let desc = it.doc_value().as_ref().map(|doc| plain_text_summary(&doc));
        let desc = if let Some(desc) = desc {
            desc
        } else if it.is_crate() {
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
            self.id_map.borrow_mut().populate(&INITIAL_IDS);
        }

        if !self.render_redirect_pages {
            layout::render(
                &self.shared.layout,
                &page,
                |buf: &mut _| print_sidebar(self, it, buf),
                |buf: &mut _| print_item(self, it, buf),
                &self.shared.style_files,
            )
        } else {
            if let Some(&(ref names, ty)) = self.cache.paths.get(&it.def_id) {
                let mut path = String::new();
                for name in &names[..names.len() - 1] {
                    path.push_str(name);
                    path.push('/');
                }
                path.push_str(&item_path(ty, names.last().unwrap()));
                match self.redirections {
                    Some(ref redirections) => {
                        let mut current_path = String::new();
                        for name in &self.current {
                            current_path.push_str(name);
                            current_path.push('/');
                        }
                        current_path.push_str(&item_path(ty, names.last().unwrap()));
                        redirections.borrow_mut().insert(current_path, path);
                    }
                    None => return layout::redirect(&format!("{}{}", self.root_path(), path)),
                }
            }
            String::new()
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
                Some(item.doc_value().map_or_else(String::new, |s| plain_text_summary(&s))),
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
    pub(super) fn src_href(&self, item: &clean::Item) -> Option<String> {
        if item.source.is_dummy() {
            return None;
        }
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
            let (krate, src_root) = match *self.cache.extern_locations.get(&cnum)? {
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

/// Generates the documentation for `crate` into the directory `dst`
impl<'tcx> FormatRenderer<'tcx> for Context<'tcx> {
    fn descr() -> &'static str {
        "html"
    }

    fn init(
        mut krate: clean::Crate,
        options: RenderOptions,
        edition: Edition,
        mut cache: Cache,
        tcx: TyCtxt<'tcx>,
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
            generate_redirect_map,
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
        let index = build_index(&krate, &mut cache, tcx);

        let mut cx = Context {
            current: Vec::new(),
            dst,
            render_redirect_pages: false,
            id_map: Rc::new(RefCell::new(id_map)),
            deref_id_map: Rc::new(RefCell::new(FxHashMap::default())),
            shared: Arc::new(scx),
            all: Rc::new(RefCell::new(AllTypes::new())),
            errors: Rc::new(receiver),
            cache: Rc::new(cache),
            redirections: if generate_redirect_map { Some(Default::default()) } else { None },
        };

        CURRENT_DEPTH.with(|s| s.set(0));

        // Write shared runs within a flock; disable thread dispatching of IO temporarily.
        Arc::get_mut(&mut cx.shared).unwrap().fs.set_sync_only(true);
        write_shared(&cx, &krate, index, &md_opts)?;
        Arc::get_mut(&mut cx.shared).unwrap().fs.set_sync_only(false);
        Ok((cx, krate))
    }

    fn after_krate(
        &mut self,
        krate: &clean::Crate,
        diag: &rustc_errors::Handler,
    ) -> Result<(), Error> {
        let final_file = self.dst.join(&*krate.name.as_str()).join("all.html");
        let settings_file = self.dst.join("settings.html");
        let crate_name = krate.name;

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
        let sidebar = if let Some(ref version) = self.cache.crate_version {
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
        if let Some(redirections) = self.redirections.take() {
            if !redirections.borrow().is_empty() {
                let redirect_map_path =
                    self.dst.join(&*krate.name.as_str()).join("redirect-map.json");
                let paths = serde_json::to_string(&*redirections.borrow()).unwrap();
                self.shared.ensure_dir(&self.dst.join(&*krate.name.as_str()))?;
                self.shared.fs.write(&redirect_map_path, paths.as_bytes())?;
            }
        }

        // Flush pending errors.
        Arc::get_mut(&mut self.shared).unwrap().fs.close();
        let nb_errors = self.errors.iter().map(|err| diag.struct_err(&err).emit()).count();
        if nb_errors > 0 {
            Err(Error::new(io::Error::new(io::ErrorKind::Other, "I/O error"), ""))
        } else {
            Ok(())
        }
    }

    fn mod_item_in(&mut self, item: &clean::Item, item_name: &str) -> Result<(), Error> {
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

        let buf = self.render_item(item, false);
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

    fn item(&mut self, item: clean::Item) -> Result<(), Error> {
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

        let buf = self.render_item(&item, true);
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
                if let Some(ref redirections) = self.redirections {
                    let crate_name = &self.shared.layout.krate;
                    redirections.borrow_mut().insert(
                        format!("{}/{}", crate_name, redir_name),
                        format!("{}/{}", crate_name, file_name),
                    );
                } else {
                    let v = layout::redirect(file_name);
                    let redir_dst = self.dst.join(redir_name);
                    self.shared.fs.write(&redir_dst, v.as_bytes())?;
                }
            }
        }
        Ok(())
    }

    fn cache(&self) -> &Cache {
        &self.cache
    }
}

fn make_item_keywords(it: &clean::Item) -> String {
    format!("{}, {}", BASIC_KEYWORDS, it.name.as_ref().unwrap())
}
