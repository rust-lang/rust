use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use askama::Template;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::def_id::{DefIdMap, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::{FileName, Symbol, sym};
use tracing::info;

use super::print_item::{full_path, item_path, print_item};
use super::sidebar::{ModuleLike, Sidebar, print_sidebar, sidebar_module_like};
use super::{AllTypes, LinkFromSrc, StylePath, collect_spans_and_sources, scrape_examples_help};
use crate::clean::types::ExternalLocation;
use crate::clean::utils::has_doc_flag;
use crate::clean::{self, ExternalCrate};
use crate::config::{ModuleSorting, RenderOptions, ShouldMerge};
use crate::docfs::{DocFS, PathError};
use crate::error::Error;
use crate::formats::FormatRenderer;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::escape::Escape;
use crate::html::format::join_with_double_colon;
use crate::html::layout::{self, BufDisplay};
use crate::html::markdown::{self, ErrorCodes, IdMap, plain_text_summary};
use crate::html::render::write_shared::write_shared;
use crate::html::url_parts_builder::UrlPartsBuilder;
use crate::html::{sources, static_files};
use crate::scrape_examples::AllCallLocations;
use crate::{DOC_RUST_LANG_ORG_VERSION, try_err};

/// Major driving force in all rustdoc rendering. This contains information
/// about where in the tree-like hierarchy rendering is occurring and controls
/// how the current page is being rendered.
///
/// It is intended that this context is a lightweight object which can be fairly
/// easily cloned because it is cloned per work-job (about once per item in the
/// rustdoc tree).
pub(crate) struct Context<'tcx> {
    /// Current hierarchy of components leading down to what's currently being
    /// rendered
    pub(crate) current: Vec<Symbol>,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    pub(crate) dst: PathBuf,
    /// Tracks section IDs for `Deref` targets so they match in both the main
    /// body and the sidebar.
    pub(super) deref_id_map: RefCell<DefIdMap<String>>,
    /// The map used to ensure all generated 'id=' attributes are unique.
    pub(super) id_map: RefCell<IdMap>,
    /// Shared mutable state.
    ///
    /// Issue for improving the situation: [#82381][]
    ///
    /// [#82381]: https://github.com/rust-lang/rust/issues/82381
    pub(crate) shared: SharedContext<'tcx>,
    /// Collection of all types with notable traits referenced in the current module.
    pub(crate) types_with_notable_traits: RefCell<FxIndexSet<clean::Type>>,
    /// Contains information that needs to be saved and reset after rendering an item which is
    /// not a module.
    pub(crate) info: ContextInfo,
}

/// This struct contains the information that needs to be reset between each
/// [`FormatRenderer::item`] call.
///
/// When we enter a new module, we set these values for the whole module but they might be updated
/// in each child item (especially if it's a module). So to prevent these changes to impact other
/// items rendering in the same module, we need to reset them to the module's set values.
#[derive(Clone, Copy)]
pub(crate) struct ContextInfo {
    /// A flag, which when `true`, will render pages which redirect to the
    /// real location of an item. This is used to allow external links to
    /// publicly reused items to redirect to the right location.
    pub(super) render_redirect_pages: bool,
    /// This flag indicates whether source links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    pub(crate) include_sources: bool,
    /// Field used during rendering, to know if we're inside an inlined item.
    pub(crate) is_inside_inlined_module: bool,
}

impl ContextInfo {
    fn new(include_sources: bool) -> Self {
        Self { render_redirect_pages: false, include_sources, is_inside_inlined_module: false }
    }
}

/// Shared mutable state used in [`Context`] and elsewhere.
pub(crate) struct SharedContext<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    /// The path to the crate root source minus the file name.
    /// Used for simplifying paths to the highlighted source code files.
    pub(crate) src_root: PathBuf,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon and added html).
    pub(crate) layout: layout::Layout,
    /// The local file sources we've emitted and their respective url-paths.
    pub(crate) local_sources: FxIndexMap<PathBuf, String>,
    /// Show the memory layout of types in the docs.
    pub(super) show_type_layout: bool,
    /// The base-URL of the issue tracker for when an item has been tagged with
    /// an issue number.
    pub(super) issue_tracker_base_url: Option<String>,
    /// The directories that have already been created in this doc run. Used to reduce the number
    /// of spurious `create_dir_all` calls.
    created_dirs: RefCell<FxHashSet<PathBuf>>,
    /// This flag indicates whether listings of modules (in the side bar and documentation itself)
    /// should be ordered alphabetically or in order of appearance (in the source code).
    pub(super) module_sorting: ModuleSorting,
    /// Additional CSS files to be added to the generated docs.
    pub(crate) style_files: Vec<StylePath>,
    /// Suffix to add on resource files (if suffix is "-v2" then "search-index.js" becomes
    /// "search-index-v2.js").
    pub(crate) resource_suffix: String,
    /// Optional path string to be used to load static files on output pages. If not set, uses
    /// combinations of `../` to reach the documentation root.
    pub(crate) static_root_path: Option<String>,
    /// The fs handle we are working with.
    pub(crate) fs: DocFS,
    pub(super) codes: ErrorCodes,
    pub(super) playground: Option<markdown::Playground>,
    all: RefCell<AllTypes>,
    /// Storage for the errors produced while generating documentation so they
    /// can be printed together at the end.
    errors: Receiver<String>,
    /// `None` by default, depends on the `generate-redirect-map` option flag. If this field is set
    /// to `Some(...)`, it'll store redirections and then generate a JSON file at the top level of
    /// the crate.
    redirections: Option<RefCell<FxHashMap<String, String>>>,

    /// Correspondence map used to link types used in the source code pages to allow to click on
    /// links to jump to the type's definition.
    pub(crate) span_correspondence_map: FxHashMap<rustc_span::Span, LinkFromSrc>,
    /// The [`Cache`] used during rendering.
    pub(crate) cache: Cache,
    pub(crate) call_locations: AllCallLocations,
    /// Controls whether we read / write to cci files in the doc root. Defaults read=true,
    /// write=true
    should_merge: ShouldMerge,
}

impl SharedContext<'_> {
    pub(crate) fn ensure_dir(&self, dst: &Path) -> Result<(), Error> {
        let mut dirs = self.created_dirs.borrow_mut();
        if !dirs.contains(dst) {
            try_err!(self.fs.create_dir_all(dst), dst);
            dirs.insert(dst.to_path_buf());
        }

        Ok(())
    }

    pub(crate) fn edition(&self) -> Edition {
        self.tcx.sess.edition()
    }
}

impl<'tcx> Context<'tcx> {
    pub(crate) fn tcx(&self) -> TyCtxt<'tcx> {
        self.shared.tcx
    }

    pub(crate) fn cache(&self) -> &Cache {
        &self.shared.cache
    }

    pub(super) fn sess(&self) -> &'tcx Session {
        self.shared.tcx.sess
    }

    pub(super) fn derive_id<S: AsRef<str> + ToString>(&self, id: S) -> String {
        self.id_map.borrow_mut().derive(id)
    }

    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    pub(super) fn root_path(&self) -> String {
        "../".repeat(self.current.len())
    }

    fn render_item(&mut self, it: &clean::Item, is_module: bool) -> String {
        let mut render_redirect_pages = self.info.render_redirect_pages;
        // If the item is stripped but inlined, links won't point to the item so no need to generate
        // a file for it.
        if it.is_stripped()
            && let Some(def_id) = it.def_id()
            && def_id.is_local()
        {
            if self.info.is_inside_inlined_module
                || self.shared.cache.inlined_items.contains(&def_id)
            {
                // For now we're forced to generate a redirect page for stripped items until
                // `record_extern_fqn` correctly points to external items.
                render_redirect_pages = true;
            }
        }
        let mut title = String::new();
        if !is_module {
            title.push_str(it.name.unwrap().as_str());
        }
        if !it.is_primitive() && !it.is_keyword() {
            if !is_module {
                title.push_str(" in ");
            }
            // No need to include the namespace for primitive types and keywords
            title.push_str(&join_with_double_colon(&self.current));
        };
        title.push_str(" - Rust");
        let tyname = it.type_();
        let desc = plain_text_summary(&it.doc_value(), &it.link_names(self.cache()));
        let desc = if !desc.is_empty() {
            desc
        } else if it.is_crate() {
            format!("API documentation for the Rust `{}` crate.", self.shared.layout.krate)
        } else {
            format!(
                "API documentation for the Rust `{name}` {tyname} in crate `{krate}`.",
                name = it.name.as_ref().unwrap(),
                krate = self.shared.layout.krate,
            )
        };
        let name;
        let tyname_s = if it.is_crate() {
            name = format!("{tyname} crate");
            name.as_str()
        } else {
            tyname.as_str()
        };

        if !render_redirect_pages {
            let content = print_item(self, it);
            let page = layout::Page {
                css_class: tyname_s,
                root_path: &self.root_path(),
                static_root_path: self.shared.static_root_path.as_deref(),
                title: &title,
                description: &desc,
                resource_suffix: &self.shared.resource_suffix,
                rust_logo: has_doc_flag(self.tcx(), LOCAL_CRATE.as_def_id(), sym::rust_logo),
            };
            layout::render(
                &self.shared.layout,
                &page,
                BufDisplay(|buf: &mut String| {
                    print_sidebar(self, it, buf);
                }),
                content,
                &self.shared.style_files,
            )
        } else {
            if let Some(&(ref names, ty)) = self.cache().paths.get(&it.item_id.expect_def_id()) {
                if self.current.len() + 1 != names.len()
                    || self.current.iter().zip(names.iter()).any(|(a, b)| a != b)
                {
                    // We checked that the redirection isn't pointing to the current file,
                    // preventing an infinite redirection loop in the generated
                    // documentation.

                    let path = fmt::from_fn(|f| {
                        for name in &names[..names.len() - 1] {
                            write!(f, "{name}/")?;
                        }
                        write!(f, "{}", item_path(ty, names.last().unwrap().as_str()))
                    });
                    match self.shared.redirections {
                        Some(ref redirections) => {
                            let mut current_path = String::new();
                            for name in &self.current {
                                current_path.push_str(name.as_str());
                                current_path.push('/');
                            }
                            let _ = write!(
                                current_path,
                                "{}",
                                item_path(ty, names.last().unwrap().as_str())
                            );
                            redirections.borrow_mut().insert(current_path, path.to_string());
                        }
                        None => {
                            return layout::redirect(&format!(
                                "{root}{path}",
                                root = self.root_path()
                            ));
                        }
                    }
                }
            }
            String::new()
        }
    }

    /// Construct a map of items shown in the sidebar to a plain-text summary of their docs.
    fn build_sidebar_items(&self, m: &clean::Module) -> BTreeMap<String, Vec<String>> {
        // BTreeMap instead of HashMap to get a sorted output
        let mut map: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut inserted: FxHashMap<ItemType, FxHashSet<Symbol>> = FxHashMap::default();

        for item in &m.items {
            if item.is_stripped() {
                continue;
            }

            let short = item.type_();
            let myname = match item.name {
                None => continue,
                Some(s) => s,
            };
            if inserted.entry(short).or_default().insert(myname) {
                let short = short.to_string();
                let myname = myname.to_string();
                map.entry(short).or_default().push(myname);
            }
        }

        match self.shared.module_sorting {
            ModuleSorting::Alphabetical => {
                for items in map.values_mut() {
                    items.sort();
                }
            }
            ModuleSorting::DeclarationOrder => {}
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
        self.href_from_span(item.span(self.tcx())?, true)
    }

    pub(crate) fn href_from_span(&self, span: clean::Span, with_lines: bool) -> Option<String> {
        let mut root = self.root_path();
        let mut path: String;
        let cnum = span.cnum(self.sess());

        // We can safely ignore synthetic `SourceFile`s.
        let file = match span.filename(self.sess()) {
            FileName::Real(ref path) => path.local_path_if_available().to_path_buf(),
            _ => return None,
        };
        let file = &file;

        let krate_sym;
        let (krate, path) = if cnum == LOCAL_CRATE {
            if let Some(path) = self.shared.local_sources.get(file) {
                (self.shared.layout.krate.as_str(), path)
            } else {
                return None;
            }
        } else {
            let (krate, src_root) = match *self.cache().extern_locations.get(&cnum)? {
                ExternalLocation::Local => {
                    let e = ExternalCrate { crate_num: cnum };
                    (e.name(self.tcx()), e.src_root(self.tcx()))
                }
                ExternalLocation::Remote(ref s) => {
                    root = s.to_string();
                    let e = ExternalCrate { crate_num: cnum };
                    (e.name(self.tcx()), e.src_root(self.tcx()))
                }
                ExternalLocation::Unknown => return None,
            };

            let href = RefCell::new(PathBuf::new());
            sources::clean_path(
                &src_root,
                file,
                |component| {
                    href.borrow_mut().push(component);
                },
                || {
                    href.borrow_mut().pop();
                },
            );

            path = href.into_inner().to_string_lossy().into_owned();

            if let Some(c) = path.as_bytes().last()
                && *c != b'/'
            {
                path.push('/');
            }

            let mut fname = file.file_name().expect("source has no filename").to_os_string();
            fname.push(".html");
            path.push_str(&fname.to_string_lossy());
            krate_sym = krate;
            (krate_sym.as_str(), &path)
        };

        let anchor = if with_lines {
            let loline = span.lo(self.sess()).line;
            let hiline = span.hi(self.sess()).line;
            format!(
                "#{}",
                if loline == hiline { loline.to_string() } else { format!("{loline}-{hiline}") }
            )
        } else {
            "".to_string()
        };
        Some(format!(
            "{root}src/{krate}/{path}{anchor}",
            root = Escape(&root),
            krate = krate,
            path = path,
            anchor = anchor
        ))
    }

    pub(crate) fn href_from_span_relative(
        &self,
        span: clean::Span,
        relative_to: &str,
    ) -> Option<String> {
        self.href_from_span(span, false).map(|s| {
            let mut url = UrlPartsBuilder::new();
            let mut dest_href_parts = s.split('/');
            let mut cur_href_parts = relative_to.split('/');
            for (cur_href_part, dest_href_part) in (&mut cur_href_parts).zip(&mut dest_href_parts) {
                if cur_href_part != dest_href_part {
                    url.push(dest_href_part);
                    break;
                }
            }
            for dest_href_part in dest_href_parts {
                url.push(dest_href_part);
            }
            let loline = span.lo(self.sess()).line;
            let hiline = span.hi(self.sess()).line;
            format!(
                "{}{}#{}",
                "../".repeat(cur_href_parts.count()),
                url.finish(),
                if loline == hiline { loline.to_string() } else { format!("{loline}-{hiline}") }
            )
        })
    }
}

/// Generates the documentation for `crate` into the directory `dst`
impl<'tcx> FormatRenderer<'tcx> for Context<'tcx> {
    fn descr() -> &'static str {
        "html"
    }

    const RUN_ON_MODULE: bool = true;
    type ModuleData = ContextInfo;

    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error> {
        // need to save a copy of the options for rendering the index page
        let md_opts = options.clone();
        let emit_crate = options.should_emit_crate();
        let RenderOptions {
            output,
            external_html,
            id_map,
            playground_url,
            module_sorting,
            themes: style_files,
            default_settings,
            extension_css,
            resource_suffix,
            static_root_path,
            generate_redirect_map,
            show_type_layout,
            generate_link_to_definition,
            call_locations,
            no_emit_shared,
            html_no_source,
            ..
        } = options;

        let src_root = match krate.src(tcx) {
            FileName::Real(ref p) => match p.local_path_if_available().parent() {
                Some(p) => p.to_path_buf(),
                None => PathBuf::new(),
            },
            _ => PathBuf::new(),
        };
        // If user passed in `--playground-url` arg, we fill in crate name here
        let mut playground = None;
        if let Some(url) = playground_url {
            playground = Some(markdown::Playground { crate_name: Some(krate.name(tcx)), url });
        }
        let krate_version = cache.crate_version.as_deref().unwrap_or_default();
        let mut layout = layout::Layout {
            logo: String::new(),
            favicon: String::new(),
            external_html,
            default_settings,
            krate: krate.name(tcx).to_string(),
            krate_version: krate_version.to_string(),
            css_file_extension: extension_css,
            scrape_examples_extension: !call_locations.is_empty(),
        };
        let mut issue_tracker_base_url = None;
        let mut include_sources = !html_no_source;

        // Crawl the crate attributes looking for attributes which control how we're
        // going to emit HTML
        for attr in krate.module.attrs.lists(sym::doc) {
            match (attr.name(), attr.value_str()) {
                (Some(sym::html_favicon_url), Some(s)) => {
                    layout.favicon = s.to_string();
                }
                (Some(sym::html_logo_url), Some(s)) => {
                    layout.logo = s.to_string();
                }
                (Some(sym::html_playground_url), Some(s)) => {
                    playground = Some(markdown::Playground {
                        crate_name: Some(krate.name(tcx)),
                        url: s.to_string(),
                    });
                }
                (Some(sym::issue_tracker_base_url), Some(s)) => {
                    issue_tracker_base_url = Some(s.to_string());
                }
                (Some(sym::html_no_source), None) if attr.is_word() => {
                    include_sources = false;
                }
                _ => {}
            }
        }

        let (local_sources, matches) = collect_spans_and_sources(
            tcx,
            &krate,
            &src_root,
            include_sources,
            generate_link_to_definition,
        );

        let (sender, receiver) = channel();
        let scx = SharedContext {
            tcx,
            src_root,
            local_sources,
            issue_tracker_base_url,
            layout,
            created_dirs: Default::default(),
            module_sorting,
            style_files,
            resource_suffix,
            static_root_path,
            fs: DocFS::new(sender),
            codes: ErrorCodes::from(options.unstable_features.is_nightly_build()),
            playground,
            all: RefCell::new(AllTypes::new()),
            errors: receiver,
            redirections: if generate_redirect_map { Some(Default::default()) } else { None },
            show_type_layout,
            span_correspondence_map: matches,
            cache,
            call_locations,
            should_merge: options.should_merge,
        };

        let dst = output;
        scx.ensure_dir(&dst)?;

        let mut cx = Context {
            current: Vec::new(),
            dst,
            id_map: RefCell::new(id_map),
            deref_id_map: Default::default(),
            shared: scx,
            types_with_notable_traits: RefCell::new(FxIndexSet::default()),
            info: ContextInfo::new(include_sources),
        };

        if emit_crate {
            sources::render(&mut cx, &krate)?;
        }

        if !no_emit_shared {
            write_shared(&mut cx, &krate, &md_opts, tcx)?;
        }

        Ok((cx, krate))
    }

    fn save_module_data(&mut self) -> Self::ModuleData {
        self.deref_id_map.borrow_mut().clear();
        self.id_map.borrow_mut().clear();
        self.types_with_notable_traits.borrow_mut().clear();
        self.info
    }

    fn restore_module_data(&mut self, info: Self::ModuleData) {
        self.info = info;
    }

    fn after_krate(&mut self) -> Result<(), Error> {
        let crate_name = self.tcx().crate_name(LOCAL_CRATE);
        let final_file = self.dst.join(crate_name.as_str()).join("all.html");
        let settings_file = self.dst.join("settings.html");
        let help_file = self.dst.join("help.html");
        let scrape_examples_help_file = self.dst.join("scrape-examples-help.html");

        let mut root_path = self.dst.to_str().expect("invalid path").to_owned();
        if !root_path.ends_with('/') {
            root_path.push('/');
        }
        let shared = &self.shared;
        let mut page = layout::Page {
            title: "List of all items in this crate",
            css_class: "mod sys",
            root_path: "../",
            static_root_path: shared.static_root_path.as_deref(),
            description: "List of all items in this crate",
            resource_suffix: &shared.resource_suffix,
            rust_logo: has_doc_flag(self.tcx(), LOCAL_CRATE.as_def_id(), sym::rust_logo),
        };
        let all = shared.all.replace(AllTypes::new());
        let mut sidebar = String::new();

        // all.html is not customizable, so a blank id map is fine
        let blocks = sidebar_module_like(all.item_sections(), &mut IdMap::new(), ModuleLike::Crate);
        let bar = Sidebar {
            title_prefix: "",
            title: "",
            is_crate: false,
            is_mod: false,
            parent_is_crate: false,
            blocks: vec![blocks],
            path: String::new(),
        };

        bar.render_into(&mut sidebar).unwrap();

        let v = layout::render(&shared.layout, &page, sidebar, all.print(), &shared.style_files);
        shared.fs.write(final_file, v)?;

        // if to avoid writing help, settings files to doc root unless we're on the final invocation
        if shared.should_merge.write_rendered_cci {
            // Generating settings page.
            page.title = "Settings";
            page.description = "Settings of Rustdoc";
            page.root_path = "./";
            page.rust_logo = true;

            let sidebar = "<h2 class=\"location\">Settings</h2><div class=\"sidebar-elems\"></div>";
            let v = layout::render(
                &shared.layout,
                &page,
                sidebar,
                fmt::from_fn(|buf| {
                    write!(
                        buf,
                        "<div class=\"main-heading\">\
                         <h1>Rustdoc settings</h1>\
                         <span class=\"out-of-band\">\
                             <a id=\"back\" href=\"javascript:void(0)\" onclick=\"history.back();\">\
                                Back\
                            </a>\
                         </span>\
                         </div>\
                         <noscript>\
                            <section>\
                                You need to enable JavaScript be able to update your settings.\
                            </section>\
                         </noscript>\
                         <script defer src=\"{static_root_path}{settings_js}\"></script>",
                        static_root_path = page.get_static_root_path(),
                        settings_js = static_files::STATIC_FILES.settings_js,
                    )?;
                    // Pre-load all theme CSS files, so that switching feels seamless.
                    //
                    // When loading settings.html as a popover, the equivalent HTML is
                    // generated in main.js.
                    for file in &shared.style_files {
                        if let Ok(theme) = file.basename() {
                            write!(
                                buf,
                                "<link rel=\"preload\" href=\"{root_path}{theme}{suffix}.css\" \
                                    as=\"style\">",
                                root_path = page.static_root_path.unwrap_or(""),
                                suffix = page.resource_suffix,
                            )?;
                        }
                    }
                    Ok(())
                }),
                &shared.style_files,
            );
            shared.fs.write(settings_file, v)?;

            // Generating help page.
            page.title = "Help";
            page.description = "Documentation for Rustdoc";
            page.root_path = "./";
            page.rust_logo = true;

            let sidebar = "<h2 class=\"location\">Help</h2><div class=\"sidebar-elems\"></div>";
            let v = layout::render(
                &shared.layout,
                &page,
                sidebar,
                format_args!(
                    "<div class=\"main-heading\">\
                        <h1>Rustdoc help</h1>\
                        <span class=\"out-of-band\">\
                            <a id=\"back\" href=\"javascript:void(0)\" onclick=\"history.back();\">\
                            Back\
                        </a>\
                        </span>\
                        </div>\
                        <noscript>\
                        <section>\
                            <p>You need to enable JavaScript to use keyboard commands or search.</p>\
                            <p>For more information, browse the <a href=\"{DOC_RUST_LANG_ORG_VERSION}/rustdoc/\">rustdoc handbook</a>.</p>\
                        </section>\
                        </noscript>",
                ),
                &shared.style_files,
            );
            shared.fs.write(help_file, v)?;
        }

        // if to avoid writing files to doc root unless we're on the final invocation
        if shared.layout.scrape_examples_extension && shared.should_merge.write_rendered_cci {
            page.title = "About scraped examples";
            page.description = "How the scraped examples feature works in Rustdoc";
            let v = layout::render(
                &shared.layout,
                &page,
                "",
                scrape_examples_help(shared),
                &shared.style_files,
            );
            shared.fs.write(scrape_examples_help_file, v)?;
        }

        if let Some(ref redirections) = shared.redirections
            && !redirections.borrow().is_empty()
        {
            let redirect_map_path = self.dst.join(crate_name.as_str()).join("redirect-map.json");
            let paths = serde_json::to_string(&*redirections.borrow()).unwrap();
            shared.ensure_dir(&self.dst.join(crate_name.as_str()))?;
            shared.fs.write(redirect_map_path, paths)?;
        }

        // Flush pending errors.
        self.shared.fs.close();
        let nb_errors = self.shared.errors.iter().map(|err| self.tcx().dcx().err(err)).count();
        if nb_errors > 0 {
            Err(Error::new(io::Error::new(io::ErrorKind::Other, "I/O error"), ""))
        } else {
            Ok(())
        }
    }

    fn mod_item_in(&mut self, item: &clean::Item) -> Result<(), Error> {
        // Stripped modules survive the rustdoc passes (i.e., `strip-private`)
        // if they contain impls for public types. These modules can also
        // contain items such as publicly re-exported structures.
        //
        // External crates will provide links to these structures, so
        // these modules are recursed into, but not rendered normally
        // (a flag on the context).
        if !self.info.render_redirect_pages {
            self.info.render_redirect_pages = item.is_stripped();
        }
        let item_name = item.name.unwrap();
        self.dst.push(item_name.as_str());
        self.current.push(item_name);

        info!("Recursing into {}", self.dst.display());

        if !item.is_stripped() {
            let buf = self.render_item(item, true);
            // buf will be empty if the module is stripped and there is no redirect for it
            if !buf.is_empty() {
                self.shared.ensure_dir(&self.dst)?;
                let joint_dst = self.dst.join("index.html");
                self.shared.fs.write(joint_dst, buf)?;
            }
        }
        if !self.info.is_inside_inlined_module {
            if let Some(def_id) = item.def_id()
                && self.cache().inlined_items.contains(&def_id)
            {
                self.info.is_inside_inlined_module = true;
            }
        } else if !self.cache().document_hidden && item.is_doc_hidden() {
            // We're not inside an inlined module anymore since this one cannot be re-exported.
            self.info.is_inside_inlined_module = false;
        }

        // Render sidebar-items.js used throughout this module.
        if !self.info.render_redirect_pages {
            let (clean::StrippedItem(box clean::ModuleItem(ref module))
            | clean::ModuleItem(ref module)) = item.kind
            else {
                unreachable!()
            };
            let items = self.build_sidebar_items(module);
            let js_dst = self.dst.join(format!("sidebar-items{}.js", self.shared.resource_suffix));
            let v = format!("window.SIDEBAR_ITEMS = {};", serde_json::to_string(&items).unwrap());
            self.shared.fs.write(js_dst, v)?;
        }
        Ok(())
    }

    fn mod_item_out(&mut self) -> Result<(), Error> {
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
        if !self.info.render_redirect_pages {
            self.info.render_redirect_pages = item.is_stripped();
        }

        let buf = self.render_item(&item, false);
        // buf will be empty if the item is stripped and there is no redirect for it
        if !buf.is_empty() {
            let name = item.name.as_ref().unwrap();
            let item_type = item.type_();
            let file_name = item_path(item_type, name.as_str()).to_string();
            self.shared.ensure_dir(&self.dst)?;
            let joint_dst = self.dst.join(&file_name);
            self.shared.fs.write(joint_dst, buf)?;

            if !self.info.render_redirect_pages {
                self.shared.all.borrow_mut().append(full_path(self, &item), &item_type);
            }
            // If the item is a macro, redirect from the old macro URL (with !)
            // to the new one (without).
            if item_type == ItemType::Macro {
                let redir_name = format!("{item_type}.{name}!.html");
                if let Some(ref redirections) = self.shared.redirections {
                    let crate_name = &self.shared.layout.krate;
                    redirections.borrow_mut().insert(
                        format!("{crate_name}/{redir_name}"),
                        format!("{crate_name}/{file_name}"),
                    );
                } else {
                    let v = layout::redirect(&file_name);
                    let redir_dst = self.dst.join(redir_name);
                    self.shared.fs.write(redir_dst, v)?;
                }
            }
        }

        Ok(())
    }

    fn cache(&self) -> &Cache {
        &self.shared.cache
    }
}
