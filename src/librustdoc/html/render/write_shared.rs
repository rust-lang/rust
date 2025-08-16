//! Rustdoc writes aut two kinds of shared files:
//!  - Static files, which are embedded in the rustdoc binary and are written with a
//!    filename that includes a hash of their contents. These will always have a new
//!    URL if the contents change, so they are safe to cache with the
//!    `Cache-Control: immutable` directive. They are written under the static.files/
//!    directory and are written when --emit-type is empty (default) or contains
//!    "toolchain-specific". If using the --static-root-path flag, it should point
//!    to a URL path prefix where each of these filenames can be fetched.
//!  - Invocation specific files. These are generated based on the crate(s) being
//!    documented. Their filenames need to be predictable without knowing their
//!    contents, so they do not include a hash in their filename and are not safe to
//!    cache with `Cache-Control: immutable`. They include the contents of the
//!    --resource-suffix flag and are emitted when --emit-type is empty (default)
//!    or contains "invocation-specific".

use std::cell::RefCell;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Write as _};
use std::iter::once;
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::rc::{Rc, Weak};
use std::str::FromStr;
use std::{fmt, fs};

use indexmap::IndexMap;
use regex::Regex;
use rustc_ast::join_path_syms;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_span::Symbol;
use rustc_span::def_id::DefId;
use serde::de::DeserializeOwned;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};

use super::{Context, RenderMode, collect_paths_for_type, ensure_trailing_slash};
use crate::clean::{Crate, Item, ItemId, ItemKind};
use crate::config::{EmitType, PathToParts, RenderOptions, ShouldMerge};
use crate::docfs::PathError;
use crate::error::Error;
use crate::formats::Impl;
use crate::formats::item_type::ItemType;
use crate::html::layout;
use crate::html::render::ordered_json::{EscapedJson, OrderedJson};
use crate::html::render::search_index::{SerializedSearchIndex, build_index};
use crate::html::render::sorted_template::{self, FileFormat, SortedTemplate};
use crate::html::render::{AssocItemLink, ImplRenderingParameters, StylePath};
use crate::html::static_files::{self, suffix_path};
use crate::visit::DocVisitor;
use crate::{try_err, try_none};

pub(crate) fn write_shared(
    cx: &mut Context<'_>,
    krate: &Crate,
    opt: &RenderOptions,
    tcx: TyCtxt<'_>,
) -> Result<(), Error> {
    // NOTE(EtomicBomb): I don't think we need sync here because no read-after-write?
    cx.shared.fs.set_sync_only(true);
    let lock_file = cx.dst.join(".lock");
    // Write shared runs within a flock; disable thread dispatching of IO temporarily.
    let _lock = try_err!(flock::Lock::new(&lock_file, true, true, true), &lock_file);

    let search_index =
        build_index(krate, &mut cx.shared.cache, tcx, &cx.dst, &cx.shared.resource_suffix)?;

    let crate_name = krate.name(cx.tcx());
    let crate_name = crate_name.as_str(); // rand
    let crate_name_json = OrderedJson::serialize(crate_name).unwrap(); // "rand"
    let external_crates = hack_get_external_crate_names(&cx.dst, &cx.shared.resource_suffix)?;
    let info = CrateInfo {
        version: CrateInfoVersion::V2,
        src_files_js: SourcesPart::get(cx, &crate_name_json)?,
        search_index,
        all_crates: AllCratesPart::get(crate_name_json.clone(), &cx.shared.resource_suffix)?,
        crates_index: CratesIndexPart::get(crate_name, &external_crates)?,
        trait_impl: TraitAliasPart::get(cx, &crate_name_json)?,
        type_impl: TypeAliasPart::get(cx, krate, &crate_name_json)?,
    };

    if let Some(parts_out_dir) = &opt.parts_out_dir {
        create_parents(&parts_out_dir.0)?;
        try_err!(
            fs::write(&parts_out_dir.0, serde_json::to_string(&info).unwrap()),
            &parts_out_dir.0
        );
    }

    let mut crates = CrateInfo::read_many(&opt.include_parts_dir)?;
    crates.push(info);

    if opt.should_merge.write_rendered_cci {
        write_not_crate_specific(
            &crates,
            &cx.dst,
            opt,
            &cx.shared.style_files,
            cx.shared.layout.css_file_extension.as_deref(),
            &cx.shared.resource_suffix,
            cx.info.include_sources,
        )?;
        match &opt.index_page {
            Some(index_page) if opt.enable_index_page => {
                let mut md_opts = opt.clone();
                md_opts.output = cx.dst.clone();
                md_opts.external_html = cx.shared.layout.external_html.clone();
                try_err!(
                    crate::markdown::render_and_write(index_page, md_opts, cx.shared.edition()),
                    &index_page
                );
            }
            None if opt.enable_index_page => {
                write_rendered_cci::<CratesIndexPart, _>(
                    || CratesIndexPart::blank(cx),
                    &cx.dst,
                    &crates,
                    &opt.should_merge,
                )?;
            }
            _ => {} // they don't want an index page
        }
    }

    cx.shared.fs.set_sync_only(false);
    Ok(())
}

/// Writes files that are written directly to the `--out-dir`, without the prefix from the current
/// crate. These are the rendered cross-crate files that encode info from multiple crates (e.g.
/// search index), and the static files.
pub(crate) fn write_not_crate_specific(
    crates: &[CrateInfo],
    dst: &Path,
    opt: &RenderOptions,
    style_files: &[StylePath],
    css_file_extension: Option<&Path>,
    resource_suffix: &str,
    include_sources: bool,
) -> Result<(), Error> {
    write_rendered_cross_crate_info(crates, dst, opt, include_sources, resource_suffix)?;
    write_static_files(dst, opt, style_files, css_file_extension, resource_suffix)?;
    Ok(())
}

fn write_rendered_cross_crate_info(
    crates: &[CrateInfo],
    dst: &Path,
    opt: &RenderOptions,
    include_sources: bool,
    resource_suffix: &str,
) -> Result<(), Error> {
    let m = &opt.should_merge;
    if opt.should_emit_crate() {
        if include_sources {
            write_rendered_cci::<SourcesPart, _>(SourcesPart::blank, dst, crates, m)?;
        }
        crates
            .iter()
            .fold(SerializedSearchIndex::default(), |a, b| a.union(&b.search_index))
            .sort()
            .write_to(dst, resource_suffix)?;
        write_rendered_cci::<AllCratesPart, _>(AllCratesPart::blank, dst, crates, m)?;
    }
    write_rendered_cci::<TraitAliasPart, _>(TraitAliasPart::blank, dst, crates, m)?;
    write_rendered_cci::<TypeAliasPart, _>(TypeAliasPart::blank, dst, crates, m)?;
    Ok(())
}

/// Writes the static files, the style files, and the css extensions.
/// Have to be careful about these, because they write to the root out dir.
fn write_static_files(
    dst: &Path,
    opt: &RenderOptions,
    style_files: &[StylePath],
    css_file_extension: Option<&Path>,
    resource_suffix: &str,
) -> Result<(), Error> {
    let static_dir = dst.join("static.files");
    try_err!(fs::create_dir_all(&static_dir), &static_dir);

    // Handle added third-party themes
    for entry in style_files {
        let theme = entry.basename()?;
        let extension =
            try_none!(try_none!(entry.path.extension(), &entry.path).to_str(), &entry.path);

        // Skip the official themes. They are written below as part of STATIC_FILES_LIST.
        if matches!(theme.as_str(), "light" | "dark" | "ayu") {
            continue;
        }

        let bytes = try_err!(fs::read(&entry.path), &entry.path);
        let filename = format!("{theme}{resource_suffix}.{extension}");
        let dst_filename = dst.join(filename);
        try_err!(fs::write(&dst_filename, bytes), &dst_filename);
    }

    // When the user adds their own CSS files with --extend-css, we write that as an
    // invocation-specific file (that is, with a resource suffix).
    if let Some(css) = css_file_extension {
        let buffer = try_err!(fs::read_to_string(css), css);
        let path = static_files::suffix_path("theme.css", resource_suffix);
        let dst_path = dst.join(path);
        try_err!(fs::write(&dst_path, buffer), &dst_path);
    }

    if opt.emit.is_empty() || opt.emit.contains(&EmitType::Toolchain) {
        static_files::for_each(|f: &static_files::StaticFile| {
            let filename = static_dir.join(f.output_filename());
            let contents: &[u8] =
                if opt.disable_minification { f.src_bytes } else { f.minified_bytes };
            fs::write(&filename, contents).map_err(|e| PathError::new(e, &filename))
        })?;
    }

    Ok(())
}

/// Contains pre-rendered contents to insert into the CCI template
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct CrateInfo {
    version: CrateInfoVersion,
    src_files_js: PartsAndLocations<SourcesPart>,
    search_index: SerializedSearchIndex,
    all_crates: PartsAndLocations<AllCratesPart>,
    crates_index: PartsAndLocations<CratesIndexPart>,
    trait_impl: PartsAndLocations<TraitAliasPart>,
    type_impl: PartsAndLocations<TypeAliasPart>,
}

impl CrateInfo {
    /// Read all of the crate info from its location on the filesystem
    pub(crate) fn read_many(parts_paths: &[PathToParts]) -> Result<Vec<Self>, Error> {
        parts_paths
            .iter()
            .map(|parts_path| {
                let path = &parts_path.0;
                let parts = try_err!(fs::read(path), &path);
                let parts: CrateInfo = try_err!(serde_json::from_slice(&parts), &path);
                Ok::<_, Error>(parts)
            })
            .collect::<Result<Vec<CrateInfo>, Error>>()
    }
}

/// Version for the format of the crate-info file.
///
/// This enum should only ever have one variant, representing the current version.
/// Gives pretty good error message about expecting the current version on deserialize.
///
/// Must be incremented (V2, V3, etc.) upon any changes to the search index or CrateInfo,
/// to provide better diagnostics about including an invalid file.
#[derive(Serialize, Deserialize, Clone, Debug)]
enum CrateInfoVersion {
    V2,
}

/// Paths (relative to the doc root) and their pre-merge contents
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(transparent)]
struct PartsAndLocations<P> {
    parts: Vec<(PathBuf, P)>,
}

impl<P> Default for PartsAndLocations<P> {
    fn default() -> Self {
        Self { parts: Vec::default() }
    }
}

impl<T, U> PartsAndLocations<Part<T, U>> {
    fn push(&mut self, path: PathBuf, item: U) {
        self.parts.push((path, Part { _artifact: PhantomData, item }));
    }

    /// Singleton part, one file
    fn with(path: PathBuf, part: U) -> Self {
        let mut ret = Self::default();
        ret.push(path, part);
        ret
    }
}

/// A piece of one of the shared artifacts for documentation (search index, sources, alias list, etc.)
///
/// Merged at a user specified time and written to the `doc/` directory
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(transparent)]
struct Part<T, U> {
    #[serde(skip)]
    _artifact: PhantomData<T>,
    item: U,
}

impl<T, U: fmt::Display> fmt::Display for Part<T, U> {
    /// Writes serialized JSON
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.item)
    }
}

/// Wrapper trait for `Part<T, U>`
trait CciPart: Sized + fmt::Display + DeserializeOwned + 'static {
    /// Identifies the file format of the cross-crate information
    type FileFormat: sorted_template::FileFormat;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self>;
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct AllCrates;
type AllCratesPart = Part<AllCrates, OrderedJson>;
impl CciPart for AllCratesPart {
    type FileFormat = sorted_template::Js;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self> {
        &crate_info.all_crates
    }
}

impl AllCratesPart {
    fn blank() -> SortedTemplate<<Self as CciPart>::FileFormat> {
        SortedTemplate::from_before_after("window.ALL_CRATES = [", "];")
    }

    fn get(
        crate_name_json: OrderedJson,
        resource_suffix: &str,
    ) -> Result<PartsAndLocations<Self>, Error> {
        // external hack_get_external_crate_names not needed here, because
        // there's no way that we write the search index but not crates.js
        let path = suffix_path("crates.js", resource_suffix);
        Ok(PartsAndLocations::with(path, crate_name_json))
    }
}

/// Reads `crates.js`, which seems like the best
/// place to obtain the list of externally documented crates if the index
/// page was disabled when documenting the deps.
///
/// This is to match the current behavior of rustdoc, which allows you to get all crates
/// on the index page, even if --enable-index-page is only passed to the last crate.
fn hack_get_external_crate_names(
    doc_root: &Path,
    resource_suffix: &str,
) -> Result<Vec<String>, Error> {
    let path = doc_root.join(suffix_path("crates.js", resource_suffix));
    let Ok(content) = fs::read_to_string(&path) else {
        // they didn't emit invocation specific, so we just say there were no crates
        return Ok(Vec::default());
    };
    // this is only run once so it's fine not to cache it
    // !dot_matches_new_line: all crates on same line. greedy: match last bracket
    let regex = Regex::new(r"\[.*\]").unwrap();
    let Some(content) = regex.find(&content) else {
        return Err(Error::new("could not find crates list in crates.js", path));
    };
    let content: Vec<String> = try_err!(serde_json::from_str(content.as_str()), &path);
    Ok(content)
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct CratesIndex;
type CratesIndexPart = Part<CratesIndex, String>;
impl CciPart for CratesIndexPart {
    type FileFormat = sorted_template::Html;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self> {
        &crate_info.crates_index
    }
}

impl CratesIndexPart {
    fn blank(cx: &Context<'_>) -> SortedTemplate<<Self as CciPart>::FileFormat> {
        let page = layout::Page {
            title: "Index of crates",
            short_title: "Crates",
            css_class: "mod sys",
            root_path: "./",
            static_root_path: cx.shared.static_root_path.as_deref(),
            description: "List of crates",
            resource_suffix: &cx.shared.resource_suffix,
            rust_logo: true,
        };
        let layout = &cx.shared.layout;
        let style_files = &cx.shared.style_files;
        const DELIMITER: &str = "\u{FFFC}"; // users are being naughty if they have this
        let content =
            format!("<h1>List of all crates</h1><ul class=\"all-items\">{DELIMITER}</ul>");
        let template = layout::render(layout, &page, "", content, style_files);
        SortedTemplate::from_template(&template, DELIMITER)
            .expect("Object Replacement Character (U+FFFC) should not appear in the --index-page")
    }

    /// Might return parts that are duplicate with ones in preexisting index.html
    fn get(crate_name: &str, external_crates: &[String]) -> Result<PartsAndLocations<Self>, Error> {
        let mut ret = PartsAndLocations::default();
        let path = Path::new("index.html");
        for crate_name in external_crates.iter().map(|s| s.as_str()).chain(once(crate_name)) {
            let part = format!(
                "<li><a href=\"{trailing_slash}index.html\">{crate_name}</a></li>",
                trailing_slash = ensure_trailing_slash(crate_name),
            );
            ret.push(path.to_path_buf(), part);
        }
        Ok(ret)
    }
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct Sources;
type SourcesPart = Part<Sources, EscapedJson>;
impl CciPart for SourcesPart {
    type FileFormat = sorted_template::Js;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self> {
        &crate_info.src_files_js
    }
}

impl SourcesPart {
    fn blank() -> SortedTemplate<<Self as CciPart>::FileFormat> {
        // This needs to be `var`, not `const`.
        // This variable needs declared in the current global scope so that if
        // src-script.js loads first, it can pick it up.
        SortedTemplate::from_before_after(r"createSrcSidebar('[", r"]');")
    }

    fn get(cx: &Context<'_>, crate_name: &OrderedJson) -> Result<PartsAndLocations<Self>, Error> {
        let hierarchy = Rc::new(Hierarchy::default());
        cx.shared
            .local_sources
            .iter()
            .filter_map(|p| p.0.strip_prefix(&cx.shared.src_root).ok())
            .for_each(|source| hierarchy.add_path(source));
        let path = suffix_path("src-files.js", &cx.shared.resource_suffix);
        let hierarchy = hierarchy.to_json_string();
        let part = OrderedJson::array_unsorted([crate_name, &hierarchy]);
        let part = EscapedJson::from(part);
        Ok(PartsAndLocations::with(path, part))
    }
}

/// Source files directory tree
#[derive(Debug, Default)]
struct Hierarchy {
    parent: Weak<Self>,
    elem: OsString,
    children: RefCell<FxIndexMap<OsString, Rc<Self>>>,
    elems: RefCell<FxIndexSet<OsString>>,
}

impl Hierarchy {
    fn with_parent(elem: OsString, parent: &Rc<Self>) -> Self {
        Self { elem, parent: Rc::downgrade(parent), ..Self::default() }
    }

    fn to_json_string(&self) -> OrderedJson {
        let subs = self.children.borrow();
        let files = self.elems.borrow();
        let name = OrderedJson::serialize(self.elem.to_str().expect("invalid osstring conversion"))
            .unwrap();
        let mut out = Vec::from([name]);
        if !subs.is_empty() || !files.is_empty() {
            let subs = subs.iter().map(|(_, s)| s.to_json_string());
            out.push(OrderedJson::array_sorted(subs));
        }
        if !files.is_empty() {
            let files = files
                .iter()
                .map(|s| OrderedJson::serialize(s.to_str().expect("invalid osstring")).unwrap());
            out.push(OrderedJson::array_sorted(files));
        }
        OrderedJson::array_unsorted(out)
    }

    fn add_path(self: &Rc<Self>, path: &Path) {
        let mut h = Rc::clone(self);
        let mut elems = path
            .components()
            .filter_map(|s| match s {
                Component::Normal(s) => Some(s.to_owned()),
                Component::ParentDir => Some(OsString::from("..")),
                _ => None,
            })
            .peekable();
        loop {
            let cur_elem = elems.next().expect("empty file path");
            if cur_elem == ".." {
                if let Some(parent) = h.parent.upgrade() {
                    h = parent;
                }
                continue;
            }
            if elems.peek().is_none() {
                h.elems.borrow_mut().insert(cur_elem);
                break;
            } else {
                let entry = Rc::clone(
                    h.children
                        .borrow_mut()
                        .entry(cur_elem.clone())
                        .or_insert_with(|| Rc::new(Self::with_parent(cur_elem, &h))),
                );
                h = entry;
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct TypeAlias;
type TypeAliasPart = Part<TypeAlias, OrderedJson>;
impl CciPart for TypeAliasPart {
    type FileFormat = sorted_template::Js;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self> {
        &crate_info.type_impl
    }
}

impl TypeAliasPart {
    fn blank() -> SortedTemplate<<Self as CciPart>::FileFormat> {
        SortedTemplate::from_before_after(
            r"(function() {
    var type_impls = Object.fromEntries([",
            r"]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()",
        )
    }

    fn get(
        cx: &mut Context<'_>,
        krate: &Crate,
        crate_name_json: &OrderedJson,
    ) -> Result<PartsAndLocations<Self>, Error> {
        let mut path_parts = PartsAndLocations::default();

        let mut type_impl_collector = TypeImplCollector {
            aliased_types: IndexMap::default(),
            visited_aliases: FxHashSet::default(),
            cx,
        };
        DocVisitor::visit_crate(&mut type_impl_collector, krate);
        let cx = type_impl_collector.cx;
        let aliased_types = type_impl_collector.aliased_types;
        for aliased_type in aliased_types.values() {
            let impls = aliased_type.impl_.values().filter_map(
                |AliasedTypeImpl { impl_, type_aliases }| {
                    let mut ret: Option<AliasSerializableImpl> = None;
                    // render_impl will filter out "impossible-to-call" methods
                    // to make that functionality work here, it needs to be called with
                    // each type alias, and if it gives a different result, split the impl
                    for &(type_alias_fqp, type_alias_item) in type_aliases {
                        cx.id_map.borrow_mut().clear();
                        cx.deref_id_map.borrow_mut().clear();
                        let type_alias_fqp = join_path_syms(type_alias_fqp);
                        if let Some(ret) = &mut ret {
                            ret.aliases.push(type_alias_fqp);
                        } else {
                            let target_did = impl_
                                .inner_impl()
                                .trait_
                                .as_ref()
                                .map(|trait_| trait_.def_id())
                                .or_else(|| impl_.inner_impl().for_.def_id(&cx.shared.cache));
                            let provided_methods;
                            let assoc_link = if let Some(target_did) = target_did {
                                provided_methods =
                                    impl_.inner_impl().provided_trait_methods(cx.tcx());
                                AssocItemLink::GotoSource(
                                    ItemId::DefId(target_did),
                                    &provided_methods,
                                )
                            } else {
                                AssocItemLink::Anchor(None)
                            };
                            let text = super::render_impl(
                                cx,
                                impl_,
                                type_alias_item,
                                assoc_link,
                                RenderMode::Normal,
                                None,
                                &[],
                                ImplRenderingParameters {
                                    show_def_docs: true,
                                    show_default_items: true,
                                    show_non_assoc_items: true,
                                    toggle_open_by_default: true,
                                },
                            )
                            .to_string();
                            // The alternate display prints it as plaintext instead of HTML.
                            let trait_ = impl_
                                .inner_impl()
                                .trait_
                                .as_ref()
                                .map(|trait_| format!("{:#}", trait_.print(cx)));
                            ret = Some(AliasSerializableImpl {
                                text,
                                trait_,
                                aliases: vec![type_alias_fqp],
                            })
                        }
                    }
                    ret
                },
            );

            let mut path = PathBuf::from("type.impl");
            for component in &aliased_type.target_fqp[..aliased_type.target_fqp.len() - 1] {
                path.push(component.as_str());
            }
            let aliased_item_type = aliased_type.target_type;
            path.push(format!(
                "{aliased_item_type}.{}.js",
                aliased_type.target_fqp[aliased_type.target_fqp.len() - 1]
            ));

            let part = OrderedJson::array_sorted(
                impls.map(|impl_| OrderedJson::serialize(impl_).unwrap()),
            );
            path_parts.push(path, OrderedJson::array_unsorted([crate_name_json, &part]));
        }
        Ok(path_parts)
    }
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct TraitAlias;
type TraitAliasPart = Part<TraitAlias, OrderedJson>;
impl CciPart for TraitAliasPart {
    type FileFormat = sorted_template::Js;
    fn from_crate_info(crate_info: &CrateInfo) -> &PartsAndLocations<Self> {
        &crate_info.trait_impl
    }
}

impl TraitAliasPart {
    fn blank() -> SortedTemplate<<Self as CciPart>::FileFormat> {
        SortedTemplate::from_before_after(
            r"(function() {
    var implementors = Object.fromEntries([",
            r"]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()",
        )
    }

    fn get(
        cx: &Context<'_>,
        crate_name_json: &OrderedJson,
    ) -> Result<PartsAndLocations<Self>, Error> {
        let cache = &cx.shared.cache;
        let mut path_parts = PartsAndLocations::default();
        // Update the list of all implementors for traits
        // <https://github.com/search?q=repo%3Arust-lang%2Frust+[RUSTDOCIMPL]+trait.impl&type=code>
        for (&did, imps) in &cache.implementors {
            // Private modules can leak through to this phase of rustdoc, which
            // could contain implementations for otherwise private types. In some
            // rare cases we could find an implementation for an item which wasn't
            // indexed, so we just skip this step in that case.
            //
            // FIXME: this is a vague explanation for why this can't be a `get`, in
            //        theory it should be...
            let (remote_path, remote_item_type) = match cache.exact_paths.get(&did) {
                Some(p) => match cache.paths.get(&did).or_else(|| cache.external_paths.get(&did)) {
                    Some((_, t)) => (p, t),
                    None => continue,
                },
                None => match cache.external_paths.get(&did) {
                    Some((p, t)) => (p, t),
                    None => continue,
                },
            };

            let mut implementors = imps
                .iter()
                .filter_map(|imp| {
                    // If the trait and implementation are in the same crate, then
                    // there's no need to emit information about it (there's inlining
                    // going on). If they're in different crates then the crate defining
                    // the trait will be interested in our implementation.
                    //
                    // If the implementation is from another crate then that crate
                    // should add it.
                    if imp.impl_item.item_id.krate() == did.krate
                        || !imp.impl_item.item_id.is_local()
                    {
                        None
                    } else {
                        Some(Implementor {
                            text: imp.inner_impl().print(false, cx).to_string(),
                            synthetic: imp.inner_impl().kind.is_auto(),
                            types: collect_paths_for_type(&imp.inner_impl().for_, cache),
                        })
                    }
                })
                .peekable();

            // Only create a js file if we have impls to add to it. If the trait is
            // documented locally though we always create the file to avoid dead
            // links.
            if implementors.peek().is_none() && !cache.paths.contains_key(&did) {
                continue;
            }

            let mut path = PathBuf::from("trait.impl");
            for component in &remote_path[..remote_path.len() - 1] {
                path.push(component.as_str());
            }
            path.push(format!("{remote_item_type}.{}.js", remote_path[remote_path.len() - 1]));

            let part = OrderedJson::array_sorted(
                implementors.map(|implementor| OrderedJson::serialize(implementor).unwrap()),
            );
            path_parts.push(path, OrderedJson::array_unsorted([crate_name_json, &part]));
        }
        Ok(path_parts)
    }
}

struct Implementor {
    text: String,
    synthetic: bool,
    types: Vec<String>,
}

impl Serialize for Implementor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.text)?;
        if self.synthetic {
            seq.serialize_element(&1)?;
            seq.serialize_element(&self.types)?;
        }
        seq.end()
    }
}

/// Collect the list of aliased types and their aliases.
/// <https://github.com/search?q=repo%3Arust-lang%2Frust+[RUSTDOCIMPL]+type.impl&type=code>
///
/// The clean AST has type aliases that point at their types, but
/// this visitor works to reverse that: `aliased_types` is a map
/// from target to the aliases that reference it, and each one
/// will generate one file.
struct TypeImplCollector<'cx, 'cache, 'item> {
    /// Map from DefId-of-aliased-type to its data.
    aliased_types: IndexMap<DefId, AliasedType<'cache, 'item>>,
    visited_aliases: FxHashSet<DefId>,
    cx: &'cache Context<'cx>,
}

/// Data for an aliased type.
///
/// In the final file, the format will be roughly:
///
/// ```json
/// // type.impl/CRATE/TYPENAME.js
/// JSONP(
/// "CRATE": [
///   ["IMPL1 HTML", "ALIAS1", "ALIAS2", ...],
///   ["IMPL2 HTML", "ALIAS3", "ALIAS4", ...],
///    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ struct AliasedType
///   ...
/// ]
/// )
/// ```
struct AliasedType<'cache, 'item> {
    /// This is used to generate the actual filename of this aliased type.
    target_fqp: &'cache [Symbol],
    target_type: ItemType,
    /// This is the data stored inside the file.
    /// ItemId is used to deduplicate impls.
    impl_: IndexMap<ItemId, AliasedTypeImpl<'cache, 'item>>,
}

/// The `impl_` contains data that's used to figure out if an alias will work,
/// and to generate the HTML at the end.
///
/// The `type_aliases` list is built up with each type alias that matches.
struct AliasedTypeImpl<'cache, 'item> {
    impl_: &'cache Impl,
    type_aliases: Vec<(&'cache [Symbol], &'item Item)>,
}

impl<'item> DocVisitor<'item> for TypeImplCollector<'_, '_, 'item> {
    fn visit_item(&mut self, it: &'item Item) {
        self.visit_item_recur(it);
        let cache = &self.cx.shared.cache;
        let ItemKind::TypeAliasItem(ref t) = it.kind else { return };
        let Some(self_did) = it.item_id.as_def_id() else { return };
        if !self.visited_aliases.insert(self_did) {
            return;
        }
        let Some(target_did) = t.type_.def_id(cache) else { return };
        let get_extern = { || cache.external_paths.get(&target_did) };
        let Some(&(ref target_fqp, target_type)) = cache.paths.get(&target_did).or_else(get_extern)
        else {
            return;
        };
        let aliased_type = self.aliased_types.entry(target_did).or_insert_with(|| {
            let impl_ = cache
                .impls
                .get(&target_did)
                .into_iter()
                .flatten()
                .map(|impl_| {
                    (impl_.impl_item.item_id, AliasedTypeImpl { impl_, type_aliases: Vec::new() })
                })
                .collect();
            AliasedType { target_fqp: &target_fqp[..], target_type, impl_ }
        });
        let get_local = { || cache.paths.get(&self_did).map(|(p, _)| p) };
        let Some(self_fqp) = cache.exact_paths.get(&self_did).or_else(get_local) else {
            return;
        };
        let aliased_ty = self.cx.tcx().type_of(self_did).skip_binder();
        // Exclude impls that are directly on this type. They're already in the HTML.
        // Some inlining scenarios can cause there to be two versions of the same
        // impl: one on the type alias and one on the underlying target type.
        let mut seen_impls: FxHashSet<ItemId> =
            cache.impls.get(&self_did).into_iter().flatten().map(|i| i.impl_item.item_id).collect();
        for (impl_item_id, aliased_type_impl) in &mut aliased_type.impl_ {
            // Only include this impl if it actually unifies with this alias.
            // Synthetic impls are not included; those are also included in the HTML.
            //
            // FIXME(lazy_type_alias): Once the feature is complete or stable, rewrite this
            // to use type unification.
            // Be aware of `tests/rustdoc/type-alias/deeply-nested-112515.rs` which might regress.
            let Some(impl_did) = impl_item_id.as_def_id() else { continue };
            let for_ty = self.cx.tcx().type_of(impl_did).skip_binder();
            let reject_cx = DeepRejectCtxt::relate_infer_infer(self.cx.tcx());
            if !reject_cx.types_may_unify(aliased_ty, for_ty) {
                continue;
            }
            // Avoid duplicates
            if !seen_impls.insert(*impl_item_id) {
                continue;
            }
            // This impl was not found in the set of rejected impls
            aliased_type_impl.type_aliases.push((&self_fqp[..], it));
        }
    }
}

/// Final serialized form of the alias impl
struct AliasSerializableImpl {
    text: String,
    trait_: Option<String>,
    aliases: Vec<String>,
}

impl Serialize for AliasSerializableImpl {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.text)?;
        if let Some(trait_) = &self.trait_ {
            seq.serialize_element(trait_)?;
        } else {
            seq.serialize_element(&0)?;
        }
        for type_ in &self.aliases {
            seq.serialize_element(type_)?;
        }
        seq.end()
    }
}

fn get_path_parts<T: CciPart>(
    dst: &Path,
    crates_info: &[CrateInfo],
) -> FxIndexMap<PathBuf, Vec<String>> {
    let mut templates: FxIndexMap<PathBuf, Vec<String>> = FxIndexMap::default();
    crates_info.iter().flat_map(|crate_info| T::from_crate_info(crate_info).parts.iter()).for_each(
        |(path, part)| {
            let path = dst.join(path);
            let part = part.to_string();
            templates.entry(path).or_default().push(part);
        },
    );
    templates
}

/// Create all parents
fn create_parents(path: &Path) -> Result<(), Error> {
    let parent = path.parent().expect("should not have an empty path here");
    try_err!(fs::create_dir_all(parent), parent);
    Ok(())
}

/// Returns a blank template unless we could find one to append to
fn read_template_or_blank<F, T: FileFormat>(
    mut make_blank: F,
    path: &Path,
    should_merge: &ShouldMerge,
) -> Result<SortedTemplate<T>, Error>
where
    F: FnMut() -> SortedTemplate<T>,
{
    if !should_merge.read_rendered_cci {
        return Ok(make_blank());
    }
    match fs::read_to_string(path) {
        Ok(template) => Ok(try_err!(SortedTemplate::from_str(&template), &path)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(make_blank()),
        Err(e) => Err(Error::new(e, path)),
    }
}

/// info from this crate and the --include-info-json'd crates
fn write_rendered_cci<T: CciPart, F>(
    mut make_blank: F,
    dst: &Path,
    crates_info: &[CrateInfo],
    should_merge: &ShouldMerge,
) -> Result<(), Error>
where
    F: FnMut() -> SortedTemplate<T::FileFormat>,
{
    // write the merged cci to disk
    for (path, parts) in get_path_parts::<T>(dst, crates_info) {
        create_parents(&path)?;
        // read previous rendered cci from storage, append to them
        let mut template =
            read_template_or_blank::<_, T::FileFormat>(&mut make_blank, &path, should_merge)?;
        for part in parts {
            template.append(part);
        }
        let mut file = try_err!(File::create_buffered(&path), &path);
        try_err!(write!(file, "{template}"), &path);
        try_err!(file.flush(), &path);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
