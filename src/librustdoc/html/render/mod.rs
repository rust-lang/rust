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

mod context;
mod print_item;
mod span_map;
mod templates;
mod write_shared;

crate use context::*;
crate use span_map::{collect_spans_and_sources, LinkFromSrc};

use std::collections::VecDeque;
use std::default::Default;
use std::fmt;
use std::path::PathBuf;
use std::str;
use std::string::ToString;

use rustc_ast_pretty::pprust;
use rustc_attr::{ConstStability, Deprecation, StabilityLevel};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_middle::middle::stability;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::{kw, sym, Symbol};
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use crate::clean::{self, GetDefId, ItemId, RenderedLink, SelfTy};
use crate::docfs::PathError;
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::formats::{AssocItemRender, Impl, RenderMode};
use crate::html::escape::Escape;
use crate::html::format::{
    href, print_abi_with_space, print_constness_with_space, print_default_space,
    print_generic_bounds, print_where_clause, Buffer, HrefError, PrintWithSpace,
};
use crate::html::markdown::{HeadingOffset, Markdown, MarkdownHtml, MarkdownSummaryLine};

/// A pair of name and its optional document.
crate type NameDoc = (String, Option<String>);

crate fn ensure_trailing_slash(v: &str) -> impl fmt::Display + '_ {
    crate::html::format::display_fn(move |f| {
        if !v.ends_with('/') && !v.is_empty() { write!(f, "{}/", v) } else { f.write_str(v) }
    })
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
    crate aliases: Box<[String]>,
}

/// A type used for the search index.
#[derive(Debug)]
crate struct RenderType {
    name: Option<String>,
    generics: Option<Vec<String>>,
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
    kind: ItemType,
}

impl From<(RenderType, ItemType)> for TypeWithKind {
    fn from(x: (RenderType, ItemType)) -> TypeWithKind {
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
        seq.serialize_element(&self.kind)?;
        if let Some(generics) = &self.ty.generics {
            seq.serialize_element(generics)?;
        }
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

fn write_srclink(cx: &Context<'_>, item: &clean::Item, buf: &mut Buffer) {
    if let Some(l) = cx.src_href(item) {
        write!(buf, "<a class=\"srclink\" href=\"{}\" title=\"goto source code\">[src]</a>", l)
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

impl AllTypes {
    fn print(self, f: &mut Buffer) {
        fn print_entries(f: &mut Buffer, e: &FxHashSet<ItemEntry>, title: &str, class: &str) {
            if !e.is_empty() {
                let mut e: Vec<&ItemEntry> = e.iter().collect();
                e.sort();
                write!(
                    f,
                    "<h3 id=\"{}\">{}</h3><ul class=\"{} docblock\">",
                    title.replace(' ', "-"), // IDs cannot contain whitespaces.
                    title,
                    class
                );

                for s in e.iter() {
                    write!(f, "<li>{}</li>", s.print());
                }

                f.write_str("</ul>");
            }
        }

        f.write_str(
            "<h1 class=\"fqn\">\
                 <span class=\"in-band\">List of all items</span>\
                 <span class=\"out-of-band\">\
                     <span id=\"render-detail\">\
                         <a id=\"toggle-all-docs\" href=\"javascript:void(0)\" \
                            title=\"collapse all docs\">\
                             [<span class=\"inner\">&#x2212;</span>]\
                         </a>\
                     </span>
                 </span>
             </h1>",
        );
        // Note: print_entries does not escape the title, because we know the current set of titles
        // doesn't require escaping.
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
                        if opt.0 == default_value { "selected" } else { "" },
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
        ("auto-hide-large-items", "Auto-hide item contents for large items.", true).into(),
        ("auto-hide-method-docs", "Auto-hide item methods' documentation", false).into(),
        ("auto-hide-trait-implementations", "Auto-hide trait implementation documentation", false)
            .into(),
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

fn document(
    w: &mut Buffer,
    cx: &Context<'_>,
    item: &clean::Item,
    parent: Option<&clean::Item>,
    heading_offset: HeadingOffset,
) {
    if let Some(ref name) = item.name {
        info!("Documenting {}", name);
    }
    document_item_info(w, cx, item, parent);
    if parent.is_none() {
        document_full_collapsible(w, item, cx, heading_offset);
    } else {
        document_full(w, item, cx, heading_offset);
    }
}

/// Render md_text as markdown.
fn render_markdown(
    w: &mut Buffer,
    cx: &Context<'_>,
    md_text: &str,
    links: Vec<RenderedLink>,
    heading_offset: HeadingOffset,
) {
    let mut ids = cx.id_map.borrow_mut();
    write!(
        w,
        "<div class=\"docblock\">{}</div>",
        Markdown {
            content: md_text,
            links: &links,
            ids: &mut ids,
            error_codes: cx.shared.codes,
            edition: cx.shared.edition(),
            playground: &cx.shared.playground,
            heading_offset,
        }
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
    parent: &clean::Item,
    show_def_docs: bool,
) {
    document_item_info(w, cx, item, Some(parent));
    if !show_def_docs {
        return;
    }
    if let Some(s) = item.doc_value() {
        let mut summary_html = MarkdownSummaryLine(&s, &item.links(cx)).into_string();

        if s.contains('\n') {
            let link = format!(r#" <a href="{}">Read more</a>"#, naive_assoc_href(item, link, cx));

            if let Some(idx) = summary_html.rfind("</p>") {
                summary_html.insert_str(idx, &link);
            } else {
                summary_html.push_str(&link);
            }
        }

        write!(w, "<div class='docblock'>{}</div>", summary_html,);
    }
}

fn document_full_collapsible(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &Context<'_>,
    heading_offset: HeadingOffset,
) {
    document_full_inner(w, item, cx, true, heading_offset);
}

fn document_full(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &Context<'_>,
    heading_offset: HeadingOffset,
) {
    document_full_inner(w, item, cx, false, heading_offset);
}

fn document_full_inner(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &Context<'_>,
    is_collapsible: bool,
    heading_offset: HeadingOffset,
) {
    if let Some(s) = cx.shared.maybe_collapsed_doc_value(item) {
        debug!("Doc block: =====\n{}\n=====", s);
        if is_collapsible {
            w.write_str(
                "<details class=\"rustdoc-toggle top-doc\" open>\
                <summary class=\"hideme\">\
                     <span>Expand description</span>\
                </summary>",
            );
            render_markdown(w, cx, &s, item.links(cx), heading_offset);
            w.write_str("</details>");
        } else {
            render_markdown(w, cx, &s, item.links(cx), heading_offset);
        }
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
    parent: Option<&clean::Item>,
) {
    let item_infos = short_item_info(item, cx, parent);
    if !item_infos.is_empty() {
        w.write_str("<div class=\"item-info\">");
        for info in item_infos {
            w.write_str(&info);
        }
        w.write_str("</div>");
    }
}

fn portability(item: &clean::Item, parent: Option<&clean::Item>) -> Option<String> {
    let cfg = match (&item.cfg, parent.and_then(|p| p.cfg.as_ref())) {
        (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
        (cfg, _) => cfg.as_deref().cloned(),
    };

    debug!("Portability {:?} - {:?} = {:?}", item.cfg, parent.and_then(|p| p.cfg.as_ref()), cfg);

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

    if let Some(depr @ Deprecation { note, since, is_since_rustc_version: _, suggestion: _ }) =
        item.deprecation(cx.tcx())
    {
        // We display deprecation messages for #[deprecated] and #[rustc_deprecated]
        // but only display the future-deprecation messages for #[rustc_deprecated].
        let mut message = if let Some(since) = since {
            let since = &since.as_str();
            if !stability::deprecation_in_effect(&depr) {
                if *since == "TBD" {
                    String::from("Deprecating in a future Rust version")
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
                cx.shared.edition(),
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
                    cx.shared.edition(),
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

// Render the list of items inside one of the sections "Trait Implementations",
// "Auto Trait Implementations," "Blanket Trait Implementations" (on struct/enum pages).
fn render_impls(cx: &Context<'_>, w: &mut Buffer, impls: &[&&Impl], containing_item: &clean::Item) {
    let tcx = cx.tcx();
    let mut rendered_impls = impls
        .iter()
        .map(|i| {
            let did = i.trait_did().unwrap();
            let provided_trait_methods = i.inner_impl().provided_trait_methods(tcx);
            let assoc_link = AssocItemLink::GotoSource(did.into(), &provided_trait_methods);
            let mut buffer = if w.is_for_html() { Buffer::html() } else { Buffer::new() };
            render_impl(
                &mut buffer,
                cx,
                i,
                containing_item,
                assoc_link,
                RenderMode::Normal,
                None,
                &[],
                ImplRenderingParameters {
                    show_def_docs: true,
                    is_on_foreign_type: false,
                    show_default_items: true,
                    show_non_assoc_items: true,
                    toggle_open_by_default: true,
                },
            );
            buffer.into_inner()
        })
        .collect::<Vec<_>>();
    rendered_impls.sort();
    w.write_str(&rendered_impls.join(""));
}

fn naive_assoc_href(it: &clean::Item, link: AssocItemLink<'_>, cx: &Context<'_>) -> String {
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
            href(did.expect_def_id(), cx).map(|p| format!("{}{}", p.0, anchor)).unwrap_or(anchor)
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
        "{}{}const <a href=\"{}\" class=\"constant\">{}</a>: {}",
        extra,
        it.visibility.print_with_space(it.def_id, cx),
        naive_assoc_href(it, link, cx),
        it.name.as_ref().unwrap(),
        ty.print(cx)
    );
}

fn assoc_type(
    w: &mut Buffer,
    it: &clean::Item,
    bounds: &[clean::GenericBound],
    default: Option<&clean::Type>,
    link: AssocItemLink<'_>,
    extra: &str,
    cx: &Context<'_>,
) {
    write!(
        w,
        "{}type <a href=\"{}\" class=\"type\">{}</a>",
        extra,
        naive_assoc_href(it, link, cx),
        it.name.as_ref().unwrap()
    );
    if !bounds.is_empty() {
        write!(w, ": {}", print_generic_bounds(bounds, cx))
    }
    if let Some(default) = default {
        write!(w, " = {}", default.print(cx))
    }
}

fn render_stability_since_raw(
    w: &mut Buffer,
    ver: Option<&str>,
    const_stability: Option<&ConstStability>,
    containing_ver: Option<&str>,
    containing_const_ver: Option<&str>,
) {
    let ver = ver.filter(|inner| !inner.is_empty());

    match (ver, const_stability) {
        // stable and const stable
        (Some(v), Some(ConstStability { level: StabilityLevel::Stable { since }, .. }))
            if Some(since.as_str()).as_deref() != containing_const_ver =>
        {
            write!(
                w,
                "<span class=\"since\" title=\"Stable since Rust version {0}, const since {1}\">{0} (const: {1})</span>",
                v, since
            );
        }
        // stable and const unstable
        (
            Some(v),
            Some(ConstStability { level: StabilityLevel::Unstable { issue, .. }, feature, .. }),
        ) => {
            write!(
                w,
                "<span class=\"since\" title=\"Stable since Rust version {0}, const unstable\">{0} (const: ",
                v
            );
            if let Some(n) = issue {
                write!(
                    w,
                    "<a href=\"https://github.com/rust-lang/rust/issues/{}\" title=\"Tracking issue for {}\">unstable</a>",
                    n, feature
                );
            } else {
                write!(w, "unstable");
            }
            write!(w, ")</span>");
        }
        // stable
        (Some(v), _) if ver != containing_ver => {
            write!(
                w,
                "<span class=\"since\" title=\"Stable since Rust version {0}\">{0}</span>",
                v
            );
        }
        _ => {}
    }
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
        let href = match link {
            AssocItemLink::Anchor(Some(ref id)) => Some(format!("#{}", id)),
            AssocItemLink::Anchor(None) => Some(format!("#{}.{}", meth.type_(), name)),
            AssocItemLink::GotoSource(did, provided_methods) => {
                // We're creating a link from an impl-item to the corresponding
                // trait-item and need to map the anchored type accordingly.
                let ty = if provided_methods.contains(&name) {
                    ItemType::Method
                } else {
                    ItemType::TyMethod
                };

                match (href(did.expect_def_id(), cx), ty) {
                    (Ok(p), ty) => Some(format!("{}#{}.{}", p.0, ty, name)),
                    (Err(HrefError::DocumentationNotBuilt), ItemType::TyMethod) => None,
                    (Err(_), ty) => Some(format!("#{}.{}", ty, name)),
                }
            }
        };
        let vis = meth.visibility.print_with_space(meth.def_id, cx).to_string();
        let constness =
            print_constness_with_space(&header.constness, meth.const_stability(cx.tcx()));
        let asyncness = header.asyncness.print_with_space();
        let unsafety = header.unsafety.print_with_space();
        let defaultness = print_default_space(meth.is_default());
        let abi = print_abi_with_space(header.abi).to_string();

        // NOTE: `{:#}` does not print HTML formatting, `{}` does. So `g.print` can't be reused between the length calculation and `write!`.
        let generics_len = format!("{:#}", g.print(cx)).len();
        let mut header_len = "fn ".len()
            + vis.len()
            + constness.len()
            + asyncness.len()
            + unsafety.len()
            + defaultness.len()
            + abi.len()
            + name.as_str().len()
            + generics_len;

        let (indent, indent_str, end_newline) = if parent == ItemType::Trait {
            header_len += 4;
            let indent_str = "    ";
            render_attributes_in_pre(w, meth, indent_str);
            (4, indent_str, false)
        } else {
            render_attributes_in_code(w, meth);
            (0, "", true)
        };
        w.reserve(header_len + "<a href=\"\" class=\"fnname\">{".len() + "</a>".len());
        write!(
            w,
            "{indent}{vis}{constness}{asyncness}{unsafety}{defaultness}{abi}fn <a {href} class=\"fnname\">{name}</a>\
             {generics}{decl}{notable_traits}{where_clause}",
            indent = indent_str,
            vis = vis,
            constness = constness,
            asyncness = asyncness,
            unsafety = unsafety,
            defaultness = defaultness,
            abi = abi,
            // links without a href are valid - https://www.w3schools.com/tags/att_a_href.asp
            href = href.map(|href| format!("href=\"{}\"", href)).unwrap_or_else(|| "".to_string()),
            name = name,
            generics = g.print(cx),
            decl = d.full_print(header_len, indent, header.asyncness, cx),
            notable_traits = notable_traits_decl(&d, cx),
            where_clause = print_where_clause(g, cx, indent, end_newline),
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
            cx,
        ),
        _ => panic!("render_assoc_item called on non-associated-item"),
    }
}

const ALLOWED_ATTRIBUTES: &[Symbol] =
    &[sym::export_name, sym::link_section, sym::no_mangle, sym::repr, sym::non_exhaustive];

fn attributes(it: &clean::Item) -> Vec<String> {
    it.attrs
        .other_attrs
        .iter()
        .filter_map(|attr| {
            if ALLOWED_ATTRIBUTES.contains(&attr.name_or_empty()) {
                Some(pprust::attribute_to_string(&attr).replace("\n", "").replace("  ", " "))
            } else {
                None
            }
        })
        .collect()
}

// When an attribute is rendered inside a `<pre>` tag, it is formatted using
// a whitespace prefix and newline.
fn render_attributes_in_pre(w: &mut Buffer, it: &clean::Item, prefix: &str) {
    for a in attributes(it) {
        writeln!(w, "{}{}", prefix, a);
    }
}

// When an attribute is rendered inside a <code> tag, it is formatted using
// a div to produce a newline after it.
fn render_attributes_in_code(w: &mut Buffer, it: &clean::Item) {
    for a in attributes(it) {
        write!(w, "<div class=\"code-attribute\">{}</div>", a);
    }
}

#[derive(Copy, Clone)]
enum AssocItemLink<'a> {
    Anchor(Option<&'a str>),
    GotoSource(ItemId, &'a FxHashSet<Symbol>),
}

impl<'a> AssocItemLink<'a> {
    fn anchor(&self, id: &'a str) -> Self {
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
) {
    info!("Documenting associated items of {:?}", containing_item.name);
    let cache = cx.cache();
    let v = match cache.impls.get(&it) {
        Some(v) => v,
        None => return,
    };
    let (non_trait, traits): (Vec<_>, _) = v.iter().partition(|i| i.inner_impl().trait_.is_none());
    if !non_trait.is_empty() {
        let render_mode = match what {
            AssocItemRender::All => {
                w.write_str(
                    "<h2 id=\"implementations\" class=\"small-section-header\">\
                         Implementations<a href=\"#implementations\" class=\"anchor\"></a>\
                    </h2>",
                );
                RenderMode::Normal
            }
            AssocItemRender::DerefFor { trait_, type_, deref_mut_ } => {
                write!(
                    w,
                    "<h2 id=\"deref-methods\" class=\"small-section-header\">\
                         <span>Methods from {trait_}&lt;Target = {type_}&gt;</span>\
                         <a href=\"#deref-methods\" class=\"anchor\"></a>\
                     </h2>",
                    trait_ = trait_.print(cx),
                    type_ = type_.print(cx),
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
                None,
                &[],
                ImplRenderingParameters {
                    show_def_docs: true,
                    is_on_foreign_type: false,
                    show_default_items: true,
                    show_non_assoc_items: true,
                    toggle_open_by_default: true,
                },
            );
        }
    }
    if let AssocItemRender::DerefFor { .. } = what {
        return;
    }
    if !traits.is_empty() {
        let deref_impl =
            traits.iter().find(|t| t.trait_did() == cx.tcx().lang_items().deref_trait());
        if let Some(impl_) = deref_impl {
            let has_deref_mut =
                traits.iter().any(|t| t.trait_did() == cx.tcx().lang_items().deref_mut_trait());
            render_deref_methods(w, cx, impl_, containing_item, has_deref_mut);
        }
        let (synthetic, concrete): (Vec<&&Impl>, Vec<&&Impl>) =
            traits.iter().partition(|t| t.inner_impl().synthetic);
        let (blanket_impl, concrete): (Vec<&&Impl>, _) =
            concrete.into_iter().partition(|t| t.inner_impl().blanket_impl.is_some());

        let mut impls = Buffer::empty_from(&w);
        render_impls(cx, &mut impls, &concrete, containing_item);
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
            w.write_str(
                "<h2 id=\"synthetic-implementations\" class=\"small-section-header\">\
                     Auto Trait Implementations\
                     <a href=\"#synthetic-implementations\" class=\"anchor\"></a>\
                 </h2>\
                 <div id=\"synthetic-implementations-list\">",
            );
            render_impls(cx, w, &synthetic, containing_item);
            w.write_str("</div>");
        }

        if !blanket_impl.is_empty() {
            w.write_str(
                "<h2 id=\"blanket-implementations\" class=\"small-section-header\">\
                     Blanket Implementations\
                     <a href=\"#blanket-implementations\" class=\"anchor\"></a>\
                 </h2>\
                 <div id=\"blanket-implementations-list\">",
            );
            render_impls(cx, w, &blanket_impl, containing_item);
            w.write_str("</div>");
        }
    }
}

fn render_deref_methods(
    w: &mut Buffer,
    cx: &Context<'_>,
    impl_: &Impl,
    container_item: &clean::Item,
    deref_mut: bool,
) {
    let cache = cx.cache();
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
    debug!("Render deref methods for {:#?}, target {:#?}", impl_.inner_impl().for_, target);
    let what =
        AssocItemRender::DerefFor { trait_: deref_type, type_: real_target, deref_mut_: deref_mut };
    if let Some(did) = target.def_id_full(cache) {
        if let Some(type_did) = impl_.inner_impl().for_.def_id_full(cache) {
            // `impl Deref<Target = S> for S`
            if did == type_did {
                // Avoid infinite cycles
                return;
            }
        }
        render_assoc_items(w, cx, container_item, did, what);
    } else {
        if let Some(prim) = target.primitive_type() {
            if let Some(&did) = cache.primitive_locations.get(&prim) {
                render_assoc_items(w, cx, container_item, did, what);
            }
        }
    }
}

fn should_render_item(item: &clean::Item, deref_mut_: bool, tcx: TyCtxt<'_>) -> bool {
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
                (false, Some(did) == tcx.lang_items().owned_box(), false)
            }
            SelfTy::SelfValue => (false, false, true),
            _ => (false, false, false),
        };

        (deref_mut_ || !by_mut_ref) && !by_box && !by_value
    } else {
        false
    }
}

fn notable_traits_decl(decl: &clean::FnDecl, cx: &Context<'_>) -> String {
    let mut out = Buffer::html();

    if let Some(did) = decl.output.def_id_full(cx.cache()) {
        if let Some(impls) = cx.cache().impls.get(&did) {
            for i in impls {
                let impl_ = i.inner_impl();
                if let Some(trait_) = &impl_.trait_ {
                    let trait_did = trait_.def_id();

                    if cx.cache().traits.get(&trait_did).map_or(false, |t| t.is_notable) {
                        if out.is_empty() {
                            write!(
                                &mut out,
                                "<div class=\"notable\">Notable traits for {}</div>\
                             <code class=\"content\">",
                                impl_.for_.print(cx)
                            );
                        }

                        //use the "where" class here to make it small
                        write!(
                            &mut out,
                            "<span class=\"where fmt-newline\">{}</span>",
                            impl_.print(false, cx)
                        );
                        for it in &impl_.items {
                            if let clean::TypedefItem(ref tydef, _) = *it.kind {
                                out.push_str("<span class=\"where fmt-newline\">    ");
                                let empty_set = FxHashSet::default();
                                let src_link =
                                    AssocItemLink::GotoSource(trait_did.into(), &empty_set);
                                assoc_type(&mut out, it, &[], Some(&tydef.type_), src_link, "", cx);
                                out.push_str(";</span>");
                            }
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

#[derive(Clone, Copy, Debug)]
struct ImplRenderingParameters {
    show_def_docs: bool,
    is_on_foreign_type: bool,
    show_default_items: bool,
    /// Whether or not to show methods.
    show_non_assoc_items: bool,
    toggle_open_by_default: bool,
}

fn render_impl(
    w: &mut Buffer,
    cx: &Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    link: AssocItemLink<'_>,
    render_mode: RenderMode,
    use_absolute: Option<bool>,
    aliases: &[String],
    rendering_params: ImplRenderingParameters,
) {
    let cache = cx.cache();
    let traits = &cache.traits;
    let trait_ = i.trait_did().map(|did| &traits[&did]);
    let mut close_tags = String::new();

    // For trait implementations, the `interesting` output contains all methods that have doc
    // comments, and the `boring` output contains all methods that do not. The distinction is
    // used to allow hiding the boring methods.
    // `containing_item` is used for rendering stability info. If the parent is a trait impl,
    // `containing_item` will the grandparent, since trait impls can't have stability attached.
    fn doc_impl_item(
        boring: &mut Buffer,
        interesting: &mut Buffer,
        cx: &Context<'_>,
        item: &clean::Item,
        parent: &clean::Item,
        containing_item: &clean::Item,
        link: AssocItemLink<'_>,
        render_mode: RenderMode,
        is_default_item: bool,
        trait_: Option<&clean::Trait>,
        rendering_params: ImplRenderingParameters,
    ) {
        let item_type = item.type_();
        let name = item.name.as_ref().unwrap();

        let render_method_item = rendering_params.show_non_assoc_items
            && match render_mode {
                RenderMode::Normal => true,
                RenderMode::ForDeref { mut_: deref_mut_ } => {
                    should_render_item(&item, deref_mut_, cx.tcx())
                }
            };

        let in_trait_class = if trait_.is_some() { " trait-impl" } else { "" };

        let mut doc_buffer = Buffer::empty_from(boring);
        let mut info_buffer = Buffer::empty_from(boring);
        let mut short_documented = true;

        if render_method_item {
            if !is_default_item {
                if let Some(t) = trait_ {
                    // The trait item may have been stripped so we might not
                    // find any documentation or stability for it.
                    if let Some(it) = t.items.iter().find(|i| i.name == item.name) {
                        // We need the stability of the item from the trait
                        // because impls can't have a stability.
                        if item.doc_value().is_some() {
                            document_item_info(&mut info_buffer, cx, it, Some(parent));
                            document_full(&mut doc_buffer, item, cx, HeadingOffset::H5);
                            short_documented = false;
                        } else {
                            // In case the item isn't documented,
                            // provide short documentation from the trait.
                            document_short(
                                &mut doc_buffer,
                                it,
                                cx,
                                link,
                                parent,
                                rendering_params.show_def_docs,
                            );
                        }
                    }
                } else {
                    document_item_info(&mut info_buffer, cx, item, Some(parent));
                    if rendering_params.show_def_docs {
                        document_full(&mut doc_buffer, item, cx, HeadingOffset::H5);
                        short_documented = false;
                    }
                }
            } else {
                document_short(
                    &mut doc_buffer,
                    item,
                    cx,
                    link,
                    parent,
                    rendering_params.show_def_docs,
                );
            }
        }
        let w = if short_documented && trait_.is_some() { interesting } else { boring };

        let toggled = !doc_buffer.is_empty();
        if toggled {
            let method_toggle_class =
                if item_type == ItemType::Method { " method-toggle" } else { "" };
            write!(w, "<details class=\"rustdoc-toggle{}\" open><summary>", method_toggle_class);
        }
        match *item.kind {
            clean::MethodItem(..) | clean::TyMethodItem(_) => {
                // Only render when the method is not static or we allow static methods
                if render_method_item {
                    let id = cx.derive_id(format!("{}.{}", item_type, name));
                    let source_id = trait_
                        .and_then(|trait_| {
                            trait_.items.iter().find(|item| {
                                item.name.map(|n| n.as_str().eq(&name.as_str())).unwrap_or(false)
                            })
                        })
                        .map(|item| format!("{}.{}", item.type_(), name));
                    write!(
                        w,
                        "<div id=\"{}\" class=\"{}{} has-srclink\">",
                        id, item_type, in_trait_class,
                    );
                    render_rightside(w, cx, item, containing_item);
                    write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
                    w.write_str("<h4 class=\"code-header\">");
                    render_assoc_item(
                        w,
                        item,
                        link.anchor(source_id.as_ref().unwrap_or(&id)),
                        ItemType::Impl,
                        cx,
                    );
                    w.write_str("</h4>");
                    w.write_str("</div>");
                }
            }
            clean::TypedefItem(ref tydef, _) => {
                let source_id = format!("{}.{}", ItemType::AssocType, name);
                let id = cx.derive_id(source_id.clone());
                write!(
                    w,
                    "<div id=\"{}\" class=\"{}{} has-srclink\">",
                    id, item_type, in_trait_class
                );
                write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
                w.write_str("<h4 class=\"code-header\">");
                assoc_type(
                    w,
                    item,
                    &Vec::new(),
                    Some(&tydef.type_),
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    "",
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</div>");
            }
            clean::AssocConstItem(ref ty, ref default) => {
                let source_id = format!("{}.{}", item_type, name);
                let id = cx.derive_id(source_id.clone());
                write!(
                    w,
                    "<div id=\"{}\" class=\"{}{} has-srclink\">",
                    id, item_type, in_trait_class
                );
                render_rightside(w, cx, item, containing_item);
                write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
                w.write_str("<h4 class=\"code-header\">");
                assoc_const(
                    w,
                    item,
                    ty,
                    default.as_ref(),
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    "",
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</div>");
            }
            clean::AssocTypeItem(ref bounds, ref default) => {
                let source_id = format!("{}.{}", item_type, name);
                let id = cx.derive_id(source_id.clone());
                write!(w, "<div id=\"{}\" class=\"{}{}\">", id, item_type, in_trait_class,);
                write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
                w.write_str("<h4 class=\"code-header\">");
                assoc_type(
                    w,
                    item,
                    bounds,
                    default.as_ref(),
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    "",
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</div>");
            }
            clean::StrippedItem(..) => return,
            _ => panic!("can't make docs for trait item with name {:?}", item.name),
        }

        w.push_buffer(info_buffer);
        if toggled {
            w.write_str("</summary>");
            w.push_buffer(doc_buffer);
            w.push_str("</details>");
        }
    }

    let mut impl_items = Buffer::empty_from(w);
    let mut default_impl_items = Buffer::empty_from(w);

    for trait_item in &i.inner_impl().items {
        doc_impl_item(
            &mut default_impl_items,
            &mut impl_items,
            cx,
            trait_item,
            if trait_.is_some() { &i.impl_item } else { parent },
            parent,
            link,
            render_mode,
            false,
            trait_.map(|t| &t.trait_),
            rendering_params,
        );
    }

    fn render_default_items(
        boring: &mut Buffer,
        interesting: &mut Buffer,
        cx: &Context<'_>,
        t: &clean::Trait,
        i: &clean::Impl,
        parent: &clean::Item,
        containing_item: &clean::Item,
        render_mode: RenderMode,
        rendering_params: ImplRenderingParameters,
    ) {
        for trait_item in &t.items {
            let n = trait_item.name;
            if i.items.iter().any(|m| m.name == n) {
                continue;
            }
            let did = i.trait_.as_ref().unwrap().def_id();
            let provided_methods = i.provided_trait_methods(cx.tcx());
            let assoc_link = AssocItemLink::GotoSource(did.into(), &provided_methods);

            doc_impl_item(
                boring,
                interesting,
                cx,
                trait_item,
                parent,
                containing_item,
                assoc_link,
                render_mode,
                true,
                Some(t),
                rendering_params,
            );
        }
    }

    // If we've implemented a trait, then also emit documentation for all
    // default items which weren't overridden in the implementation block.
    // We don't emit documentation for default items if they appear in the
    // Implementations on Foreign Types or Implementors sections.
    if rendering_params.show_default_items {
        if let Some(t) = trait_ {
            render_default_items(
                &mut default_impl_items,
                &mut impl_items,
                cx,
                &t.trait_,
                &i.inner_impl(),
                &i.impl_item,
                parent,
                render_mode,
                rendering_params,
            );
        }
    }
    if render_mode == RenderMode::Normal {
        let toggled = !(impl_items.is_empty() && default_impl_items.is_empty());
        if toggled {
            close_tags.insert_str(0, "</details>");
            write!(
                w,
                "<details class=\"rustdoc-toggle implementors-toggle\"{}>",
                if rendering_params.toggle_open_by_default { " open" } else { "" }
            );
            write!(w, "<summary>")
        }
        render_impl_summary(
            w,
            cx,
            i,
            parent,
            parent,
            rendering_params.show_def_docs,
            use_absolute,
            rendering_params.is_on_foreign_type,
            aliases,
        );
        if toggled {
            write!(w, "</summary>")
        }

        if let Some(ref dox) = cx.shared.maybe_collapsed_doc_value(&i.impl_item) {
            let mut ids = cx.id_map.borrow_mut();
            write!(
                w,
                "<div class=\"docblock\">{}</div>",
                Markdown {
                    content: &*dox,
                    links: &i.impl_item.links(cx),
                    ids: &mut ids,
                    error_codes: cx.shared.codes,
                    edition: cx.shared.edition(),
                    playground: &cx.shared.playground,
                    heading_offset: HeadingOffset::H2
                }
                .into_string()
            );
        }
    }
    if !default_impl_items.is_empty() || !impl_items.is_empty() {
        w.write_str("<div class=\"impl-items\">");
        w.push_buffer(default_impl_items);
        w.push_buffer(impl_items);
        close_tags.insert_str(0, "</div>");
    }
    w.write_str(&close_tags);
}

// Render the items that appear on the right side of methods, impls, and
// associated types. For example "1.0.0 (const: 1.39.0) [src]".
fn render_rightside(
    w: &mut Buffer,
    cx: &Context<'_>,
    item: &clean::Item,
    containing_item: &clean::Item,
) {
    let tcx = cx.tcx();

    write!(w, "<div class=\"rightside\">");
    render_stability_since_raw(
        w,
        item.stable_since(tcx).as_deref(),
        item.const_stability(tcx),
        containing_item.stable_since(tcx).as_deref(),
        containing_item.const_stable_since(tcx).as_deref(),
    );

    write_srclink(cx, item, w);
    w.write_str("</div>");
}

pub(crate) fn render_impl_summary(
    w: &mut Buffer,
    cx: &Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    containing_item: &clean::Item,
    show_def_docs: bool,
    use_absolute: Option<bool>,
    is_on_foreign_type: bool,
    // This argument is used to reference same type with different paths to avoid duplication
    // in documentation pages for trait with automatic implementations like "Send" and "Sync".
    aliases: &[String],
) {
    let id = cx.derive_id(match i.inner_impl().trait_ {
        Some(ref t) => {
            if is_on_foreign_type {
                get_id_for_impl_on_foreign_type(&i.inner_impl().for_, t, cx)
            } else {
                format!("impl-{}", small_url_encode(format!("{:#}", t.print(cx))))
            }
        }
        None => "impl".to_string(),
    });
    let aliases = if aliases.is_empty() {
        String::new()
    } else {
        format!(" data-aliases=\"{}\"", aliases.join(","))
    };
    write!(w, "<div id=\"{}\" class=\"impl has-srclink\"{}>", id, aliases);
    render_rightside(w, cx, &i.impl_item, containing_item);
    write!(w, "<a href=\"#{}\" class=\"anchor\"></a>", id);
    write!(w, "<h3 class=\"code-header in-band\">");

    if let Some(use_absolute) = use_absolute {
        write!(w, "{}", i.inner_impl().print(use_absolute, cx));
        if show_def_docs {
            for it in &i.inner_impl().items {
                if let clean::TypedefItem(ref tydef, _) = *it.kind {
                    w.write_str("<span class=\"where fmt-newline\">  ");
                    assoc_type(w, it, &[], Some(&tydef.type_), AssocItemLink::Anchor(None), "", cx);
                    w.write_str(";</span>");
                }
            }
        }
    } else {
        write!(w, "{}", i.inner_impl().print(false, cx));
    }
    write!(w, "</h3>");

    let is_trait = i.inner_impl().trait_.is_some();
    if is_trait {
        if let Some(portability) = portability(&i.impl_item, Some(parent)) {
            write!(w, "<div class=\"item-info\">{}</div>", portability);
        }
    }

    w.write_str("</div>");
}

fn print_sidebar(cx: &Context<'_>, it: &clean::Item, buffer: &mut Buffer) {
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
            "<h2 class=\"location\">{}{}</h2>",
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
        if let Some(ref version) = cx.cache().crate_version {
            write!(
                buffer,
                "<div class=\"block version\">\
                     <div class=\"narrow-helper\"></div>\
                     <p>Version {}</p>\
                 </div>",
                Escape(version),
            );
        }
    }

    buffer.write_str("<div class=\"sidebar-elems\">");
    if it.is_crate() {
        write!(
            buffer,
            "<a id=\"all-types\" href=\"all.html\"><p>See all {}'s items</p></a>",
            it.name.as_ref().expect("crates always have a name"),
        );
    }

    match *it.kind {
        clean::StructItem(ref s) => sidebar_struct(cx, buffer, it, s),
        clean::TraitItem(ref t) => sidebar_trait(cx, buffer, it, t),
        clean::PrimitiveItem(_) => sidebar_primitive(cx, buffer, it),
        clean::UnionItem(ref u) => sidebar_union(cx, buffer, it, u),
        clean::EnumItem(ref e) => sidebar_enum(cx, buffer, it, e),
        clean::TypedefItem(_, _) => sidebar_typedef(cx, buffer, it),
        clean::ModuleItem(ref m) => sidebar_module(buffer, &m.items),
        clean::ForeignTypeItem => sidebar_foreign_type(cx, buffer, it),
        _ => {}
    }

    // The sidebar is designed to display sibling functions, modules and
    // other miscellaneous information. since there are lots of sibling
    // items (and that causes quadratic growth in large modules),
    // we refactor common parts into a shared JavaScript file per module.
    // still, we don't move everything into JS because we want to preserve
    // as much HTML as possible in order to allow non-JS-enabled browsers
    // to navigate the documentation (though slightly inefficiently).

    if !it.is_mod() {
        buffer.write_str("<h2 class=\"location\">Other items in<br>");
        for (i, name) in cx.current.iter().take(parentlen).enumerate() {
            if i > 0 {
                buffer.write_str("::<wbr>");
            }
            write!(
                buffer,
                "<a href=\"{}index.html\">{}</a>",
                &cx.root_path()[..(cx.current.len() - i - 1) * 3],
                *name
            );
        }
        buffer.write_str("</h2>");
    }

    // Sidebar refers to the enclosing module, not this module.
    let relpath = if it.is_mod() && parentlen != 0 { "./" } else { "" };
    write!(
        buffer,
        "<div id=\"sidebar-vars\" data-name=\"{name}\" data-ty=\"{ty}\" data-relpath=\"{path}\">\
        </div>",
        name = it.name.unwrap_or(kw::Empty),
        ty = it.type_(),
        path = relpath
    );
    write!(buffer, "<script defer src=\"{}sidebar-items.js\"></script>", relpath);
    // Closes sidebar-elems div.
    buffer.write_str("</div>");
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

struct SidebarLink {
    name: Symbol,
    url: String,
}

impl fmt::Display for SidebarLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<a href=\"#{}\">{}</a>", self.url, self.name)
    }
}

impl PartialEq for SidebarLink {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
    }
}

impl Eq for SidebarLink {}

impl PartialOrd for SidebarLink {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SidebarLink {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.url.cmp(&other.url)
    }
}

fn get_methods(
    i: &clean::Impl,
    for_deref: bool,
    used_links: &mut FxHashSet<String>,
    deref_mut: bool,
    tcx: TyCtxt<'_>,
) -> Vec<SidebarLink> {
    i.items
        .iter()
        .filter_map(|item| match item.name {
            Some(name) if !name.is_empty() && item.is_method() => {
                if !for_deref || should_render_item(item, deref_mut, tcx) {
                    Some(SidebarLink {
                        name,
                        url: get_next_url(used_links, format!("method.{}", name)),
                    })
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

fn get_associated_constants(
    i: &clean::Impl,
    used_links: &mut FxHashSet<String>,
) -> Vec<SidebarLink> {
    i.items
        .iter()
        .filter_map(|item| match item.name {
            Some(name) if !name.is_empty() && item.is_associated_const() => Some(SidebarLink {
                name,
                url: get_next_url(used_links, format!("associatedconstant.{}", name)),
            }),
            _ => None,
        })
        .collect::<Vec<_>>()
}

// The point is to url encode any potential character from a type with genericity.
fn small_url_encode(s: String) -> String {
    let mut st = String::new();
    let mut last_match = 0;
    for (idx, c) in s.char_indices() {
        let escaped = match c {
            '<' => "%3C",
            '>' => "%3E",
            ' ' => "%20",
            '?' => "%3F",
            '\'' => "%27",
            '&' => "%26",
            ',' => "%2C",
            ':' => "%3A",
            ';' => "%3B",
            '[' => "%5B",
            ']' => "%5D",
            '"' => "%22",
            _ => continue,
        };

        st += &s[last_match..idx];
        st += escaped;
        // NOTE: we only expect single byte characters here - which is fine as long as we
        // only match single byte characters
        last_match = idx + 1;
    }

    if last_match != 0 {
        st += &s[last_match..];
        st
    } else {
        s
    }
}

fn sidebar_assoc_items(cx: &Context<'_>, out: &mut Buffer, it: &clean::Item) {
    let did = it.def_id.expect_def_id();
    let cache = cx.cache();
    if let Some(v) = cache.impls.get(&did) {
        let mut used_links = FxHashSet::default();

        {
            let used_links_bor = &mut used_links;
            let mut assoc_consts = v
                .iter()
                .flat_map(|i| get_associated_constants(i.inner_impl(), used_links_bor))
                .collect::<Vec<_>>();
            if !assoc_consts.is_empty() {
                // We want links' order to be reproducible so we don't use unstable sort.
                assoc_consts.sort();

                out.push_str(
                    "<h3 class=\"sidebar-title\">\
                        <a href=\"#implementations\">Associated Constants</a>\
                     </h3>\
                     <div class=\"sidebar-links\">",
                );
                for line in assoc_consts {
                    write!(out, "{}", line);
                }
                out.push_str("</div>");
            }
            let mut methods = v
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| get_methods(i.inner_impl(), false, used_links_bor, false, cx.tcx()))
                .collect::<Vec<_>>();
            if !methods.is_empty() {
                // We want links' order to be reproducible so we don't use unstable sort.
                methods.sort();

                out.push_str(
                    "<h3 class=\"sidebar-title\"><a href=\"#implementations\">Methods</a></h3>\
                     <div class=\"sidebar-links\">",
                );
                for line in methods {
                    write!(out, "{}", line);
                }
                out.push_str("</div>");
            }
        }

        if v.iter().any(|i| i.inner_impl().trait_.is_some()) {
            if let Some(impl_) =
                v.iter().find(|i| i.trait_did() == cx.tcx().lang_items().deref_trait())
            {
                sidebar_deref_methods(cx, out, impl_, v);
            }

            let format_impls = |impls: Vec<&Impl>| {
                let mut links = FxHashSet::default();

                let mut ret = impls
                    .iter()
                    .filter_map(|it| {
                        if let Some(ref i) = it.inner_impl().trait_ {
                            let i_display = format!("{:#}", i.print(cx));
                            let out = Escape(&i_display);
                            let encoded = small_url_encode(format!("{:#}", i.print(cx)));
                            let generated = format!(
                                "<a href=\"#impl-{}\">{}{}</a>",
                                encoded,
                                if it.inner_impl().negative_polarity { "!" } else { "" },
                                out
                            );
                            if links.insert(generated.clone()) { Some(generated) } else { None }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<String>>();
                ret.sort();
                ret
            };

            let write_sidebar_links = |out: &mut Buffer, links: Vec<String>| {
                out.push_str("<div class=\"sidebar-links\">");
                for link in links {
                    out.push_str(&link);
                }
                out.push_str("</div>");
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
                    "<h3 class=\"sidebar-title\"><a href=\"#trait-implementations\">\
                        Trait Implementations</a></h3>",
                );
                write_sidebar_links(out, concrete_format);
            }

            if !synthetic_format.is_empty() {
                out.push_str(
                    "<h3 class=\"sidebar-title\"><a href=\"#synthetic-implementations\">\
                        Auto Trait Implementations</a></h3>",
                );
                write_sidebar_links(out, synthetic_format);
            }

            if !blanket_format.is_empty() {
                out.push_str(
                    "<h3 class=\"sidebar-title\"><a href=\"#blanket-implementations\">\
                        Blanket Implementations</a></h3>",
                );
                write_sidebar_links(out, blanket_format);
            }
        }
    }
}

fn sidebar_deref_methods(cx: &Context<'_>, out: &mut Buffer, impl_: &Impl, v: &Vec<Impl>) {
    let c = cx.cache();

    debug!("found Deref: {:?}", impl_);
    if let Some((target, real_target)) =
        impl_.inner_impl().items.iter().find_map(|item| match *item.kind {
            clean::TypedefItem(ref t, true) => Some(match *t {
                clean::Typedef { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
    {
        debug!("found target, real_target: {:?} {:?}", target, real_target);
        if let Some(did) = target.def_id_full(c) {
            if let Some(type_did) = impl_.inner_impl().for_.def_id_full(c) {
                // `impl Deref<Target = S> for S`
                if did == type_did {
                    // Avoid infinite cycles
                    return;
                }
            }
        }
        let deref_mut = v.iter().any(|i| i.trait_did() == cx.tcx().lang_items().deref_mut_trait());
        let inner_impl = target
            .def_id_full(c)
            .or_else(|| {
                target.primitive_type().and_then(|prim| c.primitive_locations.get(&prim).cloned())
            })
            .and_then(|did| c.impls.get(&did));
        if let Some(impls) = inner_impl {
            debug!("found inner_impl: {:?}", impls);
            let mut used_links = FxHashSet::default();
            let mut ret = impls
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| {
                    get_methods(i.inner_impl(), true, &mut used_links, deref_mut, cx.tcx())
                })
                .collect::<Vec<_>>();
            if !ret.is_empty() {
                write!(
                    out,
                    "<h3 class=\"sidebar-title\"><a href=\"#deref-methods\">Methods from {}&lt;Target={}&gt;</a></h3>",
                    Escape(&format!("{:#}", impl_.inner_impl().trait_.as_ref().unwrap().print(cx))),
                    Escape(&format!("{:#}", real_target.print(cx))),
                );
                // We want links' order to be reproducible so we don't use unstable sort.
                ret.sort();
                out.push_str("<div class=\"sidebar-links\">");
                for link in ret {
                    write!(out, "{}", link);
                }
                out.push_str("</div>");
            }
        }
    }
}

fn sidebar_struct(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, s: &clean::Struct) {
    let mut sidebar = Buffer::new();
    let fields = get_struct_fields_name(&s.fields);

    if !fields.is_empty() {
        if let CtorKind::Fictive = s.struct_type {
            sidebar.push_str(
                "<h3 class=\"sidebar-title\"><a href=\"#fields\">Fields</a></h3>\
                <div class=\"sidebar-links\">",
            );

            for field in fields {
                sidebar.push_str(&field);
            }

            sidebar.push_str("</div>");
        } else if let CtorKind::Fn = s.struct_type {
            sidebar
                .push_str("<h3 class=\"sidebar-title\"><a href=\"#fields\">Tuple Fields</a></h3>");
        }
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

fn get_id_for_impl_on_foreign_type(
    for_: &clean::Type,
    trait_: &clean::Path,
    cx: &Context<'_>,
) -> String {
    small_url_encode(format!("impl-{:#}-for-{:#}", trait_.print(cx), for_.print(cx)))
}

fn extract_for_impl_name(item: &clean::Item, cx: &Context<'_>) -> Option<(String, String)> {
    match *item.kind {
        clean::ItemKind::ImplItem(ref i) => {
            if let Some(ref trait_) = i.trait_ {
                // Alternative format produces no URLs,
                // so this parameter does nothing.
                Some((
                    format!("{:#}", i.for_.print(cx)),
                    get_id_for_impl_on_foreign_type(&i.for_, trait_, cx),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn sidebar_trait(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, t: &clean::Trait) {
    buf.write_str("<div class=\"block items\">");

    fn print_sidebar_section(
        out: &mut Buffer,
        items: &[clean::Item],
        before: &str,
        filter: impl Fn(&clean::Item) -> bool,
        write: impl Fn(&mut Buffer, &str),
        after: &str,
    ) {
        let mut items = items
            .iter()
            .filter_map(|m| match m.name {
                Some(ref name) if filter(m) => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if !items.is_empty() {
            items.sort_unstable();
            out.push_str(before);
            for item in items.into_iter() {
                write(out, &item);
            }
            out.push_str(after);
        }
    }

    print_sidebar_section(
        buf,
        &t.items,
        "<h3 class=\"sidebar-title\"><a href=\"#associated-types\">\
            Associated Types</a></h3><div class=\"sidebar-links\">",
        |m| m.is_associated_type(),
        |out, sym| write!(out, "<a href=\"#associatedtype.{0}\">{0}</a>", sym),
        "</div>",
    );

    print_sidebar_section(
        buf,
        &t.items,
        "<h3 class=\"sidebar-title\"><a href=\"#associated-const\">\
            Associated Constants</a></h3><div class=\"sidebar-links\">",
        |m| m.is_associated_const(),
        |out, sym| write!(out, "<a href=\"#associatedconstant.{0}\">{0}</a>", sym),
        "</div>",
    );

    print_sidebar_section(
        buf,
        &t.items,
        "<h3 class=\"sidebar-title\"><a href=\"#required-methods\">\
            Required Methods</a></h3><div class=\"sidebar-links\">",
        |m| m.is_ty_method(),
        |out, sym| write!(out, "<a href=\"#tymethod.{0}\">{0}</a>", sym),
        "</div>",
    );

    print_sidebar_section(
        buf,
        &t.items,
        "<h3 class=\"sidebar-title\"><a href=\"#provided-methods\">\
            Provided Methods</a></h3><div class=\"sidebar-links\">",
        |m| m.is_method(),
        |out, sym| write!(out, "<a href=\"#method.{0}\">{0}</a>", sym),
        "</div>",
    );

    let cache = cx.cache();
    if let Some(implementors) = cache.implementors.get(&it.def_id.expect_def_id()) {
        let mut res = implementors
            .iter()
            .filter(|i| {
                i.inner_impl()
                    .for_
                    .def_id_full(cache)
                    .map_or(false, |d| !cache.paths.contains_key(&d))
            })
            .filter_map(|i| extract_for_impl_name(&i.impl_item, cx))
            .collect::<Vec<_>>();

        if !res.is_empty() {
            res.sort();
            buf.push_str(
                "<h3 class=\"sidebar-title\"><a href=\"#foreign-impls\">\
                    Implementations on Foreign Types</a></h3>\
                 <div class=\"sidebar-links\">",
            );
            for (name, id) in res.into_iter() {
                write!(buf, "<a href=\"#{}\">{}</a>", id, Escape(&name));
            }
            buf.push_str("</div>");
        }
    }

    sidebar_assoc_items(cx, buf, it);

    buf.push_str("<h3 class=\"sidebar-title\"><a href=\"#implementors\">Implementors</a></h3>");
    if t.is_auto {
        buf.push_str(
            "<h3 class=\"sidebar-title\"><a \
                href=\"#synthetic-implementors\">Auto Implementors</a></h3>",
        );
    }

    buf.push_str("</div>")
}

fn sidebar_primitive(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();
    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

fn sidebar_typedef(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();
    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

fn get_struct_fields_name(fields: &[clean::Item]) -> Vec<String> {
    let mut fields = fields
        .iter()
        .filter(|f| matches!(*f.kind, clean::StructFieldItem(..)))
        .filter_map(|f| {
            f.name.map(|name| format!("<a href=\"#structfield.{name}\">{name}</a>", name = name))
        })
        .collect::<Vec<_>>();
    fields.sort();
    fields
}

fn sidebar_union(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, u: &clean::Union) {
    let mut sidebar = Buffer::new();
    let fields = get_struct_fields_name(&u.fields);

    if !fields.is_empty() {
        sidebar.push_str(
            "<h3 class=\"sidebar-title\"><a href=\"#fields\">Fields</a></h3>\
            <div class=\"sidebar-links\">",
        );

        for field in fields {
            sidebar.push_str(&field);
        }

        sidebar.push_str("</div>");
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

fn sidebar_enum(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, e: &clean::Enum) {
    let mut sidebar = Buffer::new();

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
            "<h3 class=\"sidebar-title\"><a href=\"#variants\">Variants</a></h3>\
             <div class=\"sidebar-links\">{}</div>",
            variants.join(""),
        ));
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

fn item_ty_to_strs(ty: ItemType) -> (&'static str, &'static str) {
    match ty {
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

    // Re-exports are handled a bit differently because they can be extern crates or imports.
    if items.iter().any(|it| {
        it.name.is_some()
            && (it.type_() == ItemType::ExternCrate
                || (it.type_() == ItemType::Import && !it.is_stripped()))
    }) {
        let (id, name) = item_ty_to_strs(ItemType::Import);
        sidebar.push_str(&format!("<li><a href=\"#{}\">{}</a></li>", id, name));
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
        if items.iter().any(|it| !it.is_stripped() && it.type_() == myty && it.name.is_some()) {
            let (id, name) = item_ty_to_strs(myty);
            sidebar.push_str(&format!("<li><a href=\"#{}\">{}</a></li>", id, name));
        }
    }

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\"><ul>{}</ul></div>", sidebar);
    }
}

fn sidebar_foreign_type(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();
    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<div class=\"block items\">{}</div>", sidebar.into_inner());
    }
}

crate const BASIC_KEYWORDS: &str = "rust, rustlang, rust-lang";

/// Returns a list of all paths used in the type.
/// This is used to help deduplicate imported impls
/// for reexported types. If any of the contained
/// types are re-exported, we don't use the corresponding
/// entry from the js file, as inlining will have already
/// picked up the impl
fn collect_paths_for_type(first_ty: clean::Type, cache: &Cache) -> Vec<String> {
    let mut out = Vec::new();
    let mut visited = FxHashSet::default();
    let mut work = VecDeque::new();

    let mut process_path = |did: DefId| {
        let get_extern = || cache.external_paths.get(&did).map(|s| s.0.clone());
        let fqp = cache.exact_paths.get(&did).cloned().or_else(get_extern);

        if let Some(path) = fqp {
            out.push(path.join("::"));
        }
    };

    work.push_back(first_ty);

    while let Some(ty) = work.pop_front() {
        if !visited.insert(ty.clone()) {
            continue;
        }

        match ty {
            clean::Type::ResolvedPath { did, .. } => process_path(did),
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
                process_path(trait_.def_id());
            }
            _ => {}
        }
    }
    out
}
