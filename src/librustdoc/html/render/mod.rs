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

pub(crate) mod search_index;

#[cfg(test)]
mod tests;

mod context;
mod print_item;
mod span_map;
mod write_shared;

pub(crate) use self::context::*;
pub(crate) use self::span_map::{collect_spans_and_sources, LinkFromSrc};

use std::collections::VecDeque;
use std::default::Default;
use std::fmt;
use std::fs;
use std::iter::Peekable;
use std::path::PathBuf;
use std::rc::Rc;
use std::str;
use std::string::ToString;

use rustc_ast_pretty::pprust;
use rustc_attr::{ConstStability, Deprecation, StabilityLevel};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_hir::Mutability;
use rustc_middle::middle::stability;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::{
    symbol::{sym, Symbol},
    BytePos, FileName, RealFileName,
};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

use crate::clean::{self, ItemId, RenderedLink, SelfTy};
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::formats::{AssocItemRender, Impl, RenderMode};
use crate::html::escape::Escape;
use crate::html::format::{
    href, join_with_double_colon, print_abi_with_space, print_constness_with_space,
    print_default_space, print_generic_bounds, print_where_clause, visibility_print_with_space,
    Buffer, Ending, HrefError, PrintWithSpace,
};
use crate::html::highlight;
use crate::html::markdown::{
    HeadingOffset, IdMap, Markdown, MarkdownItemInfo, MarkdownSummaryLine,
};
use crate::html::sources;
use crate::html::static_files::SCRAPE_EXAMPLES_HELP_MD;
use crate::scrape_examples::{CallData, CallLocation};
use crate::try_none;
use crate::DOC_RUST_LANG_ORG_CHANNEL;

/// A pair of name and its optional document.
pub(crate) type NameDoc = (String, Option<String>);

pub(crate) fn ensure_trailing_slash(v: &str) -> impl fmt::Display + '_ {
    crate::html::format::display_fn(move |f| {
        if !v.ends_with('/') && !v.is_empty() { write!(f, "{}/", v) } else { f.write_str(v) }
    })
}

// Helper structs for rendering items/sidebars and carrying along contextual
// information

/// Struct representing one entry in the JS search index. These are all emitted
/// by hand to a large JS file at the end of cache-creation.
#[derive(Debug)]
pub(crate) struct IndexItem {
    pub(crate) ty: ItemType,
    pub(crate) name: Symbol,
    pub(crate) path: String,
    pub(crate) desc: String,
    pub(crate) parent: Option<DefId>,
    pub(crate) parent_idx: Option<usize>,
    pub(crate) search_type: Option<IndexItemFunctionType>,
    pub(crate) aliases: Box<[Symbol]>,
}

/// A type used for the search index.
#[derive(Debug)]
pub(crate) struct RenderType {
    id: Option<RenderTypeId>,
    generics: Option<Vec<RenderType>>,
}

impl Serialize for RenderType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let id = match &self.id {
            // 0 is a sentinel, everything else is one-indexed
            None => 0,
            Some(RenderTypeId::Index(idx)) => idx + 1,
            _ => panic!("must convert render types to indexes before serializing"),
        };
        if let Some(generics) = &self.generics {
            let mut seq = serializer.serialize_seq(None)?;
            seq.serialize_element(&id)?;
            seq.serialize_element(generics)?;
            seq.end()
        } else {
            id.serialize(serializer)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RenderTypeId {
    DefId(DefId),
    Primitive(clean::PrimitiveType),
    Index(usize),
}

/// Full type of functions/methods in the search index.
#[derive(Debug)]
pub(crate) struct IndexItemFunctionType {
    inputs: Vec<RenderType>,
    output: Vec<RenderType>,
}

impl Serialize for IndexItemFunctionType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // If we couldn't figure out a type, just write `0`.
        let has_missing = self
            .inputs
            .iter()
            .chain(self.output.iter())
            .any(|i| i.id.is_none() && i.generics.is_none());
        if has_missing {
            0.serialize(serializer)
        } else {
            let mut seq = serializer.serialize_seq(None)?;
            match &self.inputs[..] {
                [one] if one.generics.is_none() => seq.serialize_element(one)?,
                _ => seq.serialize_element(&self.inputs)?,
            }
            match &self.output[..] {
                [] => {}
                [one] if one.generics.is_none() => seq.serialize_element(one)?,
                _ => seq.serialize_element(&self.output)?,
            }
            seq.end()
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StylePath {
    /// The path to the theme
    pub(crate) path: PathBuf,
}

impl StylePath {
    pub(crate) fn basename(&self) -> Result<String, Error> {
        Ok(try_none!(try_none!(self.path.file_stem(), &self.path).to_str(), &self.path).to_string())
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
    pub(crate) fn print(&self) -> impl fmt::Display + '_ {
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
    attribute_macros: FxHashSet<ItemEntry>,
    derive_macros: FxHashSet<ItemEntry>,
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
            attribute_macros: new_set(100),
            derive_macros: new_set(100),
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
                ItemType::ProcAttribute => {
                    self.attribute_macros.insert(ItemEntry::new(new_url, name))
                }
                ItemType::ProcDerive => self.derive_macros.insert(ItemEntry::new(new_url, name)),
                ItemType::TraitAlias => self.trait_aliases.insert(ItemEntry::new(new_url, name)),
                _ => true,
            };
        }
    }

    fn item_sections(&self) -> FxHashSet<ItemSection> {
        let mut sections = FxHashSet::default();

        if !self.structs.is_empty() {
            sections.insert(ItemSection::Structs);
        }
        if !self.enums.is_empty() {
            sections.insert(ItemSection::Enums);
        }
        if !self.unions.is_empty() {
            sections.insert(ItemSection::Unions);
        }
        if !self.primitives.is_empty() {
            sections.insert(ItemSection::PrimitiveTypes);
        }
        if !self.traits.is_empty() {
            sections.insert(ItemSection::Traits);
        }
        if !self.macros.is_empty() {
            sections.insert(ItemSection::Macros);
        }
        if !self.functions.is_empty() {
            sections.insert(ItemSection::Functions);
        }
        if !self.typedefs.is_empty() {
            sections.insert(ItemSection::TypeDefinitions);
        }
        if !self.opaque_tys.is_empty() {
            sections.insert(ItemSection::OpaqueTypes);
        }
        if !self.statics.is_empty() {
            sections.insert(ItemSection::Statics);
        }
        if !self.constants.is_empty() {
            sections.insert(ItemSection::Constants);
        }
        if !self.attribute_macros.is_empty() {
            sections.insert(ItemSection::AttributeMacros);
        }
        if !self.derive_macros.is_empty() {
            sections.insert(ItemSection::DeriveMacros);
        }
        if !self.trait_aliases.is_empty() {
            sections.insert(ItemSection::TraitAliases);
        }

        sections
    }

    fn print(self, f: &mut Buffer) {
        fn print_entries(f: &mut Buffer, e: &FxHashSet<ItemEntry>, kind: ItemSection) {
            if !e.is_empty() {
                let mut e: Vec<&ItemEntry> = e.iter().collect();
                e.sort();
                write!(
                    f,
                    "<h3 id=\"{id}\">{title}</h3><ul class=\"all-items\">",
                    id = kind.id(),
                    title = kind.name(),
                );

                for s in e.iter() {
                    write!(f, "<li>{}</li>", s.print());
                }

                f.write_str("</ul>");
            }
        }

        f.write_str("<h1>List of all items</h1>");
        // Note: print_entries does not escape the title, because we know the current set of titles
        // doesn't require escaping.
        print_entries(f, &self.structs, ItemSection::Structs);
        print_entries(f, &self.enums, ItemSection::Enums);
        print_entries(f, &self.unions, ItemSection::Unions);
        print_entries(f, &self.primitives, ItemSection::PrimitiveTypes);
        print_entries(f, &self.traits, ItemSection::Traits);
        print_entries(f, &self.macros, ItemSection::Macros);
        print_entries(f, &self.attribute_macros, ItemSection::AttributeMacros);
        print_entries(f, &self.derive_macros, ItemSection::DeriveMacros);
        print_entries(f, &self.functions, ItemSection::Functions);
        print_entries(f, &self.typedefs, ItemSection::TypeDefinitions);
        print_entries(f, &self.trait_aliases, ItemSection::TraitAliases);
        print_entries(f, &self.opaque_tys, ItemSection::OpaqueTypes);
        print_entries(f, &self.statics, ItemSection::Statics);
        print_entries(f, &self.constants, ItemSection::Constants);
    }
}

fn scrape_examples_help(shared: &SharedContext<'_>) -> String {
    let mut content = SCRAPE_EXAMPLES_HELP_MD.to_owned();
    content.push_str(&format!(
      "## More information\n\n\
      If you want more information about this feature, please read the [corresponding chapter in the Rustdoc book]({}/rustdoc/scraped-examples.html).",
      DOC_RUST_LANG_ORG_CHANNEL));

    let mut ids = IdMap::default();
    format!(
        "<div class=\"main-heading\">\
            <h1>About scraped examples</h1>\
        </div>\
        <div>{}</div>",
        Markdown {
            content: &content,
            links: &[],
            ids: &mut ids,
            error_codes: shared.codes,
            edition: shared.edition(),
            playground: &shared.playground,
            heading_offset: HeadingOffset::H1
        }
        .into_string()
    )
}

fn document(
    w: &mut Buffer,
    cx: &mut Context<'_>,
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
    cx: &mut Context<'_>,
    md_text: &str,
    links: Vec<RenderedLink>,
    heading_offset: HeadingOffset,
) {
    write!(
        w,
        "<div class=\"docblock\">{}</div>",
        Markdown {
            content: md_text,
            links: &links,
            ids: &mut cx.id_map,
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
    cx: &mut Context<'_>,
    link: AssocItemLink<'_>,
    parent: &clean::Item,
    show_def_docs: bool,
) {
    document_item_info(w, cx, item, Some(parent));
    if !show_def_docs {
        return;
    }
    if let Some(s) = item.doc_value() {
        let (mut summary_html, has_more_content) =
            MarkdownSummaryLine(&s, &item.links(cx)).into_string_with_has_more_content();

        if has_more_content {
            let link = format!(r#" <a{}>Read more</a>"#, assoc_href_attr(item, link, cx));

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
    cx: &mut Context<'_>,
    heading_offset: HeadingOffset,
) {
    document_full_inner(w, item, cx, true, heading_offset);
}

fn document_full(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &mut Context<'_>,
    heading_offset: HeadingOffset,
) {
    document_full_inner(w, item, cx, false, heading_offset);
}

fn document_full_inner(
    w: &mut Buffer,
    item: &clean::Item,
    cx: &mut Context<'_>,
    is_collapsible: bool,
    heading_offset: HeadingOffset,
) {
    if let Some(s) = item.collapsed_doc_value() {
        debug!("Doc block: =====\n{}\n=====", s);
        if is_collapsible {
            w.write_str(
                "<details class=\"toggle top-doc\" open>\
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

    let kind = match &*item.kind {
        clean::ItemKind::StrippedItem(box kind) | kind => kind,
    };

    if let clean::ItemKind::FunctionItem(..) | clean::ItemKind::MethodItem(..) = kind {
        render_call_locations(w, cx, item);
    }
}

/// Add extra information about an item such as:
///
/// * Stability
/// * Deprecated
/// * Required features (through the `doc_cfg` feature)
fn document_item_info(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    item: &clean::Item,
    parent: Option<&clean::Item>,
) {
    let item_infos = short_item_info(item, cx, parent);
    if !item_infos.is_empty() {
        w.write_str("<span class=\"item-info\">");
        for info in item_infos {
            w.write_str(&info);
        }
        w.write_str("</span>");
    }
}

fn portability(item: &clean::Item, parent: Option<&clean::Item>) -> Option<String> {
    let cfg = match (&item.cfg, parent.and_then(|p| p.cfg.as_ref())) {
        (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
        (cfg, _) => cfg.as_deref().cloned(),
    };

    debug!(
        "Portability {:?} {:?} (parent: {:?}) - {:?} = {:?}",
        item.name,
        item.cfg,
        parent,
        parent.and_then(|p| p.cfg.as_ref()),
        cfg
    );

    Some(format!("<div class=\"stab portability\">{}</div>", cfg?.render_long_html()))
}

/// Render the stability, deprecation and portability information that is displayed at the top of
/// the item's documentation.
fn short_item_info(
    item: &clean::Item,
    cx: &mut Context<'_>,
    parent: Option<&clean::Item>,
) -> Vec<String> {
    let mut extra_info = vec![];

    if let Some(depr @ Deprecation { note, since, is_since_rustc_version: _, suggestion: _ }) =
        item.deprecation(cx.tcx())
    {
        // We display deprecation messages for #[deprecated], but only display
        // the future-deprecation messages for rustc versions.
        let mut message = if let Some(since) = since {
            let since = since.as_str();
            if !stability::deprecation_in_effect(&depr) {
                if since == "TBD" {
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
            let html = MarkdownItemInfo(note, &mut cx.id_map);
            message.push_str(&format!(": {}", html.into_string()));
        }
        extra_info.push(format!(
            "<div class=\"stab deprecated\">\
                 <span class=\"emoji\">👎</span>\
                 <span>{}</span>\
             </div>",
            message,
        ));
    }

    // Render unstable items. But don't render "rustc_private" crates (internal compiler crates).
    // Those crates are permanently unstable so it makes no sense to render "unstable" everywhere.
    if let Some((StabilityLevel::Unstable { reason: _, issue, .. }, feature)) = item
        .stability(cx.tcx())
        .as_ref()
        .filter(|stab| stab.feature != sym::rustc_private)
        .map(|stab| (stab.level, stab.feature))
    {
        let mut message = "<span class=\"emoji\">🔬</span>\
             <span>This is a nightly-only experimental API."
            .to_owned();

        let mut feature = format!("<code>{}</code>", Escape(feature.as_str()));
        if let (Some(url), Some(issue)) = (&cx.shared.issue_tracker_base_url, issue) {
            feature.push_str(&format!(
                "&nbsp;<a href=\"{url}{issue}\">#{issue}</a>",
                url = url,
                issue = issue
            ));
        }

        message.push_str(&format!(" ({})</span>", feature));

        extra_info.push(format!("<div class=\"stab unstable\">{}</div>", message));
    }

    if let Some(portability) = portability(item, parent) {
        extra_info.push(portability);
    }

    extra_info
}

// Render the list of items inside one of the sections "Trait Implementations",
// "Auto Trait Implementations," "Blanket Trait Implementations" (on struct/enum pages).
pub(crate) fn render_impls(
    cx: &mut Context<'_>,
    w: &mut Buffer,
    impls: &[&Impl],
    containing_item: &clean::Item,
    toggle_open_by_default: bool,
) {
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
                    show_default_items: true,
                    show_non_assoc_items: true,
                    toggle_open_by_default,
                },
            );
            buffer.into_inner()
        })
        .collect::<Vec<_>>();
    rendered_impls.sort();
    w.write_str(&rendered_impls.join(""));
}

/// Build a (possibly empty) `href` attribute (a key-value pair) for the given associated item.
fn assoc_href_attr(it: &clean::Item, link: AssocItemLink<'_>, cx: &Context<'_>) -> String {
    let name = it.name.unwrap();
    let item_type = it.type_();

    let href = match link {
        AssocItemLink::Anchor(Some(ref id)) => Some(format!("#{}", id)),
        AssocItemLink::Anchor(None) => Some(format!("#{}.{}", item_type, name)),
        AssocItemLink::GotoSource(did, provided_methods) => {
            // We're creating a link from the implementation of an associated item to its
            // declaration in the trait declaration.
            let item_type = match item_type {
                // For historical but not technical reasons, the item type of methods in
                // trait declarations depends on whether the method is required (`TyMethod`) or
                // provided (`Method`).
                ItemType::Method | ItemType::TyMethod => {
                    if provided_methods.contains(&name) {
                        ItemType::Method
                    } else {
                        ItemType::TyMethod
                    }
                }
                // For associated types and constants, no such distinction exists.
                item_type => item_type,
            };

            match href(did.expect_def_id(), cx) {
                Ok((url, ..)) => Some(format!("{}#{}.{}", url, item_type, name)),
                // The link is broken since it points to an external crate that wasn't documented.
                // Do not create any link in such case. This is better than falling back to a
                // dummy anchor like `#{item_type}.{name}` representing the `id` of *this* impl item
                // (that used to happen in older versions). Indeed, in most cases this dummy would
                // coincide with the `id`. However, it would not always do so.
                // In general, this dummy would be incorrect:
                // If the type with the trait impl also had an inherent impl with an assoc. item of
                // the *same* name as this impl item, the dummy would link to that one even though
                // those two items are distinct!
                // In this scenario, the actual `id` of this impl item would be
                // `#{item_type}.{name}-{n}` for some number `n` (a disambiguator).
                Err(HrefError::DocumentationNotBuilt) => None,
                Err(_) => Some(format!("#{}.{}", item_type, name)),
            }
        }
    };

    // If there is no `href` for the reason explained above, simply do not render it which is valid:
    // https://html.spec.whatwg.org/multipage/links.html#links-created-by-a-and-area-elements
    href.map(|href| format!(" href=\"{}\"", href)).unwrap_or_default()
}

fn assoc_const(
    w: &mut Buffer,
    it: &clean::Item,
    ty: &clean::Type,
    default: Option<&clean::ConstantKind>,
    link: AssocItemLink<'_>,
    extra: &str,
    cx: &Context<'_>,
) {
    let tcx = cx.tcx();
    write!(
        w,
        "{extra}{vis}const <a{href} class=\"constant\">{name}</a>: {ty}",
        extra = extra,
        vis = visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
        href = assoc_href_attr(it, link, cx),
        name = it.name.as_ref().unwrap(),
        ty = ty.print(cx),
    );
    if let Some(default) = default {
        write!(w, " = ");

        // FIXME: `.value()` uses `clean::utils::format_integer_with_underscore_sep` under the
        //        hood which adds noisy underscores and a type suffix to number literals.
        //        This hurts readability in this context especially when more complex expressions
        //        are involved and it doesn't add much of value.
        //        Find a way to print constants here without all that jazz.
        write!(w, "{}", Escape(&default.value(tcx).unwrap_or_else(|| default.expr(tcx))));
    }
}

fn assoc_type(
    w: &mut Buffer,
    it: &clean::Item,
    generics: &clean::Generics,
    bounds: &[clean::GenericBound],
    default: Option<&clean::Type>,
    link: AssocItemLink<'_>,
    indent: usize,
    cx: &Context<'_>,
) {
    write!(
        w,
        "{indent}type <a{href} class=\"associatedtype\">{name}</a>{generics}",
        indent = " ".repeat(indent),
        href = assoc_href_attr(it, link, cx),
        name = it.name.as_ref().unwrap(),
        generics = generics.print(cx),
    );
    if !bounds.is_empty() {
        write!(w, ": {}", print_generic_bounds(bounds, cx))
    }
    write!(w, "{}", print_where_clause(generics, cx, indent, Ending::NoNewline));
    if let Some(default) = default {
        write!(w, " = {}", default.print(cx))
    }
}

fn assoc_method(
    w: &mut Buffer,
    meth: &clean::Item,
    g: &clean::Generics,
    d: &clean::FnDecl,
    link: AssocItemLink<'_>,
    parent: ItemType,
    cx: &mut Context<'_>,
    render_mode: RenderMode,
) {
    let tcx = cx.tcx();
    let header = meth.fn_header(tcx).expect("Trying to get header from a non-function item");
    let name = meth.name.as_ref().unwrap();
    let vis = visibility_print_with_space(meth.visibility(tcx), meth.item_id, cx).to_string();
    // FIXME: Once https://github.com/rust-lang/rust/issues/67792 is implemented, we can remove
    // this condition.
    let constness = match render_mode {
        RenderMode::Normal => {
            print_constness_with_space(&header.constness, meth.const_stability(tcx))
        }
        RenderMode::ForDeref { .. } => "",
    };
    let asyncness = header.asyncness.print_with_space();
    let unsafety = header.unsafety.print_with_space();
    let defaultness = print_default_space(meth.is_default());
    let abi = print_abi_with_space(header.abi).to_string();
    let href = assoc_href_attr(meth, link, cx);

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

    let notable_traits = d.output.as_return().and_then(|output| notable_traits_button(output, cx));

    let (indent, indent_str, end_newline) = if parent == ItemType::Trait {
        header_len += 4;
        let indent_str = "    ";
        render_attributes_in_pre(w, meth, indent_str);
        (4, indent_str, Ending::NoNewline)
    } else {
        render_attributes_in_code(w, meth);
        (0, "", Ending::Newline)
    };
    w.reserve(header_len + "<a href=\"\" class=\"fn\">{".len() + "</a>".len());
    write!(
        w,
        "{indent}{vis}{constness}{asyncness}{unsafety}{defaultness}{abi}fn <a{href} class=\"fn\">{name}</a>\
         {generics}{decl}{notable_traits}{where_clause}",
        indent = indent_str,
        vis = vis,
        constness = constness,
        asyncness = asyncness,
        unsafety = unsafety,
        defaultness = defaultness,
        abi = abi,
        href = href,
        name = name,
        generics = g.print(cx),
        decl = d.full_print(header_len, indent, cx),
        notable_traits = notable_traits.unwrap_or_default(),
        where_clause = print_where_clause(g, cx, indent, end_newline),
    );
}

/// Writes a span containing the versions at which an item became stable and/or const-stable. For
/// example, if the item became stable at 1.0.0, and const-stable at 1.45.0, this function would
/// write a span containing "1.0.0 (const: 1.45.0)".
///
/// Returns `true` if a stability annotation was rendered.
///
/// Stability and const-stability are considered separately. If the item is unstable, no version
/// will be written. If the item is const-unstable, "const: unstable" will be appended to the
/// span, with a link to the tracking issue if present. If an item's stability or const-stability
/// version matches the version of its enclosing item, that version will be omitted.
///
/// Note that it is possible for an unstable function to be const-stable. In that case, the span
/// will include the const-stable version, but no stable version will be emitted, as a natural
/// consequence of the above rules.
fn render_stability_since_raw_with_extra(
    w: &mut Buffer,
    ver: Option<Symbol>,
    const_stability: Option<ConstStability>,
    containing_ver: Option<Symbol>,
    containing_const_ver: Option<Symbol>,
    extra_class: &str,
) -> bool {
    let stable_version = ver.filter(|inner| !inner.is_empty() && Some(*inner) != containing_ver);

    let mut title = String::new();
    let mut stability = String::new();

    if let Some(ver) = stable_version {
        stability.push_str(ver.as_str());
        title.push_str(&format!("Stable since Rust version {}", ver));
    }

    let const_title_and_stability = match const_stability {
        Some(ConstStability { level: StabilityLevel::Stable { since, .. }, .. })
            if Some(since) != containing_const_ver =>
        {
            Some((format!("const since {}", since), format!("const: {}", since)))
        }
        Some(ConstStability { level: StabilityLevel::Unstable { issue, .. }, feature, .. }) => {
            let unstable = if let Some(n) = issue {
                format!(
                    r#"<a href="https://github.com/rust-lang/rust/issues/{}" title="Tracking issue for {}">unstable</a>"#,
                    n, feature
                )
            } else {
                String::from("unstable")
            };

            Some((String::from("const unstable"), format!("const: {}", unstable)))
        }
        _ => None,
    };

    if let Some((const_title, const_stability)) = const_title_and_stability {
        if !title.is_empty() {
            title.push_str(&format!(", {}", const_title));
        } else {
            title.push_str(&const_title);
        }

        if !stability.is_empty() {
            stability.push_str(&format!(" ({})", const_stability));
        } else {
            stability.push_str(&const_stability);
        }
    }

    if !stability.is_empty() {
        write!(w, r#"<span class="since{extra_class}" title="{title}">{stability}</span>"#);
    }

    !stability.is_empty()
}

#[inline]
fn render_stability_since_raw(
    w: &mut Buffer,
    ver: Option<Symbol>,
    const_stability: Option<ConstStability>,
    containing_ver: Option<Symbol>,
    containing_const_ver: Option<Symbol>,
) -> bool {
    render_stability_since_raw_with_extra(
        w,
        ver,
        const_stability,
        containing_ver,
        containing_const_ver,
        "",
    )
}

fn render_assoc_item(
    w: &mut Buffer,
    item: &clean::Item,
    link: AssocItemLink<'_>,
    parent: ItemType,
    cx: &mut Context<'_>,
    render_mode: RenderMode,
) {
    match &*item.kind {
        clean::StrippedItem(..) => {}
        clean::TyMethodItem(m) => {
            assoc_method(w, item, &m.generics, &m.decl, link, parent, cx, render_mode)
        }
        clean::MethodItem(m, _) => {
            assoc_method(w, item, &m.generics, &m.decl, link, parent, cx, render_mode)
        }
        kind @ (clean::TyAssocConstItem(ty) | clean::AssocConstItem(ty, _)) => assoc_const(
            w,
            item,
            ty,
            match kind {
                clean::TyAssocConstItem(_) => None,
                clean::AssocConstItem(_, default) => Some(default),
                _ => unreachable!(),
            },
            link,
            if parent == ItemType::Trait { "    " } else { "" },
            cx,
        ),
        clean::TyAssocTypeItem(ref generics, ref bounds) => assoc_type(
            w,
            item,
            generics,
            bounds,
            None,
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        ),
        clean::AssocTypeItem(ref ty, ref bounds) => assoc_type(
            w,
            item,
            &ty.generics,
            bounds,
            Some(ty.item_type.as_ref().unwrap_or(&ty.type_)),
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
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
                Some(
                    pprust::attribute_to_string(attr)
                        .replace("\\\n", "")
                        .replace('\n', "")
                        .replace("  ", " "),
                )
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
            AssocItemLink::Anchor(_) => AssocItemLink::Anchor(Some(id)),
            ref other => *other,
        }
    }
}

fn write_impl_section_heading(w: &mut Buffer, title: &str, id: &str) {
    write!(
        w,
        "<h2 id=\"{id}\" class=\"small-section-header\">\
            {title}\
            <a href=\"#{id}\" class=\"anchor\">§</a>\
         </h2>"
    );
}

pub(crate) fn render_all_impls(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    containing_item: &clean::Item,
    concrete: &[&Impl],
    synthetic: &[&Impl],
    blanket_impl: &[&Impl],
) {
    let mut impls = Buffer::empty_from(w);
    render_impls(cx, &mut impls, concrete, containing_item, true);
    let impls = impls.into_inner();
    if !impls.is_empty() {
        write_impl_section_heading(w, "Trait Implementations", "trait-implementations");
        write!(w, "<div id=\"trait-implementations-list\">{}</div>", impls);
    }

    if !synthetic.is_empty() {
        write_impl_section_heading(w, "Auto Trait Implementations", "synthetic-implementations");
        w.write_str("<div id=\"synthetic-implementations-list\">");
        render_impls(cx, w, synthetic, containing_item, false);
        w.write_str("</div>");
    }

    if !blanket_impl.is_empty() {
        write_impl_section_heading(w, "Blanket Implementations", "blanket-implementations");
        w.write_str("<div id=\"blanket-implementations-list\">");
        render_impls(cx, w, blanket_impl, containing_item, false);
        w.write_str("</div>");
    }
}

fn render_assoc_items(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    containing_item: &clean::Item,
    it: DefId,
    what: AssocItemRender<'_>,
) {
    let mut derefs = DefIdSet::default();
    derefs.insert(it);
    render_assoc_items_inner(w, cx, containing_item, it, what, &mut derefs)
}

fn render_assoc_items_inner(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    containing_item: &clean::Item,
    it: DefId,
    what: AssocItemRender<'_>,
    derefs: &mut DefIdSet,
) {
    info!("Documenting associated items of {:?}", containing_item.name);
    let shared = Rc::clone(&cx.shared);
    let cache = &shared.cache;
    let Some(v) = cache.impls.get(&it) else { return };
    let (non_trait, traits): (Vec<_>, _) = v.iter().partition(|i| i.inner_impl().trait_.is_none());
    if !non_trait.is_empty() {
        let mut tmp_buf = Buffer::empty_from(w);
        let (render_mode, id) = match what {
            AssocItemRender::All => {
                write_impl_section_heading(&mut tmp_buf, "Implementations", "implementations");
                (RenderMode::Normal, "implementations-list".to_owned())
            }
            AssocItemRender::DerefFor { trait_, type_, deref_mut_ } => {
                let id =
                    cx.derive_id(small_url_encode(format!("deref-methods-{:#}", type_.print(cx))));
                if let Some(def_id) = type_.def_id(cx.cache()) {
                    cx.deref_id_map.insert(def_id, id.clone());
                }
                write_impl_section_heading(
                    &mut tmp_buf,
                    &format!(
                        "<span>Methods from {trait_}&lt;Target = {type_}&gt;</span>",
                        trait_ = trait_.print(cx),
                        type_ = type_.print(cx),
                    ),
                    &id,
                );
                (RenderMode::ForDeref { mut_: deref_mut_ }, cx.derive_id(id))
            }
        };
        let mut impls_buf = Buffer::empty_from(w);
        for i in &non_trait {
            render_impl(
                &mut impls_buf,
                cx,
                i,
                containing_item,
                AssocItemLink::Anchor(None),
                render_mode,
                None,
                &[],
                ImplRenderingParameters {
                    show_def_docs: true,
                    show_default_items: true,
                    show_non_assoc_items: true,
                    toggle_open_by_default: true,
                },
            );
        }
        if !impls_buf.is_empty() {
            w.push_buffer(tmp_buf);
            write!(w, "<div id=\"{}\">", id);
            w.push_buffer(impls_buf);
            w.write_str("</div>");
        }
    }

    if !traits.is_empty() {
        let deref_impl =
            traits.iter().find(|t| t.trait_did() == cx.tcx().lang_items().deref_trait());
        if let Some(impl_) = deref_impl {
            let has_deref_mut =
                traits.iter().any(|t| t.trait_did() == cx.tcx().lang_items().deref_mut_trait());
            render_deref_methods(w, cx, impl_, containing_item, has_deref_mut, derefs);
        }

        // If we were already one level into rendering deref methods, we don't want to render
        // anything after recursing into any further deref methods above.
        if let AssocItemRender::DerefFor { .. } = what {
            return;
        }

        let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
            traits.into_iter().partition(|t| t.inner_impl().kind.is_auto());
        let (blanket_impl, concrete): (Vec<&Impl>, _) =
            concrete.into_iter().partition(|t| t.inner_impl().kind.is_blanket());

        render_all_impls(w, cx, containing_item, &concrete, &synthetic, &blanket_impl);
    }
}

fn render_deref_methods(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    impl_: &Impl,
    container_item: &clean::Item,
    deref_mut: bool,
    derefs: &mut DefIdSet,
) {
    let cache = cx.cache();
    let deref_type = impl_.inner_impl().trait_.as_ref().unwrap();
    let (target, real_target) = impl_
        .inner_impl()
        .items
        .iter()
        .find_map(|item| match *item.kind {
            clean::AssocTypeItem(box ref t, _) => Some(match *t {
                clean::Typedef { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
        .expect("Expected associated type binding");
    debug!("Render deref methods for {:#?}, target {:#?}", impl_.inner_impl().for_, target);
    let what =
        AssocItemRender::DerefFor { trait_: deref_type, type_: real_target, deref_mut_: deref_mut };
    if let Some(did) = target.def_id(cache) {
        if let Some(type_did) = impl_.inner_impl().for_.def_id(cache) {
            // `impl Deref<Target = S> for S`
            if did == type_did || !derefs.insert(did) {
                // Avoid infinite cycles
                return;
            }
        }
        render_assoc_items_inner(w, cx, container_item, did, what, derefs);
    } else if let Some(prim) = target.primitive_type() {
        if let Some(&did) = cache.primitive_locations.get(&prim) {
            render_assoc_items_inner(w, cx, container_item, did, what, derefs);
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
            SelfTy::SelfExplicit(clean::Type::Path { path }) => {
                (false, Some(path.def_id()) == tcx.lang_items().owned_box(), false)
            }
            SelfTy::SelfValue => (false, false, true),
            _ => (false, false, false),
        };

        (deref_mut_ || !by_mut_ref) && !by_box && !by_value
    } else {
        false
    }
}

pub(crate) fn notable_traits_button(ty: &clean::Type, cx: &mut Context<'_>) -> Option<String> {
    let mut has_notable_trait = false;

    let did = ty.def_id(cx.cache())?;

    // Box has pass-through impls for Read, Write, Iterator, and Future when the
    // boxed type implements one of those. We don't want to treat every Box return
    // as being notably an Iterator (etc), though, so we exempt it. Pin has the same
    // issue, with a pass-through impl for Future.
    if Some(did) == cx.tcx().lang_items().owned_box()
        || Some(did) == cx.tcx().lang_items().pin_type()
    {
        return None;
    }

    if let Some(impls) = cx.cache().impls.get(&did) {
        for i in impls {
            let impl_ = i.inner_impl();
            if !impl_.for_.without_borrowed_ref().is_same(ty.without_borrowed_ref(), cx.cache()) {
                // Two different types might have the same did,
                // without actually being the same.
                continue;
            }
            if let Some(trait_) = &impl_.trait_ {
                let trait_did = trait_.def_id();

                if cx.cache().traits.get(&trait_did).map_or(false, |t| t.is_notable_trait(cx.tcx()))
                {
                    has_notable_trait = true;
                }
            }
        }
    }

    if has_notable_trait {
        cx.types_with_notable_traits.insert(ty.clone());
        Some(format!(
            " <a href=\"#\" class=\"notable-traits\" data-ty=\"{ty}\">ⓘ</a>",
            ty = Escape(&format!("{:#}", ty.print(cx))),
        ))
    } else {
        None
    }
}

fn notable_traits_decl(ty: &clean::Type, cx: &Context<'_>) -> (String, String) {
    let mut out = Buffer::html();

    let did = ty.def_id(cx.cache()).expect("notable_traits_button already checked this");

    let impls = cx.cache().impls.get(&did).expect("notable_traits_button already checked this");

    for i in impls {
        let impl_ = i.inner_impl();
        if !impl_.for_.without_borrowed_ref().is_same(ty.without_borrowed_ref(), cx.cache()) {
            // Two different types might have the same did,
            // without actually being the same.
            continue;
        }
        if let Some(trait_) = &impl_.trait_ {
            let trait_did = trait_.def_id();

            if cx.cache().traits.get(&trait_did).map_or(false, |t| t.is_notable_trait(cx.tcx())) {
                if out.is_empty() {
                    write!(
                        &mut out,
                        "<h3>Notable traits for <code>{}</code></h3>\
                     <pre><code>",
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
                    if let clean::AssocTypeItem(ref tydef, ref _bounds) = *it.kind {
                        out.push_str("<span class=\"where fmt-newline\">    ");
                        let empty_set = FxHashSet::default();
                        let src_link = AssocItemLink::GotoSource(trait_did.into(), &empty_set);
                        assoc_type(
                            &mut out,
                            it,
                            &tydef.generics,
                            &[], // intentionally leaving out bounds
                            Some(&tydef.type_),
                            src_link,
                            0,
                            cx,
                        );
                        out.push_str(";</span>");
                    }
                }
            }
        }
    }
    if out.is_empty() {
        write!(&mut out, "</code></pre>",);
    }

    (format!("{:#}", ty.print(cx)), out.into_inner())
}

pub(crate) fn notable_traits_json<'a>(
    tys: impl Iterator<Item = &'a clean::Type>,
    cx: &Context<'_>,
) -> String {
    let mut mp: Vec<(String, String)> = tys.map(|ty| notable_traits_decl(ty, cx)).collect();
    mp.sort_by(|(name1, _html1), (name2, _html2)| name1.cmp(name2));
    struct NotableTraitsMap(Vec<(String, String)>);
    impl Serialize for NotableTraitsMap {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut map = serializer.serialize_map(Some(self.0.len()))?;
            for item in &self.0 {
                map.serialize_entry(&item.0, &item.1)?;
            }
            map.end()
        }
    }
    serde_json::to_string(&NotableTraitsMap(mp))
        .expect("serialize (string, string) -> json object cannot fail")
}

#[derive(Clone, Copy, Debug)]
struct ImplRenderingParameters {
    show_def_docs: bool,
    show_default_items: bool,
    /// Whether or not to show methods.
    show_non_assoc_items: bool,
    toggle_open_by_default: bool,
}

fn render_impl(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    link: AssocItemLink<'_>,
    render_mode: RenderMode,
    use_absolute: Option<bool>,
    aliases: &[String],
    rendering_params: ImplRenderingParameters,
) {
    let shared = Rc::clone(&cx.shared);
    let cache = &shared.cache;
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
        cx: &mut Context<'_>,
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
                    should_render_item(item, deref_mut_, cx.tcx())
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
            let method_toggle_class = if item_type.is_method() { " method-toggle" } else { "" };
            write!(w, "<details class=\"toggle{}\" open><summary>", method_toggle_class);
        }
        match &*item.kind {
            clean::MethodItem(..) | clean::TyMethodItem(_) => {
                // Only render when the method is not static or we allow static methods
                if render_method_item {
                    let id = cx.derive_id(format!("{}.{}", item_type, name));
                    let source_id = trait_
                        .and_then(|trait_| {
                            trait_.items.iter().find(|item| {
                                item.name.map(|n| n.as_str().eq(name.as_str())).unwrap_or(false)
                            })
                        })
                        .map(|item| format!("{}.{}", item.type_(), name));
                    write!(w, "<section id=\"{}\" class=\"{}{}\">", id, item_type, in_trait_class,);
                    render_rightside(w, cx, item, containing_item, render_mode);
                    if trait_.is_some() {
                        // Anchors are only used on trait impls.
                        write!(w, "<a href=\"#{}\" class=\"anchor\">§</a>", id);
                    }
                    w.write_str("<h4 class=\"code-header\">");
                    render_assoc_item(
                        w,
                        item,
                        link.anchor(source_id.as_ref().unwrap_or(&id)),
                        ItemType::Impl,
                        cx,
                        render_mode,
                    );
                    w.write_str("</h4>");
                    w.write_str("</section>");
                }
            }
            kind @ (clean::TyAssocConstItem(ty) | clean::AssocConstItem(ty, _)) => {
                let source_id = format!("{}.{}", item_type, name);
                let id = cx.derive_id(source_id.clone());
                write!(w, "<section id=\"{}\" class=\"{}{}\">", id, item_type, in_trait_class);
                render_rightside(w, cx, item, containing_item, render_mode);
                if trait_.is_some() {
                    // Anchors are only used on trait impls.
                    write!(w, "<a href=\"#{}\" class=\"anchor\">§</a>", id);
                }
                w.write_str("<h4 class=\"code-header\">");
                assoc_const(
                    w,
                    item,
                    ty,
                    match kind {
                        clean::TyAssocConstItem(_) => None,
                        clean::AssocConstItem(_, default) => Some(default),
                        _ => unreachable!(),
                    },
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    "",
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</section>");
            }
            clean::TyAssocTypeItem(generics, bounds) => {
                let source_id = format!("{}.{}", item_type, name);
                let id = cx.derive_id(source_id.clone());
                write!(w, "<section id=\"{}\" class=\"{}{}\">", id, item_type, in_trait_class);
                if trait_.is_some() {
                    // Anchors are only used on trait impls.
                    write!(w, "<a href=\"#{}\" class=\"anchor\">§</a>", id);
                }
                w.write_str("<h4 class=\"code-header\">");
                assoc_type(
                    w,
                    item,
                    generics,
                    bounds,
                    None,
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    0,
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</section>");
            }
            clean::AssocTypeItem(tydef, _bounds) => {
                let source_id = format!("{}.{}", item_type, name);
                let id = cx.derive_id(source_id.clone());
                write!(w, "<section id=\"{}\" class=\"{}{}\">", id, item_type, in_trait_class);
                if trait_.is_some() {
                    // Anchors are only used on trait impls.
                    write!(w, "<a href=\"#{}\" class=\"anchor\">§</a>", id);
                }
                w.write_str("<h4 class=\"code-header\">");
                assoc_type(
                    w,
                    item,
                    &tydef.generics,
                    &[], // intentionally leaving out bounds
                    Some(tydef.item_type.as_ref().unwrap_or(&tydef.type_)),
                    link.anchor(if trait_.is_some() { &source_id } else { &id }),
                    0,
                    cx,
                );
                w.write_str("</h4>");
                w.write_str("</section>");
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
            trait_,
            rendering_params,
        );
    }

    fn render_default_items(
        boring: &mut Buffer,
        interesting: &mut Buffer,
        cx: &mut Context<'_>,
        t: &clean::Trait,
        i: &clean::Impl,
        parent: &clean::Item,
        containing_item: &clean::Item,
        render_mode: RenderMode,
        rendering_params: ImplRenderingParameters,
    ) {
        for trait_item in &t.items {
            // Skip over any default trait items that are impossible to call
            // (e.g. if it has a `Self: Sized` bound on an unsized type).
            if let Some(impl_def_id) = parent.item_id.as_def_id()
                && let Some(trait_item_def_id) = trait_item.item_id.as_def_id()
                && cx.tcx().is_impossible_method((impl_def_id, trait_item_def_id))
            {
                continue;
            }

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
                t,
                i.inner_impl(),
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
                "<details class=\"toggle implementors-toggle\"{}>",
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
            aliases,
        );
        if toggled {
            write!(w, "</summary>")
        }

        if let Some(ref dox) = i.impl_item.collapsed_doc_value() {
            if trait_.is_none() && i.inner_impl().items.is_empty() {
                w.write_str(
                    "<div class=\"item-info\">\
                    <div class=\"stab empty-impl\">This impl block contains no items.</div>
                </div>",
                );
            }
            write!(
                w,
                "<div class=\"docblock\">{}</div>",
                Markdown {
                    content: &*dox,
                    links: &i.impl_item.links(cx),
                    ids: &mut cx.id_map,
                    error_codes: cx.shared.codes,
                    edition: cx.shared.edition(),
                    playground: &cx.shared.playground,
                    heading_offset: HeadingOffset::H4
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
// associated types. For example "1.0.0 (const: 1.39.0) · source".
fn render_rightside(
    w: &mut Buffer,
    cx: &Context<'_>,
    item: &clean::Item,
    containing_item: &clean::Item,
    render_mode: RenderMode,
) {
    let tcx = cx.tcx();

    // FIXME: Once https://github.com/rust-lang/rust/issues/67792 is implemented, we can remove
    // this condition.
    let (const_stability, const_stable_since) = match render_mode {
        RenderMode::Normal => (item.const_stability(tcx), containing_item.const_stable_since(tcx)),
        RenderMode::ForDeref { .. } => (None, None),
    };
    let src_href = cx.src_href(item);
    let has_src_ref = src_href.is_some();

    let mut rightside = Buffer::new();
    let has_stability = render_stability_since_raw_with_extra(
        &mut rightside,
        item.stable_since(tcx),
        const_stability,
        containing_item.stable_since(tcx),
        const_stable_since,
        if has_src_ref { "" } else { " rightside" },
    );
    if let Some(l) = src_href {
        if has_stability {
            write!(rightside, " · <a class=\"srclink\" href=\"{}\">source</a>", l)
        } else {
            write!(rightside, "<a class=\"srclink rightside\" href=\"{}\">source</a>", l)
        }
    }
    if has_stability && has_src_ref {
        write!(w, "<span class=\"rightside\">{}</span>", rightside.into_inner());
    } else {
        w.push_buffer(rightside);
    }
}

pub(crate) fn render_impl_summary(
    w: &mut Buffer,
    cx: &mut Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    containing_item: &clean::Item,
    show_def_docs: bool,
    use_absolute: Option<bool>,
    // This argument is used to reference same type with different paths to avoid duplication
    // in documentation pages for trait with automatic implementations like "Send" and "Sync".
    aliases: &[String],
) {
    let inner_impl = i.inner_impl();
    let id = cx.derive_id(get_id_for_impl(&inner_impl.for_, inner_impl.trait_.as_ref(), cx));
    let aliases = if aliases.is_empty() {
        String::new()
    } else {
        format!(" data-aliases=\"{}\"", aliases.join(","))
    };
    write!(w, "<section id=\"{}\" class=\"impl\"{}>", id, aliases);
    render_rightside(w, cx, &i.impl_item, containing_item, RenderMode::Normal);
    write!(w, "<a href=\"#{}\" class=\"anchor\">§</a>", id);
    write!(w, "<h3 class=\"code-header\">");

    if let Some(use_absolute) = use_absolute {
        write!(w, "{}", inner_impl.print(use_absolute, cx));
        if show_def_docs {
            for it in &inner_impl.items {
                if let clean::AssocTypeItem(ref tydef, ref _bounds) = *it.kind {
                    w.write_str("<span class=\"where fmt-newline\">  ");
                    assoc_type(
                        w,
                        it,
                        &tydef.generics,
                        &[], // intentionally leaving out bounds
                        Some(&tydef.type_),
                        AssocItemLink::Anchor(None),
                        0,
                        cx,
                    );
                    w.write_str(";</span>");
                }
            }
        }
    } else {
        write!(w, "{}", inner_impl.print(false, cx));
    }
    write!(w, "</h3>");

    let is_trait = inner_impl.trait_.is_some();
    if is_trait {
        if let Some(portability) = portability(&i.impl_item, Some(parent)) {
            write!(w, "<span class=\"item-info\">{}</span>", portability);
        }
    }

    w.write_str("</section>");
}

fn print_sidebar(cx: &Context<'_>, it: &clean::Item, buffer: &mut Buffer) {
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
            "<h2 class=\"location\"><a href=\"#\">{}{}</a></h2>",
            match *it.kind {
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

    buffer.write_str("<div class=\"sidebar-elems\">");
    if it.is_crate() {
        write!(buffer, "<ul class=\"block\">");
        if let Some(ref version) = cx.cache().crate_version {
            write!(buffer, "<li class=\"version\">Version {}</li>", Escape(version));
        }
        write!(buffer, "<li><a id=\"all-types\" href=\"all.html\">All Items</a></li>");
        buffer.write_str("</ul>");
    }

    match *it.kind {
        clean::StructItem(ref s) => sidebar_struct(cx, buffer, it, s),
        clean::TraitItem(ref t) => sidebar_trait(cx, buffer, it, t),
        clean::PrimitiveItem(_) => sidebar_primitive(cx, buffer, it),
        clean::UnionItem(ref u) => sidebar_union(cx, buffer, it, u),
        clean::EnumItem(ref e) => sidebar_enum(cx, buffer, it, e),
        clean::TypedefItem(_) => sidebar_typedef(cx, buffer, it),
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
        let path: String = cx.current.iter().map(|s| s.as_str()).intersperse("::").collect();

        write!(buffer, "<h2><a href=\"index.html\">In {}</a></h2>", path);
    }

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
                        url: get_next_url(used_links, format!("{}.{}", ItemType::Method, name)),
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
                url: get_next_url(used_links, format!("{}.{}", ItemType::AssocConst, name)),
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

pub(crate) fn sidebar_render_assoc_items(
    cx: &Context<'_>,
    out: &mut Buffer,
    id_map: &mut IdMap,
    concrete: Vec<&Impl>,
    synthetic: Vec<&Impl>,
    blanket_impl: Vec<&Impl>,
) {
    let format_impls = |impls: Vec<&Impl>, id_map: &mut IdMap| {
        let mut links = FxHashSet::default();

        let mut ret = impls
            .iter()
            .filter_map(|it| {
                let trait_ = it.inner_impl().trait_.as_ref()?;
                let encoded =
                    id_map.derive(get_id_for_impl(&it.inner_impl().for_, Some(trait_), cx));

                let i_display = format!("{:#}", trait_.print(cx));
                let out = Escape(&i_display);
                let prefix = match it.inner_impl().polarity {
                    ty::ImplPolarity::Positive | ty::ImplPolarity::Reservation => "",
                    ty::ImplPolarity::Negative => "!",
                };
                let generated = format!("<a href=\"#{}\">{}{}</a>", encoded, prefix, out);
                if links.insert(generated.clone()) { Some(generated) } else { None }
            })
            .collect::<Vec<String>>();
        ret.sort();
        ret
    };

    let concrete_format = format_impls(concrete, id_map);
    let synthetic_format = format_impls(synthetic, id_map);
    let blanket_format = format_impls(blanket_impl, id_map);

    if !concrete_format.is_empty() {
        print_sidebar_block(
            out,
            "trait-implementations",
            "Trait Implementations",
            concrete_format.iter(),
        );
    }

    if !synthetic_format.is_empty() {
        print_sidebar_block(
            out,
            "synthetic-implementations",
            "Auto Trait Implementations",
            synthetic_format.iter(),
        );
    }

    if !blanket_format.is_empty() {
        print_sidebar_block(
            out,
            "blanket-implementations",
            "Blanket Implementations",
            blanket_format.iter(),
        );
    }
}

fn sidebar_assoc_items(cx: &Context<'_>, out: &mut Buffer, it: &clean::Item) {
    let did = it.item_id.expect_def_id();
    let cache = cx.cache();

    if let Some(v) = cache.impls.get(&did) {
        let mut used_links = FxHashSet::default();
        let mut id_map = IdMap::new();

        {
            let used_links_bor = &mut used_links;
            let mut assoc_consts = v
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| get_associated_constants(i.inner_impl(), used_links_bor))
                .collect::<Vec<_>>();
            if !assoc_consts.is_empty() {
                // We want links' order to be reproducible so we don't use unstable sort.
                assoc_consts.sort();

                print_sidebar_block(
                    out,
                    "implementations",
                    "Associated Constants",
                    assoc_consts.iter(),
                );
            }
            let mut methods = v
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| get_methods(i.inner_impl(), false, used_links_bor, false, cx.tcx()))
                .collect::<Vec<_>>();
            if !methods.is_empty() {
                // We want links' order to be reproducible so we don't use unstable sort.
                methods.sort();

                print_sidebar_block(out, "implementations", "Methods", methods.iter());
            }
        }

        if v.iter().any(|i| i.inner_impl().trait_.is_some()) {
            if let Some(impl_) =
                v.iter().find(|i| i.trait_did() == cx.tcx().lang_items().deref_trait())
            {
                let mut derefs = DefIdSet::default();
                derefs.insert(did);
                sidebar_deref_methods(cx, out, impl_, v, &mut derefs, &mut used_links);
            }

            let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
                v.iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_auto());
            let (blanket_impl, concrete): (Vec<&Impl>, Vec<&Impl>) =
                concrete.into_iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_blanket());

            sidebar_render_assoc_items(cx, out, &mut id_map, concrete, synthetic, blanket_impl);
        }
    }
}

fn sidebar_deref_methods(
    cx: &Context<'_>,
    out: &mut Buffer,
    impl_: &Impl,
    v: &[Impl],
    derefs: &mut DefIdSet,
    used_links: &mut FxHashSet<String>,
) {
    let c = cx.cache();

    debug!("found Deref: {:?}", impl_);
    if let Some((target, real_target)) =
        impl_.inner_impl().items.iter().find_map(|item| match *item.kind {
            clean::AssocTypeItem(box ref t, _) => Some(match *t {
                clean::Typedef { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
    {
        debug!("found target, real_target: {:?} {:?}", target, real_target);
        if let Some(did) = target.def_id(c) {
            if let Some(type_did) = impl_.inner_impl().for_.def_id(c) {
                // `impl Deref<Target = S> for S`
                if did == type_did || !derefs.insert(did) {
                    // Avoid infinite cycles
                    return;
                }
            }
        }
        let deref_mut = v.iter().any(|i| i.trait_did() == cx.tcx().lang_items().deref_mut_trait());
        let inner_impl = target
            .def_id(c)
            .or_else(|| {
                target.primitive_type().and_then(|prim| c.primitive_locations.get(&prim).cloned())
            })
            .and_then(|did| c.impls.get(&did));
        if let Some(impls) = inner_impl {
            debug!("found inner_impl: {:?}", impls);
            let mut ret = impls
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| get_methods(i.inner_impl(), true, used_links, deref_mut, cx.tcx()))
                .collect::<Vec<_>>();
            if !ret.is_empty() {
                let id = if let Some(target_def_id) = real_target.def_id(c) {
                    cx.deref_id_map.get(&target_def_id).expect("Deref section without derived id")
                } else {
                    "deref-methods"
                };
                let title = format!(
                    "Methods from {}&lt;Target={}&gt;",
                    Escape(&format!("{:#}", impl_.inner_impl().trait_.as_ref().unwrap().print(cx))),
                    Escape(&format!("{:#}", real_target.print(cx))),
                );
                // We want links' order to be reproducible so we don't use unstable sort.
                ret.sort();
                print_sidebar_block(out, id, &title, ret.iter());
            }
        }

        // Recurse into any further impls that might exist for `target`
        if let Some(target_did) = target.def_id(c) {
            if let Some(target_impls) = c.impls.get(&target_did) {
                if let Some(target_deref_impl) = target_impls.iter().find(|i| {
                    i.inner_impl()
                        .trait_
                        .as_ref()
                        .map(|t| Some(t.def_id()) == cx.tcx().lang_items().deref_trait())
                        .unwrap_or(false)
                }) {
                    sidebar_deref_methods(
                        cx,
                        out,
                        target_deref_impl,
                        target_impls,
                        derefs,
                        used_links,
                    );
                }
            }
        }
    }
}

fn sidebar_struct(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, s: &clean::Struct) {
    let mut sidebar = Buffer::new();
    let fields = get_struct_fields_name(&s.fields);

    if !fields.is_empty() {
        match s.ctor_kind {
            None => {
                print_sidebar_block(&mut sidebar, "fields", "Fields", fields.iter());
            }
            Some(CtorKind::Fn) => print_sidebar_title(&mut sidebar, "fields", "Tuple Fields"),
            Some(CtorKind::Const) => {}
        }
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
    }
}

fn get_id_for_impl(for_: &clean::Type, trait_: Option<&clean::Path>, cx: &Context<'_>) -> String {
    match trait_ {
        Some(t) => small_url_encode(format!("impl-{:#}-for-{:#}", t.print(cx), for_.print(cx))),
        None => small_url_encode(format!("impl-{:#}", for_.print(cx))),
    }
}

fn extract_for_impl_name(item: &clean::Item, cx: &Context<'_>) -> Option<(String, String)> {
    match *item.kind {
        clean::ItemKind::ImplItem(ref i) => {
            i.trait_.as_ref().map(|trait_| {
                // Alternative format produces no URLs,
                // so this parameter does nothing.
                (format!("{:#}", i.for_.print(cx)), get_id_for_impl(&i.for_, Some(trait_), cx))
            })
        }
        _ => None,
    }
}

fn print_sidebar_title(buf: &mut Buffer, id: &str, title: &str) {
    write!(buf, "<h3><a href=\"#{}\">{}</a></h3>", id, title);
}

fn print_sidebar_block(
    buf: &mut Buffer,
    id: &str,
    title: &str,
    items: impl Iterator<Item = impl fmt::Display>,
) {
    print_sidebar_title(buf, id, title);
    buf.push_str("<ul class=\"block\">");
    for item in items {
        write!(buf, "<li>{}</li>", item);
    }
    buf.push_str("</ul>");
}

fn sidebar_trait(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, t: &clean::Trait) {
    buf.write_str("<section>");

    fn print_sidebar_section(
        out: &mut Buffer,
        items: &[clean::Item],
        id: &str,
        title: &str,
        filter: impl Fn(&clean::Item) -> bool,
        mapper: impl Fn(&str) -> String,
    ) {
        let mut items: Vec<&str> = items
            .iter()
            .filter_map(|m| match m.name {
                Some(ref name) if filter(m) => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if !items.is_empty() {
            items.sort_unstable();
            print_sidebar_block(out, id, title, items.into_iter().map(mapper));
        }
    }

    print_sidebar_section(
        buf,
        &t.items,
        "required-associated-types",
        "Required Associated Types",
        |m| m.is_ty_associated_type(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::AssocType),
    );

    print_sidebar_section(
        buf,
        &t.items,
        "provided-associated-types",
        "Provided Associated Types",
        |m| m.is_associated_type(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::AssocType),
    );

    print_sidebar_section(
        buf,
        &t.items,
        "required-associated-consts",
        "Required Associated Constants",
        |m| m.is_ty_associated_const(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::AssocConst),
    );

    print_sidebar_section(
        buf,
        &t.items,
        "provided-associated-consts",
        "Provided Associated Constants",
        |m| m.is_associated_const(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::AssocConst),
    );

    print_sidebar_section(
        buf,
        &t.items,
        "required-methods",
        "Required Methods",
        |m| m.is_ty_method(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::TyMethod),
    );

    print_sidebar_section(
        buf,
        &t.items,
        "provided-methods",
        "Provided Methods",
        |m| m.is_method(),
        |sym| format!("<a href=\"#{1}.{0}\">{0}</a>", sym, ItemType::Method),
    );

    if let Some(implementors) = cx.cache().implementors.get(&it.item_id.expect_def_id()) {
        let mut res = implementors
            .iter()
            .filter(|i| !i.is_on_local_type(cx))
            .filter_map(|i| extract_for_impl_name(&i.impl_item, cx))
            .collect::<Vec<_>>();

        if !res.is_empty() {
            res.sort();
            print_sidebar_block(
                buf,
                "foreign-impls",
                "Implementations on Foreign Types",
                res.iter().map(|(name, id)| format!("<a href=\"#{}\">{}</a>", id, Escape(name))),
            );
        }
    }

    sidebar_assoc_items(cx, buf, it);

    print_sidebar_title(buf, "implementors", "Implementors");
    if t.is_auto(cx.tcx()) {
        print_sidebar_title(buf, "synthetic-implementors", "Auto Implementors");
    }

    buf.push_str("</section>")
}

/// Returns the list of implementations for the primitive reference type, filtering out any
/// implementations that are on concrete or partially generic types, only keeping implementations
/// of the form `impl<T> Trait for &T`.
pub(crate) fn get_filtered_impls_for_reference<'a>(
    shared: &'a Rc<SharedContext<'_>>,
    it: &clean::Item,
) -> (Vec<&'a Impl>, Vec<&'a Impl>, Vec<&'a Impl>) {
    let def_id = it.item_id.expect_def_id();
    // If the reference primitive is somehow not defined, exit early.
    let Some(v) = shared.cache.impls.get(&def_id) else { return (Vec::new(), Vec::new(), Vec::new()) };
    // Since there is no "direct implementation" on the reference primitive type, we filter out
    // every implementation which isn't a trait implementation.
    let traits = v.iter().filter(|i| i.inner_impl().trait_.is_some());
    let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
        traits.partition(|t| t.inner_impl().kind.is_auto());

    let (blanket_impl, concrete): (Vec<&Impl>, _) =
        concrete.into_iter().partition(|t| t.inner_impl().kind.is_blanket());
    // Now we keep only references over full generic types.
    let concrete: Vec<_> = concrete
        .into_iter()
        .filter(|t| match t.inner_impl().for_ {
            clean::Type::BorrowedRef { ref type_, .. } => type_.is_full_generic(),
            _ => false,
        })
        .collect();

    (concrete, synthetic, blanket_impl)
}

fn sidebar_primitive(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();

    if it.name.map(|n| n.as_str() != "reference").unwrap_or(false) {
        sidebar_assoc_items(cx, &mut sidebar, it);
    } else {
        let shared = Rc::clone(&cx.shared);
        let (concrete, synthetic, blanket_impl) = get_filtered_impls_for_reference(&shared, it);

        sidebar_render_assoc_items(
            cx,
            &mut sidebar,
            &mut IdMap::new(),
            concrete,
            synthetic,
            blanket_impl,
        );
    }

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
    }
}

fn sidebar_typedef(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();
    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
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
        print_sidebar_block(&mut sidebar, "fields", "Fields", fields.iter());
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
    }
}

fn sidebar_enum(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item, e: &clean::Enum) {
    let mut sidebar = Buffer::new();

    let mut variants = e
        .variants()
        .filter_map(|v| {
            v.name
                .as_ref()
                .map(|name| format!("<a href=\"#variant.{name}\">{name}</a>", name = name))
        })
        .collect::<Vec<_>>();
    if !variants.is_empty() {
        variants.sort_unstable();
        print_sidebar_block(&mut sidebar, "variants", "Variants", variants.iter());
    }

    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ItemSection {
    Reexports,
    PrimitiveTypes,
    Modules,
    Macros,
    Structs,
    Enums,
    Constants,
    Statics,
    Traits,
    Functions,
    TypeDefinitions,
    Unions,
    Implementations,
    TypeMethods,
    Methods,
    StructFields,
    Variants,
    AssociatedTypes,
    AssociatedConstants,
    ForeignTypes,
    Keywords,
    OpaqueTypes,
    AttributeMacros,
    DeriveMacros,
    TraitAliases,
}

impl ItemSection {
    const ALL: &'static [Self] = {
        use ItemSection::*;
        // NOTE: The order here affects the order in the UI.
        &[
            Reexports,
            PrimitiveTypes,
            Modules,
            Macros,
            Structs,
            Enums,
            Constants,
            Statics,
            Traits,
            Functions,
            TypeDefinitions,
            Unions,
            Implementations,
            TypeMethods,
            Methods,
            StructFields,
            Variants,
            AssociatedTypes,
            AssociatedConstants,
            ForeignTypes,
            Keywords,
            OpaqueTypes,
            AttributeMacros,
            DeriveMacros,
            TraitAliases,
        ]
    };

    fn id(self) -> &'static str {
        match self {
            Self::Reexports => "reexports",
            Self::Modules => "modules",
            Self::Structs => "structs",
            Self::Unions => "unions",
            Self::Enums => "enums",
            Self::Functions => "functions",
            Self::TypeDefinitions => "types",
            Self::Statics => "statics",
            Self::Constants => "constants",
            Self::Traits => "traits",
            Self::Implementations => "impls",
            Self::TypeMethods => "tymethods",
            Self::Methods => "methods",
            Self::StructFields => "fields",
            Self::Variants => "variants",
            Self::Macros => "macros",
            Self::PrimitiveTypes => "primitives",
            Self::AssociatedTypes => "associated-types",
            Self::AssociatedConstants => "associated-consts",
            Self::ForeignTypes => "foreign-types",
            Self::Keywords => "keywords",
            Self::OpaqueTypes => "opaque-types",
            Self::AttributeMacros => "attributes",
            Self::DeriveMacros => "derives",
            Self::TraitAliases => "trait-aliases",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Reexports => "Re-exports",
            Self::Modules => "Modules",
            Self::Structs => "Structs",
            Self::Unions => "Unions",
            Self::Enums => "Enums",
            Self::Functions => "Functions",
            Self::TypeDefinitions => "Type Definitions",
            Self::Statics => "Statics",
            Self::Constants => "Constants",
            Self::Traits => "Traits",
            Self::Implementations => "Implementations",
            Self::TypeMethods => "Type Methods",
            Self::Methods => "Methods",
            Self::StructFields => "Struct Fields",
            Self::Variants => "Variants",
            Self::Macros => "Macros",
            Self::PrimitiveTypes => "Primitive Types",
            Self::AssociatedTypes => "Associated Types",
            Self::AssociatedConstants => "Associated Constants",
            Self::ForeignTypes => "Foreign Types",
            Self::Keywords => "Keywords",
            Self::OpaqueTypes => "Opaque Types",
            Self::AttributeMacros => "Attribute Macros",
            Self::DeriveMacros => "Derive Macros",
            Self::TraitAliases => "Trait Aliases",
        }
    }
}

fn item_ty_to_section(ty: ItemType) -> ItemSection {
    match ty {
        ItemType::ExternCrate | ItemType::Import => ItemSection::Reexports,
        ItemType::Module => ItemSection::Modules,
        ItemType::Struct => ItemSection::Structs,
        ItemType::Union => ItemSection::Unions,
        ItemType::Enum => ItemSection::Enums,
        ItemType::Function => ItemSection::Functions,
        ItemType::Typedef => ItemSection::TypeDefinitions,
        ItemType::Static => ItemSection::Statics,
        ItemType::Constant => ItemSection::Constants,
        ItemType::Trait => ItemSection::Traits,
        ItemType::Impl => ItemSection::Implementations,
        ItemType::TyMethod => ItemSection::TypeMethods,
        ItemType::Method => ItemSection::Methods,
        ItemType::StructField => ItemSection::StructFields,
        ItemType::Variant => ItemSection::Variants,
        ItemType::Macro => ItemSection::Macros,
        ItemType::Primitive => ItemSection::PrimitiveTypes,
        ItemType::AssocType => ItemSection::AssociatedTypes,
        ItemType::AssocConst => ItemSection::AssociatedConstants,
        ItemType::ForeignType => ItemSection::ForeignTypes,
        ItemType::Keyword => ItemSection::Keywords,
        ItemType::OpaqueTy => ItemSection::OpaqueTypes,
        ItemType::ProcAttribute => ItemSection::AttributeMacros,
        ItemType::ProcDerive => ItemSection::DeriveMacros,
        ItemType::TraitAlias => ItemSection::TraitAliases,
    }
}

pub(crate) fn sidebar_module_like(buf: &mut Buffer, item_sections_in_use: FxHashSet<ItemSection>) {
    use std::fmt::Write as _;

    let mut sidebar = String::new();

    for &sec in ItemSection::ALL.iter().filter(|sec| item_sections_in_use.contains(sec)) {
        let _ = write!(sidebar, "<li><a href=\"#{}\">{}</a></li>", sec.id(), sec.name());
    }

    if !sidebar.is_empty() {
        write!(
            buf,
            "<section>\
                 <ul class=\"block\">{}</ul>\
             </section>",
            sidebar
        );
    }
}

fn sidebar_module(buf: &mut Buffer, items: &[clean::Item]) {
    let item_sections_in_use: FxHashSet<_> = items
        .iter()
        .filter(|it| {
            !it.is_stripped()
                && it
                    .name
                    .or_else(|| {
                        if let clean::ImportItem(ref i) = *it.kind &&
                            let clean::ImportKind::Simple(s) = i.kind { Some(s) } else { None }
                    })
                    .is_some()
        })
        .map(|it| item_ty_to_section(it.type_()))
        .collect();

    sidebar_module_like(buf, item_sections_in_use);
}

fn sidebar_foreign_type(cx: &Context<'_>, buf: &mut Buffer, it: &clean::Item) {
    let mut sidebar = Buffer::new();
    sidebar_assoc_items(cx, &mut sidebar, it);

    if !sidebar.is_empty() {
        write!(buf, "<section>{}</section>", sidebar.into_inner());
    }
}

pub(crate) const BASIC_KEYWORDS: &str = "rust, rustlang, rust-lang";

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
        let get_extern = || cache.external_paths.get(&did).map(|s| &s.0);
        let fqp = cache.exact_paths.get(&did).or_else(get_extern);

        if let Some(path) = fqp {
            out.push(join_with_double_colon(&path));
        }
    };

    work.push_back(first_ty);

    while let Some(ty) = work.pop_front() {
        if !visited.insert(ty.clone()) {
            continue;
        }

        match ty {
            clean::Type::Path { path } => process_path(path.def_id()),
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
            clean::Type::QPath(box clean::QPathData { self_type, trait_, .. }) => {
                work.push_back(self_type);
                process_path(trait_.def_id());
            }
            _ => {}
        }
    }
    out
}

const MAX_FULL_EXAMPLES: usize = 5;
const NUM_VISIBLE_LINES: usize = 10;

/// Generates the HTML for example call locations generated via the --scrape-examples flag.
fn render_call_locations(w: &mut Buffer, cx: &mut Context<'_>, item: &clean::Item) {
    let tcx = cx.tcx();
    let def_id = item.item_id.expect_def_id();
    let key = tcx.def_path_hash(def_id);
    let Some(call_locations) = cx.shared.call_locations.get(&key) else { return };

    // Generate a unique ID so users can link to this section for a given method
    let id = cx.id_map.derive("scraped-examples");
    write!(
        w,
        "<div class=\"docblock scraped-example-list\">\
          <span></span>\
          <h5 id=\"{id}\">\
             <a href=\"#{id}\">Examples found in repository</a>\
             <a class=\"scrape-help\" href=\"{root_path}scrape-examples-help.html\">?</a>\
          </h5>",
        root_path = cx.root_path(),
        id = id
    );

    // Create a URL to a particular location in a reverse-dependency's source file
    let link_to_loc = |call_data: &CallData, loc: &CallLocation| -> (String, String) {
        let (line_lo, line_hi) = loc.call_expr.line_span;
        let (anchor, title) = if line_lo == line_hi {
            ((line_lo + 1).to_string(), format!("line {}", line_lo + 1))
        } else {
            (
                format!("{}-{}", line_lo + 1, line_hi + 1),
                format!("lines {}-{}", line_lo + 1, line_hi + 1),
            )
        };
        let url = format!("{}{}#{}", cx.root_path(), call_data.url, anchor);
        (url, title)
    };

    // Generate the HTML for a single example, being the title and code block
    let write_example = |w: &mut Buffer, (path, call_data): (&PathBuf, &CallData)| -> bool {
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(err) => {
                let span = item.span(tcx).map_or(rustc_span::DUMMY_SP, |span| span.inner());
                tcx.sess
                    .span_err(span, &format!("failed to read file {}: {}", path.display(), err));
                return false;
            }
        };

        // To reduce file sizes, we only want to embed the source code needed to understand the example, not
        // the entire file. So we find the smallest byte range that covers all items enclosing examples.
        assert!(!call_data.locations.is_empty());
        let min_loc =
            call_data.locations.iter().min_by_key(|loc| loc.enclosing_item.byte_span.0).unwrap();
        let byte_min = min_loc.enclosing_item.byte_span.0;
        let line_min = min_loc.enclosing_item.line_span.0;
        let max_loc =
            call_data.locations.iter().max_by_key(|loc| loc.enclosing_item.byte_span.1).unwrap();
        let byte_max = max_loc.enclosing_item.byte_span.1;
        let line_max = max_loc.enclosing_item.line_span.1;

        // The output code is limited to that byte range.
        let contents_subset = &contents[(byte_min as usize)..(byte_max as usize)];

        // The call locations need to be updated to reflect that the size of the program has changed.
        // Specifically, the ranges are all subtracted by `byte_min` since that's the new zero point.
        let (mut byte_ranges, line_ranges): (Vec<_>, Vec<_>) = call_data
            .locations
            .iter()
            .map(|loc| {
                let (byte_lo, byte_hi) = loc.call_ident.byte_span;
                let (line_lo, line_hi) = loc.call_expr.line_span;
                let byte_range = (byte_lo - byte_min, byte_hi - byte_min);

                let line_range = (line_lo - line_min, line_hi - line_min);
                let (line_url, line_title) = link_to_loc(call_data, loc);

                (byte_range, (line_range, line_url, line_title))
            })
            .unzip();

        let (_, init_url, init_title) = &line_ranges[0];
        let needs_expansion = line_max - line_min > NUM_VISIBLE_LINES;
        let locations_encoded = serde_json::to_string(&line_ranges).unwrap();

        write!(
            w,
            "<div class=\"scraped-example {expanded_cls}\" data-locs=\"{locations}\">\
                <div class=\"scraped-example-title\">\
                   {name} (<a href=\"{url}\">{title}</a>)\
                </div>\
                <div class=\"code-wrapper\">",
            expanded_cls = if needs_expansion { "" } else { "expanded" },
            name = call_data.display_name,
            url = init_url,
            title = init_title,
            // The locations are encoded as a data attribute, so they can be read
            // later by the JS for interactions.
            locations = Escape(&locations_encoded)
        );

        if line_ranges.len() > 1 {
            write!(w, r#"<button class="prev">&pr;</button> <button class="next">&sc;</button>"#);
        }

        // Look for the example file in the source map if it exists, otherwise return a dummy span
        let file_span = (|| {
            let source_map = tcx.sess.source_map();
            let crate_src = tcx.sess.local_crate_source_file()?;
            let abs_crate_src = crate_src.canonicalize().ok()?;
            let crate_root = abs_crate_src.parent()?.parent()?;
            let rel_path = path.strip_prefix(crate_root).ok()?;
            let files = source_map.files();
            let file = files.iter().find(|file| match &file.name {
                FileName::Real(RealFileName::LocalPath(other_path)) => rel_path == other_path,
                _ => false,
            })?;
            Some(rustc_span::Span::with_root_ctxt(
                file.start_pos + BytePos(byte_min),
                file.start_pos + BytePos(byte_max),
            ))
        })()
        .unwrap_or(rustc_span::DUMMY_SP);

        let mut decoration_info = FxHashMap::default();
        decoration_info.insert("highlight focus", vec![byte_ranges.remove(0)]);
        decoration_info.insert("highlight", byte_ranges);

        sources::print_src(
            w,
            contents_subset,
            file_span,
            cx,
            &cx.root_path(),
            highlight::DecorationInfo(decoration_info),
            sources::SourceContext::Embedded { offset: line_min, needs_expansion },
        );
        write!(w, "</div></div>");

        true
    };

    // The call locations are output in sequence, so that sequence needs to be determined.
    // Ideally the most "relevant" examples would be shown first, but there's no general algorithm
    // for determining relevance. We instead proxy relevance with the following heuristics:
    //   1. Code written to be an example is better than code not written to be an example, e.g.
    //      a snippet from examples/foo.rs is better than src/lib.rs. We don't know the Cargo
    //      directory structure in Rustdoc, so we proxy this by prioritizing code that comes from
    //      a --crate-type bin.
    //   2. Smaller examples are better than large examples. So we prioritize snippets that have
    //      the smallest number of lines in their enclosing item.
    //   3. Finally we sort by the displayed file name, which is arbitrary but prevents the
    //      ordering of examples from randomly changing between Rustdoc invocations.
    let ordered_locations = {
        fn sort_criterion<'a>(
            (_, call_data): &(&PathBuf, &'a CallData),
        ) -> (bool, u32, &'a String) {
            // Use the first location because that's what the user will see initially
            let (lo, hi) = call_data.locations[0].enclosing_item.byte_span;
            (!call_data.is_bin, hi - lo, &call_data.display_name)
        }

        let mut locs = call_locations.iter().collect::<Vec<_>>();
        locs.sort_by_key(sort_criterion);
        locs
    };

    let mut it = ordered_locations.into_iter().peekable();

    // An example may fail to write if its source can't be read for some reason, so this method
    // continues iterating until a write succeeds
    let write_and_skip_failure = |w: &mut Buffer, it: &mut Peekable<_>| {
        while let Some(example) = it.next() {
            if write_example(&mut *w, example) {
                break;
            }
        }
    };

    // Write just one example that's visible by default in the method's description.
    write_and_skip_failure(w, &mut it);

    // Then add the remaining examples in a hidden section.
    if it.peek().is_some() {
        write!(
            w,
            "<details class=\"toggle more-examples-toggle\">\
                  <summary class=\"hideme\">\
                     <span>More examples</span>\
                  </summary>\
                  <div class=\"hide-more\">Hide additional examples</div>\
                  <div class=\"more-scraped-examples\">\
                    <div class=\"toggle-line\"><div class=\"toggle-line-inner\"></div></div>"
        );

        // Only generate inline code for MAX_FULL_EXAMPLES number of examples. Otherwise we could
        // make the page arbitrarily huge!
        for _ in 0..MAX_FULL_EXAMPLES {
            write_and_skip_failure(w, &mut it);
        }

        // For the remaining examples, generate a <ul> containing links to the source files.
        if it.peek().is_some() {
            write!(w, r#"<div class="example-links">Additional examples can be found in:<br><ul>"#);
            it.for_each(|(_, call_data)| {
                let (url, _) = link_to_loc(call_data, &call_data.locations[0]);
                write!(
                    w,
                    r#"<li><a href="{url}">{name}</a></li>"#,
                    url = url,
                    name = call_data.display_name
                );
            });
            write!(w, "</ul></div>");
        }

        write!(w, "</div></details>");
    }

    write!(w, "</div>");
}
