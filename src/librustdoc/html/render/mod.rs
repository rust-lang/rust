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
mod ordered_json;
mod print_item;
pub(crate) mod sidebar;
mod sorted_template;
mod span_map;
mod type_layout;
mod write_shared;

use std::collections::VecDeque;
use std::fmt::{self, Display as _, Write};
use std::iter::Peekable;
use std::path::PathBuf;
use std::{fs, str};

use askama::Template;
use rustc_attr_parsing::{
    ConstStability, DeprecatedSince, Deprecation, RustcVersion, StabilityLevel, StableSince,
};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::Mutability;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_middle::ty::print::PrintTraitRefExt;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::symbol::{Symbol, sym};
use rustc_span::{BytePos, DUMMY_SP, FileName, RealFileName};
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use tracing::{debug, info};

pub(crate) use self::context::*;
pub(crate) use self::span_map::{LinkFromSrc, collect_spans_and_sources};
pub(crate) use self::write_shared::*;
use crate::clean::{self, ItemId, RenderedLink};
use crate::display::{Joined as _, MaybeDisplay as _};
use crate::error::Error;
use crate::formats::Impl;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::escape::Escape;
use crate::html::format::{
    Ending, HrefError, PrintWithSpace, href, join_with_double_colon, print_abi_with_space,
    print_constness_with_space, print_default_space, print_generic_bounds, print_where_clause,
    visibility_print_with_space, write_str,
};
use crate::html::markdown::{
    HeadingOffset, IdMap, Markdown, MarkdownItemInfo, MarkdownSummaryLine,
};
use crate::html::static_files::SCRAPE_EXAMPLES_HELP_MD;
use crate::html::{highlight, sources};
use crate::scrape_examples::{CallData, CallLocation};
use crate::{DOC_RUST_LANG_ORG_VERSION, try_none};

pub(crate) fn ensure_trailing_slash(v: &str) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        if !v.ends_with('/') && !v.is_empty() { write!(f, "{v}/") } else { f.write_str(v) }
    })
}

/// Specifies whether rendering directly implemented trait items or ones from a certain Deref
/// impl.
#[derive(Copy, Clone, Debug)]
pub(crate) enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Path, type_: &'a clean::Type, deref_mut_: bool },
}

/// For different handling of associated items from the Deref target of a type rather than the type
/// itself.
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum RenderMode {
    Normal,
    ForDeref { mut_: bool },
}

// Helper structs for rendering items/sidebars and carrying along contextual
// information

/// Struct representing one entry in the JS search index. These are all emitted
/// by hand to a large JS file at the end of cache-creation.
#[derive(Debug)]
pub(crate) struct IndexItem {
    pub(crate) ty: ItemType,
    pub(crate) defid: Option<DefId>,
    pub(crate) name: Symbol,
    pub(crate) path: String,
    pub(crate) desc: String,
    pub(crate) parent: Option<DefId>,
    pub(crate) parent_idx: Option<isize>,
    pub(crate) exact_path: Option<String>,
    pub(crate) impl_id: Option<DefId>,
    pub(crate) search_type: Option<IndexItemFunctionType>,
    pub(crate) aliases: Box<[Symbol]>,
    pub(crate) deprecation: Option<Deprecation>,
}

/// A type used for the search index.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct RenderType {
    id: Option<RenderTypeId>,
    generics: Option<Vec<RenderType>>,
    bindings: Option<Vec<(RenderTypeId, Vec<RenderType>)>>,
}

impl RenderType {
    // Types are rendered as lists of lists, because that's pretty compact.
    // The contents of the lists are always integers in self-terminating hex
    // form, handled by `RenderTypeId::write_to_string`, so no commas are
    // needed to separate the items.
    pub fn write_to_string(&self, string: &mut String) {
        fn write_optional_id(id: Option<RenderTypeId>, string: &mut String) {
            // 0 is a sentinel, everything else is one-indexed
            match id {
                Some(id) => id.write_to_string(string),
                None => string.push('`'),
            }
        }
        // Either just the type id, or `{type, generics, bindings?}`
        // where generics is a list of types,
        // and bindings is a list of `{id, typelist}` pairs.
        if self.generics.is_some() || self.bindings.is_some() {
            string.push('{');
            write_optional_id(self.id, string);
            string.push('{');
            for generic in self.generics.as_deref().unwrap_or_default() {
                generic.write_to_string(string);
            }
            string.push('}');
            if self.bindings.is_some() {
                string.push('{');
                for binding in self.bindings.as_deref().unwrap_or_default() {
                    string.push('{');
                    binding.0.write_to_string(string);
                    string.push('{');
                    for constraint in &binding.1[..] {
                        constraint.write_to_string(string);
                    }
                    string.push_str("}}");
                }
                string.push('}');
            }
            string.push('}');
        } else {
            write_optional_id(self.id, string);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderTypeId {
    DefId(DefId),
    Primitive(clean::PrimitiveType),
    AssociatedType(Symbol),
    Index(isize),
    Mut,
}

impl RenderTypeId {
    pub fn write_to_string(&self, string: &mut String) {
        let id: i32 = match &self {
            // 0 is a sentinel, everything else is one-indexed
            // concrete type
            RenderTypeId::Index(idx) if *idx >= 0 => (idx + 1isize).try_into().unwrap(),
            // generic type parameter
            RenderTypeId::Index(idx) => (*idx).try_into().unwrap(),
            _ => panic!("must convert render types to indexes before serializing"),
        };
        search_index::encode::write_vlqhex_to_string(id, string);
    }
}

/// Full type of functions/methods in the search index.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct IndexItemFunctionType {
    inputs: Vec<RenderType>,
    output: Vec<RenderType>,
    where_clause: Vec<Vec<RenderType>>,
    param_names: Vec<Symbol>,
}

impl IndexItemFunctionType {
    pub fn write_to_string<'a>(
        &'a self,
        string: &mut String,
        backref_queue: &mut VecDeque<&'a IndexItemFunctionType>,
    ) {
        assert!(backref_queue.len() <= 16);
        // If we couldn't figure out a type, just write 0,
        // which is encoded as `` ` `` (see RenderTypeId::write_to_string).
        let has_missing = self
            .inputs
            .iter()
            .chain(self.output.iter())
            .any(|i| i.id.is_none() && i.generics.is_none());
        if has_missing {
            string.push('`');
        } else if let Some(idx) = backref_queue.iter().position(|other| *other == self) {
            // The backref queue has 16 items, so backrefs use
            // a single hexit, disjoint from the ones used for numbers.
            string.push(
                char::try_from('0' as u32 + u32::try_from(idx).unwrap())
                    .expect("last possible value is '?'"),
            );
        } else {
            backref_queue.push_front(self);
            if backref_queue.len() > 16 {
                backref_queue.pop_back();
            }
            string.push('{');
            match &self.inputs[..] {
                [one] if one.generics.is_none() && one.bindings.is_none() => {
                    one.write_to_string(string);
                }
                _ => {
                    string.push('{');
                    for item in &self.inputs[..] {
                        item.write_to_string(string);
                    }
                    string.push('}');
                }
            }
            match &self.output[..] {
                [] if self.where_clause.is_empty() => {}
                [one] if one.generics.is_none() && one.bindings.is_none() => {
                    one.write_to_string(string);
                }
                _ => {
                    string.push('{');
                    for item in &self.output[..] {
                        item.write_to_string(string);
                    }
                    string.push('}');
                }
            }
            for constraint in &self.where_clause {
                if let [one] = &constraint[..]
                    && one.generics.is_none()
                    && one.bindings.is_none()
                {
                    one.write_to_string(string);
                } else {
                    string.push('{');
                    for item in &constraint[..] {
                        item.write_to_string(string);
                    }
                    string.push('}');
                }
            }
            string.push('}');
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
    pub(crate) fn print(&self) -> impl fmt::Display {
        fmt::from_fn(move |f| write!(f, "<a href=\"{}\">{}</a>", self.url, Escape(&self.name)))
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
    structs: FxIndexSet<ItemEntry>,
    enums: FxIndexSet<ItemEntry>,
    unions: FxIndexSet<ItemEntry>,
    primitives: FxIndexSet<ItemEntry>,
    traits: FxIndexSet<ItemEntry>,
    macros: FxIndexSet<ItemEntry>,
    functions: FxIndexSet<ItemEntry>,
    type_aliases: FxIndexSet<ItemEntry>,
    statics: FxIndexSet<ItemEntry>,
    constants: FxIndexSet<ItemEntry>,
    attribute_macros: FxIndexSet<ItemEntry>,
    derive_macros: FxIndexSet<ItemEntry>,
    trait_aliases: FxIndexSet<ItemEntry>,
}

impl AllTypes {
    fn new() -> AllTypes {
        let new_set = |cap| FxIndexSet::with_capacity_and_hasher(cap, Default::default());
        AllTypes {
            structs: new_set(100),
            enums: new_set(100),
            unions: new_set(100),
            primitives: new_set(26),
            traits: new_set(100),
            macros: new_set(100),
            functions: new_set(100),
            type_aliases: new_set(100),
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
            let new_url = format!("{}/{item_type}.{name}.html", url.join("/"));
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
                ItemType::TypeAlias => self.type_aliases.insert(ItemEntry::new(new_url, name)),
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
        if !self.type_aliases.is_empty() {
            sections.insert(ItemSection::TypeAliases);
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

    fn print(&self, f: &mut String) {
        fn print_entries(f: &mut String, e: &FxIndexSet<ItemEntry>, kind: ItemSection) {
            if !e.is_empty() {
                let mut e: Vec<&ItemEntry> = e.iter().collect();
                e.sort();
                write_str(
                    f,
                    format_args!(
                        "<h3 id=\"{id}\">{title}</h3><ul class=\"all-items\">",
                        id = kind.id(),
                        title = kind.name(),
                    ),
                );

                for s in e.iter() {
                    write_str(f, format_args!("<li>{}</li>", s.print()));
                }

                f.push_str("</ul>");
            }
        }

        f.push_str("<h1>List of all items</h1>");
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
        print_entries(f, &self.type_aliases, ItemSection::TypeAliases);
        print_entries(f, &self.trait_aliases, ItemSection::TraitAliases);
        print_entries(f, &self.statics, ItemSection::Statics);
        print_entries(f, &self.constants, ItemSection::Constants);
    }
}

fn scrape_examples_help(shared: &SharedContext<'_>) -> String {
    let mut content = SCRAPE_EXAMPLES_HELP_MD.to_owned();
    content.push_str(&format!(
        "## More information\n\n\
      If you want more information about this feature, please read the [corresponding chapter in \
      the Rustdoc book]({DOC_RUST_LANG_ORG_VERSION}/rustdoc/scraped-examples.html)."
    ));

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
            heading_offset: HeadingOffset::H1,
        }
        .into_string()
    )
}

fn document(
    cx: &Context<'_>,
    item: &clean::Item,
    parent: Option<&clean::Item>,
    heading_offset: HeadingOffset,
) -> impl fmt::Display {
    if let Some(ref name) = item.name {
        info!("Documenting {name}");
    }

    fmt::from_fn(move |f| {
        document_item_info(cx, item, parent).render_into(f).unwrap();
        if parent.is_none() {
            write!(f, "{}", document_full_collapsible(item, cx, heading_offset))
        } else {
            write!(f, "{}", document_full(item, cx, heading_offset))
        }
    })
}

/// Render md_text as markdown.
fn render_markdown(
    cx: &Context<'_>,
    md_text: &str,
    links: Vec<RenderedLink>,
    heading_offset: HeadingOffset,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        write!(
            f,
            "<div class=\"docblock\">{}</div>",
            Markdown {
                content: md_text,
                links: &links,
                ids: &mut cx.id_map.borrow_mut(),
                error_codes: cx.shared.codes,
                edition: cx.shared.edition(),
                playground: &cx.shared.playground,
                heading_offset,
            }
            .into_string()
        )
    })
}

/// Writes a documentation block containing only the first paragraph of the documentation. If the
/// docs are longer, a "Read more" link is appended to the end.
fn document_short(
    item: &clean::Item,
    cx: &Context<'_>,
    link: AssocItemLink<'_>,
    parent: &clean::Item,
    show_def_docs: bool,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        document_item_info(cx, item, Some(parent)).render_into(f).unwrap();
        if !show_def_docs {
            return Ok(());
        }
        let s = item.doc_value();
        if !s.is_empty() {
            let (mut summary_html, has_more_content) =
                MarkdownSummaryLine(&s, &item.links(cx)).into_string_with_has_more_content();

            let link = if has_more_content {
                let link = fmt::from_fn(|f| {
                    write!(
                        f,
                        " <a{}>Read more</a>",
                        assoc_href_attr(item, link, cx).maybe_display()
                    )
                });

                if let Some(idx) = summary_html.rfind("</p>") {
                    summary_html.insert_str(idx, &link.to_string());
                    None
                } else {
                    Some(link)
                }
            } else {
                None
            }
            .maybe_display();

            write!(f, "<div class='docblock'>{summary_html}{link}</div>")?;
        }
        Ok(())
    })
}

fn document_full_collapsible(
    item: &clean::Item,
    cx: &Context<'_>,
    heading_offset: HeadingOffset,
) -> impl fmt::Display {
    document_full_inner(item, cx, true, heading_offset)
}

fn document_full(
    item: &clean::Item,
    cx: &Context<'_>,
    heading_offset: HeadingOffset,
) -> impl fmt::Display {
    document_full_inner(item, cx, false, heading_offset)
}

fn document_full_inner(
    item: &clean::Item,
    cx: &Context<'_>,
    is_collapsible: bool,
    heading_offset: HeadingOffset,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        if let Some(s) = item.opt_doc_value() {
            debug!("Doc block: =====\n{s}\n=====");
            if is_collapsible {
                write!(
                    f,
                    "<details class=\"toggle top-doc\" open>\
                     <summary class=\"hideme\">\
                        <span>Expand description</span>\
                     </summary>{}</details>",
                    render_markdown(cx, &s, item.links(cx), heading_offset)
                )?;
            } else {
                write!(f, "{}", render_markdown(cx, &s, item.links(cx), heading_offset))?;
            }
        }

        let kind = match &item.kind {
            clean::ItemKind::StrippedItem(box kind) | kind => kind,
        };

        if let clean::ItemKind::FunctionItem(..) | clean::ItemKind::MethodItem(..) = kind {
            render_call_locations(f, cx, item);
        }
        Ok(())
    })
}

#[derive(Template)]
#[template(path = "item_info.html")]
struct ItemInfo {
    items: Vec<ShortItemInfo>,
}
/// Add extra information about an item such as:
///
/// * Stability
/// * Deprecated
/// * Required features (through the `doc_cfg` feature)
fn document_item_info(
    cx: &Context<'_>,
    item: &clean::Item,
    parent: Option<&clean::Item>,
) -> ItemInfo {
    let items = short_item_info(item, cx, parent);
    ItemInfo { items }
}

fn portability(item: &clean::Item, parent: Option<&clean::Item>) -> Option<String> {
    let cfg = match (&item.cfg, parent.and_then(|p| p.cfg.as_ref())) {
        (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
        (cfg, _) => cfg.as_deref().cloned(),
    };

    debug!(
        "Portability {name:?} {item_cfg:?} (parent: {parent:?}) - {parent_cfg:?} = {cfg:?}",
        name = item.name,
        item_cfg = item.cfg,
        parent_cfg = parent.and_then(|p| p.cfg.as_ref()),
    );

    Some(cfg?.render_long_html())
}

#[derive(Template)]
#[template(path = "short_item_info.html")]
enum ShortItemInfo {
    /// A message describing the deprecation of this item
    Deprecation {
        message: String,
    },
    /// The feature corresponding to an unstable item, and optionally
    /// a tracking issue URL and number.
    Unstable {
        feature: String,
        tracking: Option<(String, u32)>,
    },
    Portability {
        message: String,
    },
}

/// Render the stability, deprecation and portability information that is displayed at the top of
/// the item's documentation.
fn short_item_info(
    item: &clean::Item,
    cx: &Context<'_>,
    parent: Option<&clean::Item>,
) -> Vec<ShortItemInfo> {
    let mut extra_info = vec![];

    if let Some(depr @ Deprecation { note, since, suggestion: _ }) = item.deprecation(cx.tcx()) {
        // We display deprecation messages for #[deprecated], but only display
        // the future-deprecation messages for rustc versions.
        let mut message = match since {
            DeprecatedSince::RustcVersion(version) => {
                if depr.is_in_effect() {
                    format!("Deprecated since {version}")
                } else {
                    format!("Deprecating in {version}")
                }
            }
            DeprecatedSince::Future => String::from("Deprecating in a future version"),
            DeprecatedSince::NonStandard(since) => {
                format!("Deprecated since {}", Escape(since.as_str()))
            }
            DeprecatedSince::Unspecified | DeprecatedSince::Err => String::from("Deprecated"),
        };

        if let Some(note) = note {
            let note = note.as_str();
            let mut id_map = cx.id_map.borrow_mut();
            let html = MarkdownItemInfo(note, &mut id_map);
            message.push_str(": ");
            message.push_str(&html.into_string());
        }
        extra_info.push(ShortItemInfo::Deprecation { message });
    }

    // Render unstable items. But don't render "rustc_private" crates (internal compiler crates).
    // Those crates are permanently unstable so it makes no sense to render "unstable" everywhere.
    if let Some((StabilityLevel::Unstable { reason: _, issue, .. }, feature)) = item
        .stability(cx.tcx())
        .as_ref()
        .filter(|stab| stab.feature != sym::rustc_private)
        .map(|stab| (stab.level, stab.feature))
    {
        let tracking = if let (Some(url), Some(issue)) = (&cx.shared.issue_tracker_base_url, issue)
        {
            Some((url.clone(), issue.get()))
        } else {
            None
        };
        extra_info.push(ShortItemInfo::Unstable { feature: feature.to_string(), tracking });
    }

    if let Some(message) = portability(item, parent) {
        extra_info.push(ShortItemInfo::Portability { message });
    }

    extra_info
}

// Render the list of items inside one of the sections "Trait Implementations",
// "Auto Trait Implementations," "Blanket Trait Implementations" (on struct/enum pages).
pub(crate) fn render_impls(
    cx: &Context<'_>,
    mut w: impl Write,
    impls: &[&Impl],
    containing_item: &clean::Item,
    toggle_open_by_default: bool,
) {
    let mut rendered_impls = impls
        .iter()
        .map(|i| {
            let did = i.trait_did().unwrap();
            let provided_trait_methods = i.inner_impl().provided_trait_methods(cx.tcx());
            let assoc_link = AssocItemLink::GotoSource(did.into(), &provided_trait_methods);
            let imp = render_impl(
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
            imp.to_string()
        })
        .collect::<Vec<_>>();
    rendered_impls.sort();
    w.write_str(&rendered_impls.join("")).unwrap();
}

/// Build a (possibly empty) `href` attribute (a key-value pair) for the given associated item.
fn assoc_href_attr(
    it: &clean::Item,
    link: AssocItemLink<'_>,
    cx: &Context<'_>,
) -> Option<impl fmt::Display> {
    let name = it.name.unwrap();
    let item_type = it.type_();

    enum Href<'a> {
        AnchorId(&'a str),
        Anchor(ItemType),
        Url(String, ItemType),
    }

    let href = match link {
        AssocItemLink::Anchor(Some(id)) => Href::AnchorId(id),
        AssocItemLink::Anchor(None) => Href::Anchor(item_type),
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
                Ok((url, ..)) => Href::Url(url, item_type),
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
                Err(HrefError::DocumentationNotBuilt) => return None,
                Err(_) => Href::Anchor(item_type),
            }
        }
    };

    let href = fmt::from_fn(move |f| match &href {
        Href::AnchorId(id) => write!(f, "#{id}"),
        Href::Url(url, item_type) => {
            write!(f, "{url}#{item_type}.{name}")
        }
        Href::Anchor(item_type) => {
            write!(f, "#{item_type}.{name}")
        }
    });

    // If there is no `href` for the reason explained above, simply do not render it which is valid:
    // https://html.spec.whatwg.org/multipage/links.html#links-created-by-a-and-area-elements
    Some(fmt::from_fn(move |f| write!(f, " href=\"{href}\"")))
}

#[derive(Debug)]
enum AssocConstValue<'a> {
    // In trait definitions, it is relevant for the public API whether an
    // associated constant comes with a default value, so even if we cannot
    // render its value, the presence of a value must be shown using `= _`.
    TraitDefault(&'a clean::ConstantKind),
    // In impls, there is no need to show `= _`.
    Impl(&'a clean::ConstantKind),
    None,
}

fn assoc_const(
    it: &clean::Item,
    generics: &clean::Generics,
    ty: &clean::Type,
    value: AssocConstValue<'_>,
    link: AssocItemLink<'_>,
    indent: usize,
    cx: &Context<'_>,
) -> impl fmt::Display {
    let tcx = cx.tcx();
    fmt::from_fn(move |w| {
        write!(
            w,
            "{indent}{vis}const <a{href} class=\"constant\">{name}</a>{generics}: {ty}",
            indent = " ".repeat(indent),
            vis = visibility_print_with_space(it, cx),
            href = assoc_href_attr(it, link, cx).maybe_display(),
            name = it.name.as_ref().unwrap(),
            generics = generics.print(cx),
            ty = ty.print(cx),
        )?;
        if let AssocConstValue::TraitDefault(konst) | AssocConstValue::Impl(konst) = value {
            // FIXME: `.value()` uses `clean::utils::format_integer_with_underscore_sep` under the
            //        hood which adds noisy underscores and a type suffix to number literals.
            //        This hurts readability in this context especially when more complex expressions
            //        are involved and it doesn't add much of value.
            //        Find a way to print constants here without all that jazz.
            let repr = konst.value(tcx).unwrap_or_else(|| konst.expr(tcx));
            if match value {
                AssocConstValue::TraitDefault(_) => true, // always show
                AssocConstValue::Impl(_) => repr != "_", // show if there is a meaningful value to show
                AssocConstValue::None => unreachable!(),
            } {
                write!(w, " = {}", Escape(&repr))?;
            }
        }
        write!(w, "{}", print_where_clause(generics, cx, indent, Ending::NoNewline).maybe_display())
    })
}

fn assoc_type(
    it: &clean::Item,
    generics: &clean::Generics,
    bounds: &[clean::GenericBound],
    default: Option<&clean::Type>,
    link: AssocItemLink<'_>,
    indent: usize,
    cx: &Context<'_>,
) -> impl fmt::Display {
    fmt::from_fn(move |w| {
        write!(
            w,
            "{indent}{vis}type <a{href} class=\"associatedtype\">{name}</a>{generics}",
            indent = " ".repeat(indent),
            vis = visibility_print_with_space(it, cx),
            href = assoc_href_attr(it, link, cx).maybe_display(),
            name = it.name.as_ref().unwrap(),
            generics = generics.print(cx),
        )?;
        if !bounds.is_empty() {
            write!(w, ": {}", print_generic_bounds(bounds, cx))?;
        }
        // Render the default before the where-clause which aligns with the new recommended style. See #89122.
        if let Some(default) = default {
            write!(w, " = {}", default.print(cx))?;
        }
        write!(w, "{}", print_where_clause(generics, cx, indent, Ending::NoNewline).maybe_display())
    })
}

fn assoc_method(
    meth: &clean::Item,
    g: &clean::Generics,
    d: &clean::FnDecl,
    link: AssocItemLink<'_>,
    parent: ItemType,
    cx: &Context<'_>,
    render_mode: RenderMode,
) -> impl fmt::Display {
    let tcx = cx.tcx();
    let header = meth.fn_header(tcx).expect("Trying to get header from a non-function item");
    let name = meth.name.as_ref().unwrap();
    let vis = visibility_print_with_space(meth, cx).to_string();
    let defaultness = print_default_space(meth.is_default());
    // FIXME: Once https://github.com/rust-lang/rust/issues/67792 is implemented, we can remove
    // this condition.
    let constness = match render_mode {
        RenderMode::Normal => print_constness_with_space(
            &header.constness,
            meth.stable_since(tcx),
            meth.const_stability(tcx),
        ),
        RenderMode::ForDeref { .. } => "",
    };

    fmt::from_fn(move |w| {
        let asyncness = header.asyncness.print_with_space();
        let safety = header.safety.print_with_space();
        let abi = print_abi_with_space(header.abi).to_string();
        let href = assoc_href_attr(meth, link, cx).maybe_display();

        // NOTE: `{:#}` does not print HTML formatting, `{}` does. So `g.print` can't be reused between the length calculation and `write!`.
        let generics_len = format!("{:#}", g.print(cx)).len();
        let mut header_len = "fn ".len()
            + vis.len()
            + defaultness.len()
            + constness.len()
            + asyncness.len()
            + safety.len()
            + abi.len()
            + name.as_str().len()
            + generics_len;

        let notable_traits = notable_traits_button(&d.output, cx).maybe_display();

        let (indent, indent_str, end_newline) = if parent == ItemType::Trait {
            header_len += 4;
            let indent_str = "    ";
            write!(w, "{}", render_attributes_in_pre(meth, indent_str, cx))?;
            (4, indent_str, Ending::NoNewline)
        } else {
            render_attributes_in_code(w, meth, cx);
            (0, "", Ending::Newline)
        };
        write!(
            w,
            "{indent}{vis}{defaultness}{constness}{asyncness}{safety}{abi}fn \
            <a{href} class=\"fn\">{name}</a>{generics}{decl}{notable_traits}{where_clause}",
            indent = indent_str,
            generics = g.print(cx),
            decl = d.full_print(header_len, indent, cx),
            where_clause = print_where_clause(g, cx, indent, end_newline).maybe_display(),
        )
    })
}

/// Writes a span containing the versions at which an item became stable and/or const-stable. For
/// example, if the item became stable at 1.0.0, and const-stable at 1.45.0, this function would
/// write a span containing "1.0.0 (const: 1.45.0)".
///
/// Returns `None` if there is no stability annotation to be rendered.
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
    stable_version: Option<StableSince>,
    const_stability: Option<ConstStability>,
    extra_class: &str,
) -> Option<impl fmt::Display> {
    let mut title = String::new();
    let mut stability = String::new();

    if let Some(version) = stable_version.and_then(|version| since_to_string(&version)) {
        stability.push_str(&version);
        title.push_str(&format!("Stable since Rust version {version}"));
    }

    let const_title_and_stability = match const_stability {
        Some(ConstStability { level: StabilityLevel::Stable { since, .. }, .. }) => {
            since_to_string(&since)
                .map(|since| (format!("const since {since}"), format!("const: {since}")))
        }
        Some(ConstStability { level: StabilityLevel::Unstable { issue, .. }, feature, .. }) => {
            if stable_version.is_none() {
                // don't display const unstable if entirely unstable
                None
            } else {
                let unstable = if let Some(n) = issue {
                    format!(
                        "<a \
                        href=\"https://github.com/rust-lang/rust/issues/{n}\" \
                        title=\"Tracking issue for {feature}\"\
                       >unstable</a>"
                    )
                } else {
                    String::from("unstable")
                };

                Some((String::from("const unstable"), format!("const: {unstable}")))
            }
        }
        _ => None,
    };

    if let Some((const_title, const_stability)) = const_title_and_stability {
        if !title.is_empty() {
            title.push_str(&format!(", {const_title}"));
        } else {
            title.push_str(&const_title);
        }

        if !stability.is_empty() {
            stability.push_str(&format!(" ({const_stability})"));
        } else {
            stability.push_str(&const_stability);
        }
    }

    (!stability.is_empty()).then_some(fmt::from_fn(move |w| {
        write!(w, r#"<span class="since{extra_class}" title="{title}">{stability}</span>"#)
    }))
}

fn since_to_string(since: &StableSince) -> Option<String> {
    match since {
        StableSince::Version(since) => Some(since.to_string()),
        StableSince::Current => Some(RustcVersion::CURRENT.to_string()),
        StableSince::Err => None,
    }
}

#[inline]
fn render_stability_since_raw(
    ver: Option<StableSince>,
    const_stability: Option<ConstStability>,
) -> Option<impl fmt::Display> {
    render_stability_since_raw_with_extra(ver, const_stability, "")
}

fn render_assoc_item(
    item: &clean::Item,
    link: AssocItemLink<'_>,
    parent: ItemType,
    cx: &Context<'_>,
    render_mode: RenderMode,
) -> impl fmt::Display {
    fmt::from_fn(move |f| match &item.kind {
        clean::StrippedItem(..) => Ok(()),
        clean::RequiredMethodItem(m) | clean::MethodItem(m, _) => {
            assoc_method(item, &m.generics, &m.decl, link, parent, cx, render_mode).fmt(f)
        }
        clean::RequiredAssocConstItem(generics, ty) => assoc_const(
            item,
            generics,
            ty,
            AssocConstValue::None,
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        )
        .fmt(f),
        clean::ProvidedAssocConstItem(ci) => assoc_const(
            item,
            &ci.generics,
            &ci.type_,
            AssocConstValue::TraitDefault(&ci.kind),
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        )
        .fmt(f),
        clean::ImplAssocConstItem(ci) => assoc_const(
            item,
            &ci.generics,
            &ci.type_,
            AssocConstValue::Impl(&ci.kind),
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        )
        .fmt(f),
        clean::RequiredAssocTypeItem(generics, bounds) => assoc_type(
            item,
            generics,
            bounds,
            None,
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        )
        .fmt(f),
        clean::AssocTypeItem(ty, bounds) => assoc_type(
            item,
            &ty.generics,
            bounds,
            Some(ty.item_type.as_ref().unwrap_or(&ty.type_)),
            link,
            if parent == ItemType::Trait { 4 } else { 0 },
            cx,
        )
        .fmt(f),
        _ => panic!("render_assoc_item called on non-associated-item"),
    })
}

// When an attribute is rendered inside a `<pre>` tag, it is formatted using
// a whitespace prefix and newline.
fn render_attributes_in_pre(it: &clean::Item, prefix: &str, cx: &Context<'_>) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        for a in it.attributes(cx.tcx(), cx.cache(), false) {
            writeln!(f, "{prefix}{a}")?;
        }
        Ok(())
    })
}

// When an attribute is rendered inside a <code> tag, it is formatted using
// a div to produce a newline after it.
fn render_attributes_in_code(w: &mut impl fmt::Write, it: &clean::Item, cx: &Context<'_>) {
    for attr in it.attributes(cx.tcx(), cx.cache(), false) {
        write!(w, "<div class=\"code-attribute\">{attr}</div>").unwrap();
    }
}

#[derive(Copy, Clone)]
enum AssocItemLink<'a> {
    Anchor(Option<&'a str>),
    GotoSource(ItemId, &'a FxIndexSet<Symbol>),
}

impl<'a> AssocItemLink<'a> {
    fn anchor(&self, id: &'a str) -> Self {
        match *self {
            AssocItemLink::Anchor(_) => AssocItemLink::Anchor(Some(id)),
            ref other => *other,
        }
    }
}

pub fn write_section_heading(
    title: &str,
    id: &str,
    extra_class: Option<&str>,
    extra: impl fmt::Display,
) -> impl fmt::Display {
    fmt::from_fn(move |w| {
        let (extra_class, whitespace) = match extra_class {
            Some(extra) => (extra, " "),
            None => ("", ""),
        };
        write!(
            w,
            "<h2 id=\"{id}\" class=\"{extra_class}{whitespace}section-header\">\
            {title}\
            <a href=\"#{id}\" class=\"anchor\">§</a>\
         </h2>{extra}",
        )
    })
}

fn write_impl_section_heading(title: &str, id: &str) -> impl fmt::Display {
    write_section_heading(title, id, None, "")
}

pub(crate) fn render_all_impls(
    mut w: impl Write,
    cx: &Context<'_>,
    containing_item: &clean::Item,
    concrete: &[&Impl],
    synthetic: &[&Impl],
    blanket_impl: &[&Impl],
) {
    let impls = {
        let mut buf = String::new();
        render_impls(cx, &mut buf, concrete, containing_item, true);
        buf
    };
    if !impls.is_empty() {
        write!(
            w,
            "{}<div id=\"trait-implementations-list\">{impls}</div>",
            write_impl_section_heading("Trait Implementations", "trait-implementations")
        )
        .unwrap();
    }

    if !synthetic.is_empty() {
        write!(
            w,
            "{}<div id=\"synthetic-implementations-list\">",
            write_impl_section_heading("Auto Trait Implementations", "synthetic-implementations",)
        )
        .unwrap();
        render_impls(cx, &mut w, synthetic, containing_item, false);
        w.write_str("</div>").unwrap();
    }

    if !blanket_impl.is_empty() {
        write!(
            w,
            "{}<div id=\"blanket-implementations-list\">",
            write_impl_section_heading("Blanket Implementations", "blanket-implementations")
        )
        .unwrap();
        render_impls(cx, &mut w, blanket_impl, containing_item, false);
        w.write_str("</div>").unwrap();
    }
}

fn render_assoc_items(
    cx: &Context<'_>,
    containing_item: &clean::Item,
    it: DefId,
    what: AssocItemRender<'_>,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        let mut derefs = DefIdSet::default();
        derefs.insert(it);
        render_assoc_items_inner(f, cx, containing_item, it, what, &mut derefs);
        Ok(())
    })
}

fn render_assoc_items_inner(
    mut w: &mut dyn fmt::Write,
    cx: &Context<'_>,
    containing_item: &clean::Item,
    it: DefId,
    what: AssocItemRender<'_>,
    derefs: &mut DefIdSet,
) {
    info!("Documenting associated items of {:?}", containing_item.name);
    let cache = &cx.shared.cache;
    let Some(v) = cache.impls.get(&it) else { return };
    let (mut non_trait, traits): (Vec<_>, _) =
        v.iter().partition(|i| i.inner_impl().trait_.is_none());
    if !non_trait.is_empty() {
        let mut close_tags = <Vec<&str>>::with_capacity(1);
        let mut tmp_buf = String::new();
        let (render_mode, id, class_html) = match what {
            AssocItemRender::All => {
                write_str(
                    &mut tmp_buf,
                    format_args!(
                        "{}",
                        write_impl_section_heading("Implementations", "implementations")
                    ),
                );
                (RenderMode::Normal, "implementations-list".to_owned(), "")
            }
            AssocItemRender::DerefFor { trait_, type_, deref_mut_ } => {
                let id =
                    cx.derive_id(small_url_encode(format!("deref-methods-{:#}", type_.print(cx))));
                // the `impls.get` above only looks at the outermost type,
                // and the Deref impl may only be implemented for certain
                // values of generic parameters.
                // for example, if an item impls `Deref<[u8]>`,
                // we should not show methods from `[MaybeUninit<u8>]`.
                // this `retain` filters out any instances where
                // the types do not line up perfectly.
                non_trait.retain(|impl_| {
                    type_.is_doc_subtype_of(&impl_.inner_impl().for_, &cx.shared.cache)
                });
                let derived_id = cx.derive_id(&id);
                close_tags.push("</details>");
                write_str(
                    &mut tmp_buf,
                    format_args!(
                        "<details class=\"toggle big-toggle\" open><summary>{}</summary>",
                        write_impl_section_heading(
                            &format!(
                                "<span>Methods from {trait_}&lt;Target = {type_}&gt;</span>",
                                trait_ = trait_.print(cx),
                                type_ = type_.print(cx),
                            ),
                            &id,
                        )
                    ),
                );
                if let Some(def_id) = type_.def_id(cx.cache()) {
                    cx.deref_id_map.borrow_mut().insert(def_id, id);
                }
                (RenderMode::ForDeref { mut_: deref_mut_ }, derived_id, r#" class="impl-items""#)
            }
        };
        let mut impls_buf = String::new();
        for i in &non_trait {
            write_str(
                &mut impls_buf,
                format_args!(
                    "{}",
                    render_impl(
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
                    )
                ),
            );
        }
        if !impls_buf.is_empty() {
            write!(w, "{tmp_buf}<div id=\"{id}\"{class_html}>{impls_buf}</div>").unwrap();
            for tag in close_tags.into_iter().rev() {
                w.write_str(tag).unwrap();
            }
        }
    }

    if !traits.is_empty() {
        let deref_impl =
            traits.iter().find(|t| t.trait_did() == cx.tcx().lang_items().deref_trait());
        if let Some(impl_) = deref_impl {
            let has_deref_mut =
                traits.iter().any(|t| t.trait_did() == cx.tcx().lang_items().deref_mut_trait());
            render_deref_methods(&mut w, cx, impl_, containing_item, has_deref_mut, derefs);
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

/// `derefs` is the set of all deref targets that have already been handled.
fn render_deref_methods(
    mut w: impl Write,
    cx: &Context<'_>,
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
        .find_map(|item| match item.kind {
            clean::AssocTypeItem(box ref t, _) => Some(match *t {
                clean::TypeAlias { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
        .expect("Expected associated type binding");
    debug!(
        "Render deref methods for {for_:#?}, target {target:#?}",
        for_ = impl_.inner_impl().for_
    );
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
        render_assoc_items_inner(&mut w, cx, container_item, did, what, derefs);
    } else if let Some(prim) = target.primitive_type() {
        if let Some(&did) = cache.primitive_locations.get(&prim) {
            render_assoc_items_inner(&mut w, cx, container_item, did, what, derefs);
        }
    }
}

fn should_render_item(item: &clean::Item, deref_mut_: bool, tcx: TyCtxt<'_>) -> bool {
    let self_type_opt = match item.kind {
        clean::MethodItem(ref method, _) => method.decl.receiver_type(),
        clean::RequiredMethodItem(ref method) => method.decl.receiver_type(),
        _ => None,
    };

    if let Some(self_ty) = self_type_opt {
        let (by_mut_ref, by_box, by_value) = match *self_ty {
            clean::Type::BorrowedRef { mutability, .. } => {
                (mutability == Mutability::Mut, false, false)
            }
            clean::Type::Path { ref path } => {
                (false, Some(path.def_id()) == tcx.lang_items().owned_box(), false)
            }
            clean::Type::SelfTy => (false, false, true),
            _ => (false, false, false),
        };

        (deref_mut_ || !by_mut_ref) && !by_box && !by_value
    } else {
        false
    }
}

pub(crate) fn notable_traits_button(
    ty: &clean::Type,
    cx: &Context<'_>,
) -> Option<impl fmt::Display> {
    if ty.is_unit() {
        // Very common fast path.
        return None;
    }

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

    let impls = cx.cache().impls.get(&did)?;
    let has_notable_trait = impls
        .iter()
        .map(Impl::inner_impl)
        .filter(|impl_| {
            impl_.polarity == ty::ImplPolarity::Positive
                // Two different types might have the same did,
                // without actually being the same.
                && ty.is_doc_subtype_of(&impl_.for_, cx.cache())
        })
        .filter_map(|impl_| impl_.trait_.as_ref())
        .filter_map(|trait_| cx.cache().traits.get(&trait_.def_id()))
        .any(|t| t.is_notable_trait(cx.tcx()));

    has_notable_trait.then(|| {
        cx.types_with_notable_traits.borrow_mut().insert(ty.clone());
        fmt::from_fn(|f| {
            write!(
                f,
                " <a href=\"#\" class=\"tooltip\" data-notable-ty=\"{ty}\">ⓘ</a>",
                ty = Escape(&format!("{:#}", ty.print(cx))),
            )
        })
    })
}

fn notable_traits_decl(ty: &clean::Type, cx: &Context<'_>) -> (String, String) {
    let mut out = String::new();

    let did = ty.def_id(cx.cache()).expect("notable_traits_button already checked this");

    let impls = cx.cache().impls.get(&did).expect("notable_traits_button already checked this");

    for i in impls {
        let impl_ = i.inner_impl();
        if impl_.polarity != ty::ImplPolarity::Positive {
            continue;
        }

        if !ty.is_doc_subtype_of(&impl_.for_, cx.cache()) {
            // Two different types might have the same did,
            // without actually being the same.
            continue;
        }
        if let Some(trait_) = &impl_.trait_ {
            let trait_did = trait_.def_id();

            if cx.cache().traits.get(&trait_did).is_some_and(|t| t.is_notable_trait(cx.tcx())) {
                if out.is_empty() {
                    write_str(
                        &mut out,
                        format_args!(
                            "<h3>Notable traits for <code>{}</code></h3>\
                            <pre><code>",
                            impl_.for_.print(cx)
                        ),
                    );
                }

                write_str(
                    &mut out,
                    format_args!("<div class=\"where\">{}</div>", impl_.print(false, cx)),
                );
                for it in &impl_.items {
                    if let clean::AssocTypeItem(ref tydef, ref _bounds) = it.kind {
                        let empty_set = FxIndexSet::default();
                        let src_link = AssocItemLink::GotoSource(trait_did.into(), &empty_set);
                        write_str(
                            &mut out,
                            format_args!(
                                "<div class=\"where\">    {};</div>",
                                assoc_type(
                                    it,
                                    &tydef.generics,
                                    &[], // intentionally leaving out bounds
                                    Some(&tydef.type_),
                                    src_link,
                                    0,
                                    cx,
                                )
                            ),
                        );
                    }
                }
            }
        }
    }
    if out.is_empty() {
        out.push_str("</code></pre>");
    }

    (format!("{:#}", ty.print(cx)), out)
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
    cx: &Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    link: AssocItemLink<'_>,
    render_mode: RenderMode,
    use_absolute: Option<bool>,
    aliases: &[String],
    rendering_params: ImplRenderingParameters,
) -> impl fmt::Display {
    fmt::from_fn(move |w| {
        let cache = &cx.shared.cache;
        let traits = &cache.traits;
        let trait_ = i.trait_did().map(|did| &traits[&did]);
        let mut close_tags = <Vec<&str>>::with_capacity(2);

        // For trait implementations, the `interesting` output contains all methods that have doc
        // comments, and the `boring` output contains all methods that do not. The distinction is
        // used to allow hiding the boring methods.
        // `containing_item` is used for rendering stability info. If the parent is a trait impl,
        // `containing_item` will the grandparent, since trait impls can't have stability attached.
        fn doc_impl_item(
            boring: &mut String,
            interesting: &mut String,
            cx: &Context<'_>,
            item: &clean::Item,
            parent: &clean::Item,
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

            let mut doc_buffer = String::new();
            let mut info_buffer = String::new();
            let mut short_documented = true;

            if render_method_item {
                if !is_default_item {
                    if let Some(t) = trait_ {
                        // The trait item may have been stripped so we might not
                        // find any documentation or stability for it.
                        if let Some(it) = t.items.iter().find(|i| i.name == item.name) {
                            // We need the stability of the item from the trait
                            // because impls can't have a stability.
                            if !item.doc_value().is_empty() {
                                document_item_info(cx, it, Some(parent))
                                    .render_into(&mut info_buffer)
                                    .unwrap();
                                write_str(
                                    &mut doc_buffer,
                                    format_args!("{}", document_full(item, cx, HeadingOffset::H5)),
                                );
                                short_documented = false;
                            } else {
                                // In case the item isn't documented,
                                // provide short documentation from the trait.
                                write_str(
                                    &mut doc_buffer,
                                    format_args!(
                                        "{}",
                                        document_short(
                                            it,
                                            cx,
                                            link,
                                            parent,
                                            rendering_params.show_def_docs,
                                        )
                                    ),
                                );
                            }
                        }
                    } else {
                        document_item_info(cx, item, Some(parent))
                            .render_into(&mut info_buffer)
                            .unwrap();
                        if rendering_params.show_def_docs {
                            write_str(
                                &mut doc_buffer,
                                format_args!("{}", document_full(item, cx, HeadingOffset::H5)),
                            );
                            short_documented = false;
                        }
                    }
                } else {
                    write_str(
                        &mut doc_buffer,
                        format_args!(
                            "{}",
                            document_short(item, cx, link, parent, rendering_params.show_def_docs)
                        ),
                    );
                }
            }
            let w = if short_documented && trait_.is_some() { interesting } else { boring };

            let toggled = !doc_buffer.is_empty();
            if toggled {
                let method_toggle_class = if item_type.is_method() { " method-toggle" } else { "" };
                write_str(
                    w,
                    format_args!("<details class=\"toggle{method_toggle_class}\" open><summary>"),
                );
            }
            match &item.kind {
                clean::MethodItem(..) | clean::RequiredMethodItem(_) => {
                    // Only render when the method is not static or we allow static methods
                    if render_method_item {
                        let id = cx.derive_id(format!("{item_type}.{name}"));
                        let source_id = trait_
                            .and_then(|trait_| {
                                trait_
                                    .items
                                    .iter()
                                    .find(|item| item.name.map(|n| n == *name).unwrap_or(false))
                            })
                            .map(|item| format!("{}.{name}", item.type_()));
                        write_str(
                            w,
                            format_args!(
                                "<section id=\"{id}\" class=\"{item_type}{in_trait_class}\">\
                                {}",
                                render_rightside(cx, item, render_mode)
                            ),
                        );
                        if trait_.is_some() {
                            // Anchors are only used on trait impls.
                            write_str(w, format_args!("<a href=\"#{id}\" class=\"anchor\">§</a>"));
                        }
                        write_str(
                            w,
                            format_args!(
                                "<h4 class=\"code-header\">{}</h4></section>",
                                render_assoc_item(
                                    item,
                                    link.anchor(source_id.as_ref().unwrap_or(&id)),
                                    ItemType::Impl,
                                    cx,
                                    render_mode,
                                ),
                            ),
                        );
                    }
                }
                clean::RequiredAssocConstItem(generics, ty) => {
                    let source_id = format!("{item_type}.{name}");
                    let id = cx.derive_id(&source_id);
                    write_str(
                        w,
                        format_args!(
                            "<section id=\"{id}\" class=\"{item_type}{in_trait_class}\">\
                            {}",
                            render_rightside(cx, item, render_mode)
                        ),
                    );
                    if trait_.is_some() {
                        // Anchors are only used on trait impls.
                        write_str(w, format_args!("<a href=\"#{id}\" class=\"anchor\">§</a>"));
                    }
                    write_str(
                        w,
                        format_args!(
                            "<h4 class=\"code-header\">{}</h4></section>",
                            assoc_const(
                                item,
                                generics,
                                ty,
                                AssocConstValue::None,
                                link.anchor(if trait_.is_some() { &source_id } else { &id }),
                                0,
                                cx,
                            )
                        ),
                    );
                }
                clean::ProvidedAssocConstItem(ci) | clean::ImplAssocConstItem(ci) => {
                    let source_id = format!("{item_type}.{name}");
                    let id = cx.derive_id(&source_id);
                    write_str(
                        w,
                        format_args!(
                            "<section id=\"{id}\" class=\"{item_type}{in_trait_class}\">\
                            {}",
                            render_rightside(cx, item, render_mode)
                        ),
                    );
                    if trait_.is_some() {
                        // Anchors are only used on trait impls.
                        write_str(w, format_args!("<a href=\"#{id}\" class=\"anchor\">§</a>"));
                    }
                    write_str(
                        w,
                        format_args!(
                            "<h4 class=\"code-header\">{}</h4></section>",
                            assoc_const(
                                item,
                                &ci.generics,
                                &ci.type_,
                                match item.kind {
                                    clean::ProvidedAssocConstItem(_) =>
                                        AssocConstValue::TraitDefault(&ci.kind),
                                    clean::ImplAssocConstItem(_) => AssocConstValue::Impl(&ci.kind),
                                    _ => unreachable!(),
                                },
                                link.anchor(if trait_.is_some() { &source_id } else { &id }),
                                0,
                                cx,
                            )
                        ),
                    );
                }
                clean::RequiredAssocTypeItem(generics, bounds) => {
                    let source_id = format!("{item_type}.{name}");
                    let id = cx.derive_id(&source_id);
                    write_str(
                        w,
                        format_args!(
                            "<section id=\"{id}\" class=\"{item_type}{in_trait_class}\">\
                            {}",
                            render_rightside(cx, item, render_mode)
                        ),
                    );
                    if trait_.is_some() {
                        // Anchors are only used on trait impls.
                        write_str(w, format_args!("<a href=\"#{id}\" class=\"anchor\">§</a>"));
                    }
                    write_str(
                        w,
                        format_args!(
                            "<h4 class=\"code-header\">{}</h4></section>",
                            assoc_type(
                                item,
                                generics,
                                bounds,
                                None,
                                link.anchor(if trait_.is_some() { &source_id } else { &id }),
                                0,
                                cx,
                            )
                        ),
                    );
                }
                clean::AssocTypeItem(tydef, _bounds) => {
                    let source_id = format!("{item_type}.{name}");
                    let id = cx.derive_id(&source_id);
                    write_str(
                        w,
                        format_args!(
                            "<section id=\"{id}\" class=\"{item_type}{in_trait_class}\">\
                            {}",
                            render_rightside(cx, item, render_mode)
                        ),
                    );
                    if trait_.is_some() {
                        // Anchors are only used on trait impls.
                        write_str(w, format_args!("<a href=\"#{id}\" class=\"anchor\">§</a>"));
                    }
                    write_str(
                        w,
                        format_args!(
                            "<h4 class=\"code-header\">{}</h4></section>",
                            assoc_type(
                                item,
                                &tydef.generics,
                                &[], // intentionally leaving out bounds
                                Some(tydef.item_type.as_ref().unwrap_or(&tydef.type_)),
                                link.anchor(if trait_.is_some() { &source_id } else { &id }),
                                0,
                                cx,
                            )
                        ),
                    );
                }
                clean::StrippedItem(..) => return,
                _ => panic!("can't make docs for trait item with name {:?}", item.name),
            }

            w.push_str(&info_buffer);
            if toggled {
                w.push_str("</summary>");
                w.push_str(&doc_buffer);
                w.push_str("</details>");
            }
        }

        let mut impl_items = String::new();
        let mut default_impl_items = String::new();
        let impl_ = i.inner_impl();

        // Impl items are grouped by kinds:
        //
        // 1. Constants
        // 2. Types
        // 3. Functions
        //
        // This order is because you can have associated constants used in associated types (like array
        // length), and both in associcated functions. So with this order, when reading from top to
        // bottom, you should see items definitions before they're actually used most of the time.
        let mut assoc_types = Vec::new();
        let mut methods = Vec::new();

        if !impl_.is_negative_trait_impl() {
            for trait_item in &impl_.items {
                match trait_item.kind {
                    clean::MethodItem(..) | clean::RequiredMethodItem(_) => {
                        methods.push(trait_item)
                    }
                    clean::RequiredAssocTypeItem(..) | clean::AssocTypeItem(..) => {
                        assoc_types.push(trait_item)
                    }
                    clean::RequiredAssocConstItem(..)
                    | clean::ProvidedAssocConstItem(_)
                    | clean::ImplAssocConstItem(_) => {
                        // We render it directly since they're supposed to come first.
                        doc_impl_item(
                            &mut default_impl_items,
                            &mut impl_items,
                            cx,
                            trait_item,
                            if trait_.is_some() { &i.impl_item } else { parent },
                            link,
                            render_mode,
                            false,
                            trait_,
                            rendering_params,
                        );
                    }
                    _ => {}
                }
            }

            for assoc_type in assoc_types {
                doc_impl_item(
                    &mut default_impl_items,
                    &mut impl_items,
                    cx,
                    assoc_type,
                    if trait_.is_some() { &i.impl_item } else { parent },
                    link,
                    render_mode,
                    false,
                    trait_,
                    rendering_params,
                );
            }
            for method in methods {
                doc_impl_item(
                    &mut default_impl_items,
                    &mut impl_items,
                    cx,
                    method,
                    if trait_.is_some() { &i.impl_item } else { parent },
                    link,
                    render_mode,
                    false,
                    trait_,
                    rendering_params,
                );
            }
        }

        fn render_default_items(
            boring: &mut String,
            interesting: &mut String,
            cx: &Context<'_>,
            t: &clean::Trait,
            i: &clean::Impl,
            parent: &clean::Item,
            render_mode: RenderMode,
            rendering_params: ImplRenderingParameters,
        ) {
            for trait_item in &t.items {
                // Skip over any default trait items that are impossible to reference
                // (e.g. if it has a `Self: Sized` bound on an unsized type).
                if let Some(impl_def_id) = parent.item_id.as_def_id()
                    && let Some(trait_item_def_id) = trait_item.item_id.as_def_id()
                    && cx.tcx().is_impossible_associated_item((impl_def_id, trait_item_def_id))
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
            if let Some(t) = trait_
                && !impl_.is_negative_trait_impl()
            {
                render_default_items(
                    &mut default_impl_items,
                    &mut impl_items,
                    cx,
                    t,
                    impl_,
                    &i.impl_item,
                    render_mode,
                    rendering_params,
                );
            }
        }
        if render_mode == RenderMode::Normal {
            let toggled = !(impl_items.is_empty() && default_impl_items.is_empty());
            if toggled {
                close_tags.push("</details>");
                write!(
                    w,
                    "<details class=\"toggle implementors-toggle\"{}>\
                        <summary>",
                    if rendering_params.toggle_open_by_default { " open" } else { "" }
                )?;
            }

            let (before_dox, after_dox) = i
                .impl_item
                .opt_doc_value()
                .map(|dox| {
                    Markdown {
                        content: &dox,
                        links: &i.impl_item.links(cx),
                        ids: &mut cx.id_map.borrow_mut(),
                        error_codes: cx.shared.codes,
                        edition: cx.shared.edition(),
                        playground: &cx.shared.playground,
                        heading_offset: HeadingOffset::H4,
                    }
                    .split_summary_and_content()
                })
                .unwrap_or((None, None));
            write!(
                w,
                "{}",
                render_impl_summary(
                    cx,
                    i,
                    parent,
                    rendering_params.show_def_docs,
                    use_absolute,
                    aliases,
                    before_dox.as_deref(),
                )
            )?;
            if toggled {
                w.write_str("</summary>")?;
            }

            if before_dox.is_some() {
                if trait_.is_none() && impl_.items.is_empty() {
                    w.write_str(
                        "<div class=\"item-info\">\
                         <div class=\"stab empty-impl\">This impl block contains no items.</div>\
                     </div>",
                    )?;
                }
                if let Some(after_dox) = after_dox {
                    write!(w, "<div class=\"docblock\">{after_dox}</div>")?;
                }
            }
            if !default_impl_items.is_empty() || !impl_items.is_empty() {
                w.write_str("<div class=\"impl-items\">")?;
                close_tags.push("</div>");
            }
        }
        if !default_impl_items.is_empty() || !impl_items.is_empty() {
            w.write_str(&default_impl_items)?;
            w.write_str(&impl_items)?;
        }
        for tag in close_tags.into_iter().rev() {
            w.write_str(tag)?;
        }
        Ok(())
    })
}

// Render the items that appear on the right side of methods, impls, and
// associated types. For example "1.0.0 (const: 1.39.0) · source".
fn render_rightside(
    cx: &Context<'_>,
    item: &clean::Item,
    render_mode: RenderMode,
) -> impl fmt::Display {
    let tcx = cx.tcx();

    fmt::from_fn(move |w| {
        // FIXME: Once https://github.com/rust-lang/rust/issues/67792 is implemented, we can remove
        // this condition.
        let const_stability = match render_mode {
            RenderMode::Normal => item.const_stability(tcx),
            RenderMode::ForDeref { .. } => None,
        };
        let src_href = cx.src_href(item);
        let stability = render_stability_since_raw_with_extra(
            item.stable_since(tcx),
            const_stability,
            if src_href.is_some() { "" } else { " rightside" },
        );

        match (stability, src_href) {
            (Some(stability), Some(link)) => {
                write!(
                    w,
                    "<span class=\"rightside\">{stability} · <a class=\"src\" href=\"{link}\">Source</a></span>",
                )
            }
            (Some(stability), None) => {
                write!(w, "{stability}")
            }
            (None, Some(link)) => {
                write!(w, "<a class=\"src rightside\" href=\"{link}\">Source</a>")
            }
            (None, None) => Ok(()),
        }
    })
}

pub(crate) fn render_impl_summary(
    cx: &Context<'_>,
    i: &Impl,
    parent: &clean::Item,
    show_def_docs: bool,
    use_absolute: Option<bool>,
    // This argument is used to reference same type with different paths to avoid duplication
    // in documentation pages for trait with automatic implementations like "Send" and "Sync".
    aliases: &[String],
    doc: Option<&str>,
) -> impl fmt::Display {
    fmt::from_fn(move |w| {
        let inner_impl = i.inner_impl();
        let id = cx.derive_id(get_id_for_impl(cx.tcx(), i.impl_item.item_id));
        let aliases = (!aliases.is_empty())
            .then_some(fmt::from_fn(|f| {
                write!(f, " data-aliases=\"{}\"", fmt::from_fn(|f| aliases.iter().joined(",", f)))
            }))
            .maybe_display();
        write!(
            w,
            "<section id=\"{id}\" class=\"impl\"{aliases}>\
                {}\
                <a href=\"#{id}\" class=\"anchor\">§</a>\
                <h3 class=\"code-header\">",
            render_rightside(cx, &i.impl_item, RenderMode::Normal)
        )?;

        if let Some(use_absolute) = use_absolute {
            write!(w, "{}", inner_impl.print(use_absolute, cx))?;
            if show_def_docs {
                for it in &inner_impl.items {
                    if let clean::AssocTypeItem(ref tydef, ref _bounds) = it.kind {
                        write!(
                            w,
                            "<div class=\"where\">  {};</div>",
                            assoc_type(
                                it,
                                &tydef.generics,
                                &[], // intentionally leaving out bounds
                                Some(&tydef.type_),
                                AssocItemLink::Anchor(None),
                                0,
                                cx,
                            )
                        )?;
                    }
                }
            }
        } else {
            write!(w, "{}", inner_impl.print(false, cx))?;
        }
        w.write_str("</h3>")?;

        let is_trait = inner_impl.trait_.is_some();
        if is_trait && let Some(portability) = portability(&i.impl_item, Some(parent)) {
            write!(
                w,
                "<span class=\"item-info\">\
                    <div class=\"stab portability\">{portability}</div>\
                </span>",
            )?;
        }

        if let Some(doc) = doc {
            write!(w, "<div class=\"docblock\">{doc}</div>")?;
        }

        w.write_str("</section>")
    })
}

pub(crate) fn small_url_encode(s: String) -> String {
    // These characters don't need to be escaped in a URI.
    // See https://url.spec.whatwg.org/#query-percent-encode-set
    // and https://url.spec.whatwg.org/#urlencoded-parsing
    // and https://url.spec.whatwg.org/#url-code-points
    fn dont_escape(c: u8) -> bool {
        c.is_ascii_alphanumeric()
            || c == b'-'
            || c == b'_'
            || c == b'.'
            || c == b','
            || c == b'~'
            || c == b'!'
            || c == b'\''
            || c == b'('
            || c == b')'
            || c == b'*'
            || c == b'/'
            || c == b';'
            || c == b':'
            || c == b'?'
            // As described in urlencoded-parsing, the
            // first `=` is the one that separates key from
            // value. Following `=`s are part of the value.
            || c == b'='
    }
    let mut st = String::new();
    let mut last_match = 0;
    for (idx, b) in s.bytes().enumerate() {
        if dont_escape(b) {
            continue;
        }

        if last_match != idx {
            // Invariant: `idx` must be the first byte in a character at this point.
            st += &s[last_match..idx];
        }
        if b == b' ' {
            // URL queries are decoded with + replaced with SP.
            // While the same is not true for hashes, rustdoc only needs to be
            // consistent with itself when encoding them.
            st += "+";
        } else {
            write!(st, "%{b:02X}").unwrap();
        }
        // Invariant: if the current byte is not at the start of a multi-byte character,
        // we need to get down here so that when the next turn of the loop comes around,
        // last_match winds up equalling idx.
        //
        // In other words, dont_escape must always return `false` in multi-byte character.
        last_match = idx + 1;
    }

    if last_match != 0 {
        st += &s[last_match..];
        st
    } else {
        s
    }
}

fn get_id_for_impl(tcx: TyCtxt<'_>, impl_id: ItemId) -> String {
    use rustc_middle::ty::print::with_forced_trimmed_paths;
    let (type_, trait_) = match impl_id {
        ItemId::Auto { trait_, for_ } => {
            let ty = tcx.type_of(for_).skip_binder();
            (ty, Some(ty::TraitRef::new(tcx, trait_, [ty])))
        }
        ItemId::Blanket { impl_id, .. } | ItemId::DefId(impl_id) => {
            match tcx.impl_subject(impl_id).skip_binder() {
                ty::ImplSubject::Trait(trait_ref) => {
                    (trait_ref.args[0].expect_ty(), Some(trait_ref))
                }
                ty::ImplSubject::Inherent(ty) => (ty, None),
            }
        }
    };
    with_forced_trimmed_paths!(small_url_encode(if let Some(trait_) = trait_ {
        format!("impl-{trait_}-for-{type_}", trait_ = trait_.print_only_trait_path())
    } else {
        format!("impl-{type_}")
    }))
}

fn extract_for_impl_name(item: &clean::Item, cx: &Context<'_>) -> Option<(String, String)> {
    match item.kind {
        clean::ItemKind::ImplItem(ref i) if i.trait_.is_some() => {
            // Alternative format produces no URLs,
            // so this parameter does nothing.
            Some((format!("{:#}", i.for_.print(cx)), get_id_for_impl(cx.tcx(), item.item_id)))
        }
        _ => None,
    }
}

/// Returns the list of implementations for the primitive reference type, filtering out any
/// implementations that are on concrete or partially generic types, only keeping implementations
/// of the form `impl<T> Trait for &T`.
pub(crate) fn get_filtered_impls_for_reference<'a>(
    shared: &'a SharedContext<'_>,
    it: &clean::Item,
) -> (Vec<&'a Impl>, Vec<&'a Impl>, Vec<&'a Impl>) {
    let def_id = it.item_id.expect_def_id();
    // If the reference primitive is somehow not defined, exit early.
    let Some(v) = shared.cache.impls.get(&def_id) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
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
    TypeAliases,
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
    AttributeMacros,
    DeriveMacros,
    TraitAliases,
}

impl ItemSection {
    const ALL: &'static [Self] = {
        use ItemSection::*;
        // NOTE: The order here affects the order in the UI.
        // Keep this synchronized with addSidebarItems in main.js
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
            TypeAliases,
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
            Self::TypeAliases => "types",
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
            Self::TypeAliases => "Type Aliases",
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
        ItemType::TypeAlias => ItemSection::TypeAliases,
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
        ItemType::ProcAttribute => ItemSection::AttributeMacros,
        ItemType::ProcDerive => ItemSection::DeriveMacros,
        ItemType::TraitAlias => ItemSection::TraitAliases,
    }
}

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
            out.push(join_with_double_colon(path));
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
                if let Some(trait_) = trait_ {
                    process_path(trait_.def_id());
                }
            }
            _ => {}
        }
    }
    out
}

const MAX_FULL_EXAMPLES: usize = 5;
const NUM_VISIBLE_LINES: usize = 10;

/// Generates the HTML for example call locations generated via the --scrape-examples flag.
fn render_call_locations<W: fmt::Write>(mut w: W, cx: &Context<'_>, item: &clean::Item) {
    let tcx = cx.tcx();
    let def_id = item.item_id.expect_def_id();
    let key = tcx.def_path_hash(def_id);
    let Some(call_locations) = cx.shared.call_locations.get(&key) else { return };

    // Generate a unique ID so users can link to this section for a given method
    let id = cx.derive_id("scraped-examples");
    write!(
        &mut w,
        "<div class=\"docblock scraped-example-list\">\
          <span></span>\
          <h5 id=\"{id}\">\
             <a href=\"#{id}\">Examples found in repository</a>\
             <a class=\"scrape-help\" href=\"{root_path}scrape-examples-help.html\">?</a>\
          </h5>",
        root_path = cx.root_path(),
        id = id
    )
    .unwrap();

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
        let url = format!("{}{}#{anchor}", cx.root_path(), call_data.url);
        (url, title)
    };

    // Generate the HTML for a single example, being the title and code block
    let write_example = |w: &mut W, (path, call_data): (&PathBuf, &CallData)| -> bool {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) => {
                let span = item.span(tcx).map_or(DUMMY_SP, |span| span.inner());
                tcx.dcx().span_err(span, format!("failed to read file {}: {err}", path.display()));
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

        // Look for the example file in the source map if it exists, otherwise return a dummy span
        let file_span = (|| {
            let source_map = tcx.sess.source_map();
            let crate_src = tcx.sess.local_crate_source_file()?.into_local_path()?;
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
        .unwrap_or(DUMMY_SP);

        let mut decoration_info = FxIndexMap::default();
        decoration_info.insert("highlight focus", vec![byte_ranges.remove(0)]);
        decoration_info.insert("highlight", byte_ranges);

        sources::print_src(
            w,
            contents_subset,
            file_span,
            cx,
            &cx.root_path(),
            &highlight::DecorationInfo(decoration_info),
            &sources::SourceContext::Embedded(sources::ScrapedInfo {
                needs_expansion,
                offset: line_min,
                name: &call_data.display_name,
                url: init_url,
                title: init_title,
                locations: locations_encoded,
            }),
        );

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
    let write_and_skip_failure = |w: &mut W, it: &mut Peekable<_>| {
        for example in it.by_ref() {
            if write_example(&mut *w, example) {
                break;
            }
        }
    };

    // Write just one example that's visible by default in the method's description.
    write_and_skip_failure(&mut w, &mut it);

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
        )
        .unwrap();

        // Only generate inline code for MAX_FULL_EXAMPLES number of examples. Otherwise we could
        // make the page arbitrarily huge!
        for _ in 0..MAX_FULL_EXAMPLES {
            write_and_skip_failure(&mut w, &mut it);
        }

        // For the remaining examples, generate a <ul> containing links to the source files.
        if it.peek().is_some() {
            w.write_str(
                r#"<div class="example-links">Additional examples can be found in:<br><ul>"#,
            )
            .unwrap();
            it.for_each(|(_, call_data)| {
                let (url, _) = link_to_loc(call_data, &call_data.locations[0]);
                write!(
                    w,
                    r#"<li><a href="{url}">{name}</a></li>"#,
                    url = url,
                    name = call_data.display_name
                )
                .unwrap();
            });
            w.write_str("</ul></div>").unwrap();
        }

        w.write_str("</div></details>").unwrap();
    }

    w.write_str("</div>").unwrap();
}
