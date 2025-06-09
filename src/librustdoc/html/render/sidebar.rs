use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

use askama::Template;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{DefIdMap, DefIdSet};
use rustc_middle::ty::{self, TyCtxt};
use tracing::debug;

use super::{Context, ItemSection, item_ty_to_section};
use crate::clean;
use crate::formats::Impl;
use crate::formats::item_type::ItemType;
use crate::html::markdown::{IdMap, MarkdownWithToc};
use crate::html::render::print_item::compare_names;

#[derive(Clone, Copy)]
pub(crate) enum ModuleLike {
    Module,
    Crate,
}

impl ModuleLike {
    pub(crate) fn is_crate(self) -> bool {
        matches!(self, ModuleLike::Crate)
    }
}
impl<'a> From<&'a clean::Item> for ModuleLike {
    fn from(it: &'a clean::Item) -> ModuleLike {
        if it.is_crate() { ModuleLike::Crate } else { ModuleLike::Module }
    }
}

#[derive(Template)]
#[template(path = "sidebar.html")]
pub(super) struct Sidebar<'a> {
    pub(super) title_prefix: &'static str,
    pub(super) title: &'a str,
    pub(super) is_crate: bool,
    pub(super) parent_is_crate: bool,
    pub(super) is_mod: bool,
    pub(super) blocks: Vec<LinkBlock<'a>>,
    pub(super) path: String,
}

impl Sidebar<'_> {
    /// Only create a `<section>` if there are any blocks
    /// which should actually be rendered.
    pub fn should_render_blocks(&self) -> bool {
        self.blocks.iter().any(LinkBlock::should_render)
    }
}

/// A sidebar section such as 'Methods'.
pub(crate) struct LinkBlock<'a> {
    /// The name of this section, e.g. 'Methods'
    /// as well as the link to it, e.g. `#implementations`.
    /// Will be rendered inside an `<h3>` tag
    heading: Link<'a>,
    class: &'static str,
    links: Vec<Link<'a>>,
    /// Render the heading even if there are no links
    force_render: bool,
}

impl<'a> LinkBlock<'a> {
    pub fn new(heading: Link<'a>, class: &'static str, links: Vec<Link<'a>>) -> Self {
        Self { heading, links, class, force_render: false }
    }

    pub fn forced(heading: Link<'a>, class: &'static str) -> Self {
        Self { heading, links: vec![], class, force_render: true }
    }

    pub fn should_render(&self) -> bool {
        self.force_render || !self.links.is_empty()
    }
}

/// A link to an item. Content should not be escaped.
#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) struct Link<'a> {
    /// The content for the anchor tag and title attr
    name: Cow<'a, str>,
    /// The content for the anchor tag (if different from name)
    name_html: Option<Cow<'a, str>>,
    /// The id of an anchor within the page (without a `#` prefix)
    href: Cow<'a, str>,
    /// Nested list of links (used only in top-toc)
    children: Vec<Link<'a>>,
}

impl Ord for Link<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match compare_names(&self.name, &other.name) {
            Ordering::Equal => {}
            result => return result,
        }
        (&self.name_html, &self.href, &self.children).cmp(&(
            &other.name_html,
            &other.href,
            &other.children,
        ))
    }
}

impl PartialOrd for Link<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Link<'a> {
    pub fn new(href: impl Into<Cow<'a, str>>, name: impl Into<Cow<'a, str>>) -> Self {
        Self { href: href.into(), name: name.into(), children: vec![], name_html: None }
    }
    pub fn empty() -> Link<'static> {
        Link::new("", "")
    }
}

pub(crate) mod filters {
    use std::fmt::{self, Display};

    use askama::filters::Safe;

    use crate::html::escape::EscapeBodyTextWithWbr;
    pub(crate) fn wrapped<T, V: askama::Values>(v: T, _: V) -> askama::Result<Safe<impl Display>>
    where
        T: Display,
    {
        let string = v.to_string();
        Ok(Safe(fmt::from_fn(move |f| EscapeBodyTextWithWbr(&string).fmt(f))))
    }
}

pub(super) fn print_sidebar(
    cx: &Context<'_>,
    it: &clean::Item,
    mut buffer: impl fmt::Write,
) -> fmt::Result {
    let mut ids = IdMap::new();
    let mut blocks: Vec<LinkBlock<'_>> = docblock_toc(cx, it, &mut ids).into_iter().collect();
    let deref_id_map = cx.deref_id_map.borrow();
    match it.kind {
        clean::StructItem(ref s) => sidebar_struct(cx, it, s, &mut blocks, &deref_id_map),
        clean::TraitItem(ref t) => sidebar_trait(cx, it, t, &mut blocks, &deref_id_map),
        clean::PrimitiveItem(_) => sidebar_primitive(cx, it, &mut blocks, &deref_id_map),
        clean::UnionItem(ref u) => sidebar_union(cx, it, u, &mut blocks, &deref_id_map),
        clean::EnumItem(ref e) => sidebar_enum(cx, it, e, &mut blocks, &deref_id_map),
        clean::TypeAliasItem(ref t) => sidebar_type_alias(cx, it, t, &mut blocks, &deref_id_map),
        clean::ModuleItem(ref m) => {
            blocks.push(sidebar_module(&m.items, &mut ids, ModuleLike::from(it)))
        }
        clean::ForeignTypeItem => sidebar_foreign_type(cx, it, &mut blocks, &deref_id_map),
        _ => {}
    }
    // The sidebar is designed to display sibling functions, modules and
    // other miscellaneous information. since there are lots of sibling
    // items (and that causes quadratic growth in large modules),
    // we refactor common parts into a shared JavaScript file per module.
    // still, we don't move everything into JS because we want to preserve
    // as much HTML as possible in order to allow non-JS-enabled browsers
    // to navigate the documentation (though slightly inefficiently).
    //
    // crate title is displayed as part of logo lockup
    let (title_prefix, title) = if !blocks.is_empty() && !it.is_crate() {
        (
            match it.kind {
                clean::ModuleItem(..) => "Module ",
                _ => "",
            },
            it.name.as_ref().unwrap().as_str(),
        )
    } else {
        ("", "")
    };
    // need to show parent path header if:
    //   - it's a child module, instead of the crate root
    //   - there's a sidebar section for the item itself
    //
    // otherwise, the parent path header is redundant with the big crate
    // branding area at the top of the sidebar
    let sidebar_path =
        if it.is_mod() { &cx.current[..cx.current.len() - 1] } else { &cx.current[..] };
    let path: String = if sidebar_path.len() > 1 || !title.is_empty() {
        let path = sidebar_path.iter().map(|s| s.as_str()).intersperse("::").collect();
        if sidebar_path.len() == 1 { format!("crate {path}") } else { path }
    } else {
        "".into()
    };
    let sidebar = Sidebar {
        title_prefix,
        title,
        is_mod: it.is_mod(),
        is_crate: it.is_crate(),
        parent_is_crate: sidebar_path.len() == 1,
        blocks,
        path,
    };
    sidebar.render_into(&mut buffer)?;
    Ok(())
}

fn get_struct_fields_name<'a>(fields: &'a [clean::Item]) -> Vec<Link<'a>> {
    let mut fields = fields
        .iter()
        .filter(|f| matches!(f.kind, clean::StructFieldItem(..)))
        .filter_map(|f| {
            f.name.as_ref().map(|name| Link::new(format!("structfield.{name}"), name.as_str()))
        })
        .collect::<Vec<Link<'a>>>();
    fields.sort();
    fields
}

fn docblock_toc<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    ids: &mut IdMap,
) -> Option<LinkBlock<'a>> {
    let (toc, _) = MarkdownWithToc {
        content: &it.doc_value(),
        links: &it.links(cx),
        ids,
        error_codes: cx.shared.codes,
        edition: cx.shared.edition(),
        playground: &cx.shared.playground,
    }
    .into_parts();
    let links: Vec<Link<'_>> = toc
        .entries
        .into_iter()
        .map(|entry| {
            Link {
                name_html: if entry.html == entry.name { None } else { Some(entry.html.into()) },
                name: entry.name.into(),
                href: entry.id.into(),
                children: entry
                    .children
                    .entries
                    .into_iter()
                    .map(|entry| Link {
                        name_html: if entry.html == entry.name {
                            None
                        } else {
                            Some(entry.html.into())
                        },
                        name: entry.name.into(),
                        href: entry.id.into(),
                        // Only a single level of nesting is shown here.
                        // Going the full six could break the layout,
                        // so we have to cut it off somewhere.
                        children: vec![],
                    })
                    .collect(),
            }
        })
        .collect();
    if links.is_empty() {
        None
    } else {
        Some(LinkBlock::new(Link::new("", "Sections"), "top-toc", links))
    }
}

fn sidebar_struct<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    s: &'a clean::Struct,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    let fields = get_struct_fields_name(&s.fields);
    let field_name = match s.ctor_kind {
        Some(CtorKind::Fn) => Some("Tuple Fields"),
        None => Some("Fields"),
        _ => None,
    };
    if let Some(name) = field_name {
        items.push(LinkBlock::new(Link::new("fields", name), "structfield", fields));
    }
    sidebar_assoc_items(cx, it, items, deref_id_map);
}

fn sidebar_trait<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    t: &'a clean::Trait,
    blocks: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    fn filter_items<'a>(
        items: &'a [clean::Item],
        filt: impl Fn(&clean::Item) -> bool,
        ty: &str,
    ) -> Vec<Link<'a>> {
        let mut res = items
            .iter()
            .filter_map(|m: &clean::Item| match m.name {
                Some(ref name) if filt(m) => Some(Link::new(format!("{ty}.{name}"), name.as_str())),
                _ => None,
            })
            .collect::<Vec<Link<'a>>>();
        res.sort();
        res
    }

    let req_assoc = filter_items(&t.items, |m| m.is_required_associated_type(), "associatedtype");
    let prov_assoc = filter_items(&t.items, |m| m.is_associated_type(), "associatedtype");
    let req_assoc_const =
        filter_items(&t.items, |m| m.is_required_associated_const(), "associatedconstant");
    let prov_assoc_const =
        filter_items(&t.items, |m| m.is_associated_const(), "associatedconstant");
    let req_method = filter_items(&t.items, |m| m.is_ty_method(), "tymethod");
    let prov_method = filter_items(&t.items, |m| m.is_method(), "method");
    let mut foreign_impls = vec![];
    if let Some(implementors) = cx.cache().implementors.get(&it.item_id.expect_def_id()) {
        foreign_impls.extend(
            implementors
                .iter()
                .filter(|i| !i.is_on_local_type(cx))
                .filter_map(|i| super::extract_for_impl_name(&i.impl_item, cx))
                .map(|(name, id)| Link::new(id, name)),
        );
        foreign_impls.sort();
    }

    blocks.extend(
        [
            ("required-associated-consts", "Required Associated Constants", req_assoc_const),
            ("provided-associated-consts", "Provided Associated Constants", prov_assoc_const),
            ("required-associated-types", "Required Associated Types", req_assoc),
            ("provided-associated-types", "Provided Associated Types", prov_assoc),
            ("required-methods", "Required Methods", req_method),
            ("provided-methods", "Provided Methods", prov_method),
            ("foreign-impls", "Implementations on Foreign Types", foreign_impls),
        ]
        .into_iter()
        .map(|(id, title, items)| LinkBlock::new(Link::new(id, title), "", items)),
    );
    sidebar_assoc_items(cx, it, blocks, deref_id_map);

    if !t.is_dyn_compatible(cx.tcx()) {
        blocks.push(LinkBlock::forced(
            Link::new("dyn-compatibility", "Dyn Compatibility"),
            "dyn-compatibility-note",
        ));
    }

    blocks.push(LinkBlock::forced(Link::new("implementors", "Implementors"), "impl"));
    if t.is_auto(cx.tcx()) {
        blocks.push(LinkBlock::forced(
            Link::new("synthetic-implementors", "Auto Implementors"),
            "impl-auto",
        ));
    }
}

fn sidebar_primitive<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    if it.name.map(|n| n.as_str() != "reference").unwrap_or(false) {
        sidebar_assoc_items(cx, it, items, deref_id_map);
    } else {
        let (concrete, synthetic, blanket_impl) =
            super::get_filtered_impls_for_reference(&cx.shared, it);

        sidebar_render_assoc_items(cx, &mut IdMap::new(), concrete, synthetic, blanket_impl, items);
    }
}

fn sidebar_type_alias<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    t: &'a clean::TypeAlias,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    if let Some(inner_type) = &t.inner_type {
        items.push(LinkBlock::forced(Link::new("aliased-type", "Aliased Type"), "type"));
        match inner_type {
            clean::TypeAliasInnerType::Enum { variants, is_non_exhaustive: _ } => {
                let mut variants = variants
                    .iter()
                    .filter(|i| !i.is_stripped())
                    .filter_map(|v| v.name)
                    .map(|name| Link::new(format!("variant.{name}"), name.to_string()))
                    .collect::<Vec<_>>();
                variants.sort_unstable();

                items.push(LinkBlock::new(Link::new("variants", "Variants"), "variant", variants));
            }
            clean::TypeAliasInnerType::Union { fields }
            | clean::TypeAliasInnerType::Struct { ctor_kind: _, fields } => {
                let fields = get_struct_fields_name(fields);
                items.push(LinkBlock::new(Link::new("fields", "Fields"), "field", fields));
            }
        }
    }
    sidebar_assoc_items(cx, it, items, deref_id_map);
}

fn sidebar_union<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    u: &'a clean::Union,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    let fields = get_struct_fields_name(&u.fields);
    items.push(LinkBlock::new(Link::new("fields", "Fields"), "structfield", fields));
    sidebar_assoc_items(cx, it, items, deref_id_map);
}

/// Adds trait implementations into the blocks of links
fn sidebar_assoc_items<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    links: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    let did = it.item_id.expect_def_id();
    let cache = cx.cache();

    let mut assoc_consts = Vec::new();
    let mut assoc_types = Vec::new();
    let mut methods = Vec::new();
    if let Some(v) = cache.impls.get(&did) {
        let mut used_links = FxHashSet::default();
        let mut id_map = IdMap::new();

        {
            let used_links_bor = &mut used_links;
            for impl_ in v.iter().map(|i| i.inner_impl()).filter(|i| i.trait_.is_none()) {
                assoc_consts.extend(get_associated_constants(impl_, used_links_bor));
                assoc_types.extend(get_associated_types(impl_, used_links_bor));
                methods.extend(get_methods(impl_, false, used_links_bor, false, cx.tcx()));
            }
            // We want links' order to be reproducible so we don't use unstable sort.
            assoc_consts.sort();
            assoc_types.sort();
            methods.sort();
        }

        let mut blocks = vec![
            LinkBlock::new(
                Link::new("implementations", "Associated Constants"),
                "associatedconstant",
                assoc_consts,
            ),
            LinkBlock::new(
                Link::new("implementations", "Associated Types"),
                "associatedtype",
                assoc_types,
            ),
            LinkBlock::new(Link::new("implementations", "Methods"), "method", methods),
        ];

        if v.iter().any(|i| i.inner_impl().trait_.is_some()) {
            if let Some(impl_) =
                v.iter().find(|i| i.trait_did() == cx.tcx().lang_items().deref_trait())
            {
                let mut derefs = DefIdSet::default();
                derefs.insert(did);
                sidebar_deref_methods(
                    cx,
                    &mut blocks,
                    impl_,
                    v,
                    &mut derefs,
                    &mut used_links,
                    deref_id_map,
                );
            }

            let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
                v.iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_auto());
            let (blanket_impl, concrete): (Vec<&Impl>, Vec<&Impl>) =
                concrete.into_iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_blanket());

            sidebar_render_assoc_items(
                cx,
                &mut id_map,
                concrete,
                synthetic,
                blanket_impl,
                &mut blocks,
            );
        }

        links.append(&mut blocks);
    }
}

fn sidebar_deref_methods<'a>(
    cx: &'a Context<'_>,
    out: &mut Vec<LinkBlock<'a>>,
    impl_: &Impl,
    v: &[Impl],
    derefs: &mut DefIdSet,
    used_links: &mut FxHashSet<String>,
    deref_id_map: &'a DefIdMap<String>,
) {
    let c = cx.cache();

    debug!("found Deref: {impl_:?}");
    if let Some((target, real_target)) =
        impl_.inner_impl().items.iter().find_map(|item| match item.kind {
            clean::AssocTypeItem(box ref t, _) => Some(match *t {
                clean::TypeAlias { item_type: Some(ref type_), .. } => (type_, &t.type_),
                _ => (&t.type_, &t.type_),
            }),
            _ => None,
        })
    {
        debug!("found target, real_target: {target:?} {real_target:?}");
        if let Some(did) = target.def_id(c) &&
            let Some(type_did) = impl_.inner_impl().for_.def_id(c) &&
            // `impl Deref<Target = S> for S`
            (did == type_did || !derefs.insert(did))
        {
            // Avoid infinite cycles
            return;
        }
        let deref_mut = v.iter().any(|i| i.trait_did() == cx.tcx().lang_items().deref_mut_trait());
        let inner_impl = target
            .def_id(c)
            .or_else(|| {
                target.primitive_type().and_then(|prim| c.primitive_locations.get(&prim).cloned())
            })
            .and_then(|did| c.impls.get(&did));
        if let Some(impls) = inner_impl {
            debug!("found inner_impl: {impls:?}");
            let mut ret = impls
                .iter()
                .filter(|i| {
                    i.inner_impl().trait_.is_none()
                        && real_target.is_doc_subtype_of(&i.inner_impl().for_, &c)
                })
                .flat_map(|i| get_methods(i.inner_impl(), true, used_links, deref_mut, cx.tcx()))
                .collect::<Vec<_>>();
            if !ret.is_empty() {
                let id = if let Some(target_def_id) = real_target.def_id(c) {
                    Cow::Borrowed(
                        deref_id_map
                            .get(&target_def_id)
                            .expect("Deref section without derived id")
                            .as_str(),
                    )
                } else {
                    Cow::Borrowed("deref-methods")
                };
                let title = format!(
                    "Methods from {:#}<Target={:#}>",
                    impl_.inner_impl().trait_.as_ref().unwrap().print(cx),
                    real_target.print(cx),
                );
                // We want links' order to be reproducible so we don't use unstable sort.
                ret.sort();
                out.push(LinkBlock::new(Link::new(id, title), "deref-methods", ret));
            }
        }

        // Recurse into any further impls that might exist for `target`
        if let Some(target_did) = target.def_id(c)
            && let Some(target_impls) = c.impls.get(&target_did)
            && let Some(target_deref_impl) = target_impls.iter().find(|i| {
                i.inner_impl()
                    .trait_
                    .as_ref()
                    .map(|t| Some(t.def_id()) == cx.tcx().lang_items().deref_trait())
                    .unwrap_or(false)
            })
        {
            sidebar_deref_methods(
                cx,
                out,
                target_deref_impl,
                target_impls,
                derefs,
                used_links,
                deref_id_map,
            );
        }
    }
}

fn sidebar_enum<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    e: &'a clean::Enum,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    let mut variants = e
        .non_stripped_variants()
        .filter_map(|v| v.name)
        .map(|name| Link::new(format!("variant.{name}"), name.to_string()))
        .collect::<Vec<_>>();
    variants.sort_unstable();

    items.push(LinkBlock::new(Link::new("variants", "Variants"), "variant", variants));
    sidebar_assoc_items(cx, it, items, deref_id_map);
}

pub(crate) fn sidebar_module_like(
    item_sections_in_use: FxHashSet<ItemSection>,
    ids: &mut IdMap,
    module_like: ModuleLike,
) -> LinkBlock<'static> {
    let item_sections: Vec<Link<'_>> = ItemSection::ALL
        .iter()
        .copied()
        .filter(|sec| item_sections_in_use.contains(sec))
        .map(|sec| Link::new(ids.derive(sec.id()), sec.name()))
        .collect();
    let header = if let Some(first_section) = item_sections.first() {
        Link::new(
            first_section.href.clone(),
            if module_like.is_crate() { "Crate Items" } else { "Module Items" },
        )
    } else {
        Link::empty()
    };
    LinkBlock::new(header, "", item_sections)
}

fn sidebar_module(
    items: &[clean::Item],
    ids: &mut IdMap,
    module_like: ModuleLike,
) -> LinkBlock<'static> {
    let item_sections_in_use: FxHashSet<_> = items
        .iter()
        .filter(|it| {
            !it.is_stripped()
                && it
                    .name
                    .or_else(|| {
                        if let clean::ImportItem(ref i) = it.kind
                            && let clean::ImportKind::Simple(s) = i.kind
                        {
                            Some(s)
                        } else {
                            None
                        }
                    })
                    .is_some()
        })
        .map(|it| item_ty_to_section(it.type_()))
        .collect();

    sidebar_module_like(item_sections_in_use, ids, module_like)
}

fn sidebar_foreign_type<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    items: &mut Vec<LinkBlock<'a>>,
    deref_id_map: &'a DefIdMap<String>,
) {
    sidebar_assoc_items(cx, it, items, deref_id_map);
}

/// Renders the trait implementations for this type
fn sidebar_render_assoc_items(
    cx: &Context<'_>,
    id_map: &mut IdMap,
    concrete: Vec<&Impl>,
    synthetic: Vec<&Impl>,
    blanket_impl: Vec<&Impl>,
    items: &mut Vec<LinkBlock<'_>>,
) {
    let format_impls = |impls: Vec<&Impl>, id_map: &mut IdMap| {
        let mut links = FxHashSet::default();

        let mut ret = impls
            .iter()
            .filter_map(|it| {
                let trait_ = it.inner_impl().trait_.as_ref()?;
                let encoded = id_map.derive(super::get_id_for_impl(cx.tcx(), it.impl_item.item_id));

                let prefix = match it.inner_impl().polarity {
                    ty::ImplPolarity::Positive | ty::ImplPolarity::Reservation => "",
                    ty::ImplPolarity::Negative => "!",
                };
                let generated = Link::new(encoded, format!("{prefix}{:#}", trait_.print(cx)));
                if links.insert(generated.clone()) { Some(generated) } else { None }
            })
            .collect::<Vec<Link<'static>>>();
        ret.sort();
        ret
    };

    let concrete = format_impls(concrete, id_map);
    let synthetic = format_impls(synthetic, id_map);
    let blanket = format_impls(blanket_impl, id_map);
    items.extend([
        LinkBlock::new(
            Link::new("trait-implementations", "Trait Implementations"),
            "trait-implementation",
            concrete,
        ),
        LinkBlock::new(
            Link::new("synthetic-implementations", "Auto Trait Implementations"),
            "synthetic-implementation",
            synthetic,
        ),
        LinkBlock::new(
            Link::new("blanket-implementations", "Blanket Implementations"),
            "blanket-implementation",
            blanket,
        ),
    ]);
}

fn get_next_url(used_links: &mut FxHashSet<String>, url: String) -> String {
    if used_links.insert(url.clone()) {
        return url;
    }
    let mut add = 1;
    while !used_links.insert(format!("{url}-{add}")) {
        add += 1;
    }
    format!("{url}-{add}")
}

fn get_methods<'a>(
    i: &'a clean::Impl,
    for_deref: bool,
    used_links: &mut FxHashSet<String>,
    deref_mut: bool,
    tcx: TyCtxt<'_>,
) -> Vec<Link<'a>> {
    i.items
        .iter()
        .filter_map(|item| {
            if let Some(ref name) = item.name
                && item.is_method()
                && (!for_deref || super::should_render_item(item, deref_mut, tcx))
            {
                Some(Link::new(
                    get_next_url(used_links, format!("{typ}.{name}", typ = ItemType::Method)),
                    name.as_str(),
                ))
            } else {
                None
            }
        })
        .collect()
}

fn get_associated_constants<'a>(
    i: &'a clean::Impl,
    used_links: &mut FxHashSet<String>,
) -> Vec<Link<'a>> {
    i.items
        .iter()
        .filter_map(|item| {
            if let Some(ref name) = item.name
                && item.is_associated_const()
            {
                Some(Link::new(
                    get_next_url(used_links, format!("{typ}.{name}", typ = ItemType::AssocConst)),
                    name.as_str(),
                ))
            } else {
                None
            }
        })
        .collect()
}

fn get_associated_types<'a>(
    i: &'a clean::Impl,
    used_links: &mut FxHashSet<String>,
) -> Vec<Link<'a>> {
    i.items
        .iter()
        .filter_map(|item| {
            if let Some(ref name) = item.name
                && item.is_associated_type()
            {
                Some(Link::new(
                    get_next_url(used_links, format!("{typ}.{name}", typ = ItemType::AssocType)),
                    name.as_str(),
                ))
            } else {
                None
            }
        })
        .collect()
}
