use std::{borrow::Cow, rc::Rc};

use askama::Template;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{def::CtorKind, def_id::DefIdSet};
use rustc_middle::ty::{self, TyCtxt};

use crate::{
    clean,
    formats::{item_type::ItemType, Impl},
    html::{format::Buffer, markdown::IdMap},
};

use super::{item_ty_to_section, Context, ItemSection};

#[derive(Template)]
#[template(path = "sidebar.html")]
pub(super) struct Sidebar<'a> {
    pub(super) title_prefix: &'static str,
    pub(super) title: &'a str,
    pub(super) is_crate: bool,
    pub(super) version: &'a str,
    pub(super) blocks: Vec<LinkBlock<'a>>,
    pub(super) path: String,
}

impl<'a> Sidebar<'a> {
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
    links: Vec<Link<'a>>,
    /// Render the heading even if there are no links
    force_render: bool,
}

impl<'a> LinkBlock<'a> {
    pub fn new(heading: Link<'a>, links: Vec<Link<'a>>) -> Self {
        Self { heading, links, force_render: false }
    }

    pub fn forced(heading: Link<'a>) -> Self {
        Self { heading, links: vec![], force_render: true }
    }

    pub fn should_render(&self) -> bool {
        self.force_render || !self.links.is_empty()
    }
}

/// A link to an item. Content should not be escaped.
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
pub(crate) struct Link<'a> {
    /// The content for the anchor tag
    name: Cow<'a, str>,
    /// The id of an anchor within the page (without a `#` prefix)
    href: Cow<'a, str>,
}

impl<'a> Link<'a> {
    pub fn new(href: impl Into<Cow<'a, str>>, name: impl Into<Cow<'a, str>>) -> Self {
        Self { href: href.into(), name: name.into() }
    }
    pub fn empty() -> Link<'static> {
        Link::new("", "")
    }
}

pub(super) fn print_sidebar(cx: &Context<'_>, it: &clean::Item, buffer: &mut Buffer) {
    let blocks: Vec<LinkBlock<'_>> = match *it.kind {
        clean::StructItem(ref s) => sidebar_struct(cx, it, s),
        clean::TraitItem(ref t) => sidebar_trait(cx, it, t),
        clean::PrimitiveItem(_) => sidebar_primitive(cx, it),
        clean::UnionItem(ref u) => sidebar_union(cx, it, u),
        clean::EnumItem(ref e) => sidebar_enum(cx, it, e),
        clean::TypedefItem(_) => sidebar_typedef(cx, it),
        clean::ModuleItem(ref m) => vec![sidebar_module(&m.items)],
        clean::ForeignTypeItem => sidebar_foreign_type(cx, it),
        _ => vec![],
    };
    // The sidebar is designed to display sibling functions, modules and
    // other miscellaneous information. since there are lots of sibling
    // items (and that causes quadratic growth in large modules),
    // we refactor common parts into a shared JavaScript file per module.
    // still, we don't move everything into JS because we want to preserve
    // as much HTML as possible in order to allow non-JS-enabled browsers
    // to navigate the documentation (though slightly inefficiently).
    let (title_prefix, title) = if it.is_struct()
        || it.is_trait()
        || it.is_primitive()
        || it.is_union()
        || it.is_enum()
        || it.is_mod()
        || it.is_typedef()
    {
        (
            match *it.kind {
                clean::ModuleItem(..) if it.is_crate() => "Crate ",
                clean::ModuleItem(..) => "Module ",
                _ => "",
            },
            it.name.as_ref().unwrap().as_str(),
        )
    } else {
        ("", "")
    };
    let version = if it.is_crate() {
        cx.cache().crate_version.as_ref().map(String::as_str).unwrap_or_default()
    } else {
        ""
    };
    let path: String = if !it.is_mod() {
        cx.current.iter().map(|s| s.as_str()).intersperse("::").collect()
    } else {
        "".into()
    };
    let sidebar = Sidebar { title_prefix, title, is_crate: it.is_crate(), version, blocks, path };
    sidebar.render_into(buffer).unwrap();
}

fn get_struct_fields_name<'a>(fields: &'a [clean::Item]) -> Vec<Link<'a>> {
    let mut fields = fields
        .iter()
        .filter(|f| matches!(*f.kind, clean::StructFieldItem(..)))
        .filter_map(|f| {
            f.name.as_ref().map(|name| Link::new(format!("structfield.{name}"), name.as_str()))
        })
        .collect::<Vec<Link<'a>>>();
    fields.sort();
    fields
}

fn sidebar_struct<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    s: &'a clean::Struct,
) -> Vec<LinkBlock<'a>> {
    let fields = get_struct_fields_name(&s.fields);
    let field_name = match s.ctor_kind {
        Some(CtorKind::Fn) => Some("Tuple Fields"),
        None => Some("Fields"),
        _ => None,
    };
    let mut items = vec![];
    if let Some(name) = field_name {
        items.push(LinkBlock::new(Link::new("fields", name), fields));
    }
    sidebar_assoc_items(cx, it, &mut items);
    items
}

fn sidebar_trait<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    t: &'a clean::Trait,
) -> Vec<LinkBlock<'a>> {
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

    let req_assoc = filter_items(&t.items, |m| m.is_ty_associated_type(), "associatedtype");
    let prov_assoc = filter_items(&t.items, |m| m.is_associated_type(), "associatedtype");
    let req_assoc_const =
        filter_items(&t.items, |m| m.is_ty_associated_const(), "associatedconstant");
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

    let mut blocks: Vec<LinkBlock<'_>> = [
        ("required-associated-types", "Required Associated Types", req_assoc),
        ("provided-associated-types", "Provided Associated Types", prov_assoc),
        ("required-associated-consts", "Required Associated Constants", req_assoc_const),
        ("provided-associated-consts", "Provided Associated Constants", prov_assoc_const),
        ("required-methods", "Required Methods", req_method),
        ("provided-methods", "Provided Methods", prov_method),
        ("foreign-impls", "Implementations on Foreign Types", foreign_impls),
    ]
    .into_iter()
    .map(|(id, title, items)| LinkBlock::new(Link::new(id, title), items))
    .collect();
    sidebar_assoc_items(cx, it, &mut blocks);
    blocks.push(LinkBlock::forced(Link::new("implementors", "Implementors")));
    if t.is_auto(cx.tcx()) {
        blocks.push(LinkBlock::forced(Link::new("synthetic-implementors", "Auto Implementors")));
    }
    blocks
}

fn sidebar_primitive<'a>(cx: &'a Context<'_>, it: &'a clean::Item) -> Vec<LinkBlock<'a>> {
    if it.name.map(|n| n.as_str() != "reference").unwrap_or(false) {
        let mut items = vec![];
        sidebar_assoc_items(cx, it, &mut items);
        items
    } else {
        let shared = Rc::clone(&cx.shared);
        let (concrete, synthetic, blanket_impl) =
            super::get_filtered_impls_for_reference(&shared, it);

        sidebar_render_assoc_items(cx, &mut IdMap::new(), concrete, synthetic, blanket_impl).into()
    }
}

fn sidebar_typedef<'a>(cx: &'a Context<'_>, it: &'a clean::Item) -> Vec<LinkBlock<'a>> {
    let mut items = vec![];
    sidebar_assoc_items(cx, it, &mut items);
    items
}

fn sidebar_union<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    u: &'a clean::Union,
) -> Vec<LinkBlock<'a>> {
    let fields = get_struct_fields_name(&u.fields);
    let mut items = vec![LinkBlock::new(Link::new("fields", "Fields"), fields)];
    sidebar_assoc_items(cx, it, &mut items);
    items
}

/// Adds trait implementations into the blocks of links
fn sidebar_assoc_items<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    links: &mut Vec<LinkBlock<'a>>,
) {
    let did = it.item_id.expect_def_id();
    let cache = cx.cache();

    let mut assoc_consts = Vec::new();
    let mut methods = Vec::new();
    if let Some(v) = cache.impls.get(&did) {
        let mut used_links = FxHashSet::default();
        let mut id_map = IdMap::new();

        {
            let used_links_bor = &mut used_links;
            assoc_consts.extend(
                v.iter()
                    .filter(|i| i.inner_impl().trait_.is_none())
                    .flat_map(|i| get_associated_constants(i.inner_impl(), used_links_bor)),
            );
            // We want links' order to be reproducible so we don't use unstable sort.
            assoc_consts.sort();

            #[rustfmt::skip] // rustfmt makes the pipeline less readable
            methods.extend(
                v.iter()
                    .filter(|i| i.inner_impl().trait_.is_none())
                    .flat_map(|i| get_methods(i.inner_impl(), false, used_links_bor, false, cx.tcx())),
            );

            // We want links' order to be reproducible so we don't use unstable sort.
            methods.sort();
        }

        let mut deref_methods = Vec::new();
        let [concrete, synthetic, blanket] = if v.iter().any(|i| i.inner_impl().trait_.is_some()) {
            if let Some(impl_) =
                v.iter().find(|i| i.trait_did() == cx.tcx().lang_items().deref_trait())
            {
                let mut derefs = DefIdSet::default();
                derefs.insert(did);
                sidebar_deref_methods(
                    cx,
                    &mut deref_methods,
                    impl_,
                    v,
                    &mut derefs,
                    &mut used_links,
                );
            }

            let (synthetic, concrete): (Vec<&Impl>, Vec<&Impl>) =
                v.iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_auto());
            let (blanket_impl, concrete): (Vec<&Impl>, Vec<&Impl>) =
                concrete.into_iter().partition::<Vec<_>, _>(|i| i.inner_impl().kind.is_blanket());

            sidebar_render_assoc_items(cx, &mut id_map, concrete, synthetic, blanket_impl)
        } else {
            std::array::from_fn(|_| LinkBlock::new(Link::empty(), vec![]))
        };

        let mut blocks = vec![
            LinkBlock::new(Link::new("implementations", "Associated Constants"), assoc_consts),
            LinkBlock::new(Link::new("implementations", "Methods"), methods),
        ];
        blocks.append(&mut deref_methods);
        blocks.extend([concrete, synthetic, blanket]);
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
            debug!("found inner_impl: {:?}", impls);
            let mut ret = impls
                .iter()
                .filter(|i| i.inner_impl().trait_.is_none())
                .flat_map(|i| get_methods(i.inner_impl(), true, used_links, deref_mut, cx.tcx()))
                .collect::<Vec<_>>();
            if !ret.is_empty() {
                let id = if let Some(target_def_id) = real_target.def_id(c) {
                    Cow::Borrowed(
                        cx.deref_id_map
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
                out.push(LinkBlock::new(Link::new(id, title), ret));
            }
        }

        // Recurse into any further impls that might exist for `target`
        if let Some(target_did) = target.def_id(c) &&
            let Some(target_impls) = c.impls.get(&target_did) &&
            let Some(target_deref_impl) = target_impls.iter().find(|i| {
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
            );
        }
    }
}

fn sidebar_enum<'a>(
    cx: &'a Context<'_>,
    it: &'a clean::Item,
    e: &'a clean::Enum,
) -> Vec<LinkBlock<'a>> {
    let mut variants = e
        .variants()
        .filter_map(|v| v.name)
        .map(|name| Link::new(format!("variant.{name}"), name.to_string()))
        .collect::<Vec<_>>();
    variants.sort_unstable();

    let mut items = vec![LinkBlock::new(Link::new("variants", "Variants"), variants)];
    sidebar_assoc_items(cx, it, &mut items);
    items
}

pub(crate) fn sidebar_module_like(
    item_sections_in_use: FxHashSet<ItemSection>,
) -> LinkBlock<'static> {
    let item_sections = ItemSection::ALL
        .iter()
        .copied()
        .filter(|sec| item_sections_in_use.contains(sec))
        .map(|sec| Link::new(sec.id(), sec.name()))
        .collect();
    LinkBlock::new(Link::empty(), item_sections)
}

fn sidebar_module(items: &[clean::Item]) -> LinkBlock<'static> {
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

    sidebar_module_like(item_sections_in_use)
}

fn sidebar_foreign_type<'a>(cx: &'a Context<'_>, it: &'a clean::Item) -> Vec<LinkBlock<'a>> {
    let mut items = vec![];
    sidebar_assoc_items(cx, it, &mut items);
    items
}

/// Renders the trait implementations for this type
fn sidebar_render_assoc_items(
    cx: &Context<'_>,
    id_map: &mut IdMap,
    concrete: Vec<&Impl>,
    synthetic: Vec<&Impl>,
    blanket_impl: Vec<&Impl>,
) -> [LinkBlock<'static>; 3] {
    let format_impls = |impls: Vec<&Impl>, id_map: &mut IdMap| {
        let mut links = FxHashSet::default();

        let mut ret = impls
            .iter()
            .filter_map(|it| {
                let trait_ = it.inner_impl().trait_.as_ref()?;
                let encoded =
                    id_map.derive(super::get_id_for_impl(&it.inner_impl().for_, Some(trait_), cx));

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
    [
        LinkBlock::new(Link::new("trait-implementations", "Trait Implementations"), concrete),
        LinkBlock::new(
            Link::new("synthetic-implementations", "Auto Trait Implementations"),
            synthetic,
        ),
        LinkBlock::new(Link::new("blanket-implementations", "Blanket Implementations"), blanket),
    ]
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

fn get_methods<'a>(
    i: &'a clean::Impl,
    for_deref: bool,
    used_links: &mut FxHashSet<String>,
    deref_mut: bool,
    tcx: TyCtxt<'_>,
) -> Vec<Link<'a>> {
    i.items
        .iter()
        .filter_map(|item| match item.name {
            Some(ref name) if !name.is_empty() && item.is_method() => {
                if !for_deref || super::should_render_item(item, deref_mut, tcx) {
                    Some(Link::new(
                        get_next_url(used_links, format!("{}.{}", ItemType::Method, name)),
                        name.as_str(),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

fn get_associated_constants<'a>(
    i: &'a clean::Impl,
    used_links: &mut FxHashSet<String>,
) -> Vec<Link<'a>> {
    i.items
        .iter()
        .filter_map(|item| match item.name {
            Some(ref name) if !name.is_empty() && item.is_associated_const() => Some(Link::new(
                get_next_url(used_links, format!("{}.{}", ItemType::AssocConst, name)),
                name.as_str(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>()
}
