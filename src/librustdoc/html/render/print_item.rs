use clean::AttributesExt;

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::stability;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{self, Adt, TyCtxt};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_target::abi::{LayoutS, Primitive, TagEncoding, Variants};
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

use super::{
    collect_paths_for_type, document, ensure_trailing_slash, get_filtered_impls_for_reference,
    item_ty_to_section, notable_traits_button, notable_traits_json, render_all_impls,
    render_assoc_item, render_assoc_items, render_attributes_in_code, render_attributes_in_pre,
    render_impl, render_rightside, render_stability_since_raw,
    render_stability_since_raw_with_extra, AssocItemLink, Context, ImplRenderingParameters,
};
use crate::clean;
use crate::config::ModuleSorting;
use crate::formats::item_type::ItemType;
use crate::formats::{AssocItemRender, Impl, RenderMode};
use crate::html::escape::Escape;
use crate::html::format::{
    display_fn, join_with_double_colon, print_abi_with_space, print_constness_with_space,
    print_where_clause, visibility_print_with_space, Buffer, Ending, PrintWithSpace,
};
use crate::html::layout::Page;
use crate::html::markdown::{HeadingOffset, MarkdownSummaryLine};
use crate::html::url_parts_builder::UrlPartsBuilder;
use crate::html::{highlight, static_files};

use askama::Template;
use itertools::Itertools;

const ITEM_TABLE_OPEN: &str = "<ul class=\"item-table\">";
const ITEM_TABLE_CLOSE: &str = "</ul>";
const ITEM_TABLE_ROW_OPEN: &str = "<li>";
const ITEM_TABLE_ROW_CLOSE: &str = "</li>";

// A component in a `use` path, like `string` in std::string::ToString
struct PathComponent {
    path: String,
    name: Symbol,
}

#[derive(Template)]
#[template(path = "print_item.html")]
struct ItemVars<'a> {
    static_root_path: &'a str,
    clipboard_svg: &'static static_files::StaticFile,
    typ: &'a str,
    name: &'a str,
    item_type: &'a str,
    path_components: Vec<PathComponent>,
    stability_since_raw: &'a str,
    src_href: Option<&'a str>,
}

/// Calls `print_where_clause` and returns `true` if a `where` clause was generated.
fn print_where_clause_and_check<'a, 'tcx: 'a>(
    buffer: &mut Buffer,
    gens: &'a clean::Generics,
    cx: &'a Context<'tcx>,
) -> bool {
    let len_before = buffer.len();
    write!(buffer, "{}", print_where_clause(gens, cx, 0, Ending::Newline));
    len_before != buffer.len()
}

pub(super) fn print_item(
    cx: &mut Context<'_>,
    item: &clean::Item,
    buf: &mut Buffer,
    page: &Page<'_>,
) {
    debug_assert!(!item.is_stripped());
    let typ = match *item.kind {
        clean::ModuleItem(_) => {
            if item.is_crate() {
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
        clean::KeywordItem => "Keyword ",
        clean::OpaqueTyItem(..) => "Opaque Type ",
        clean::TraitAliasItem(..) => "Trait Alias ",
        _ => {
            // We don't generate pages for any other type.
            unreachable!();
        }
    };
    let mut stability_since_raw = Buffer::new();
    render_stability_since_raw(
        &mut stability_since_raw,
        item.stable_since(cx.tcx()),
        item.const_stability(cx.tcx()),
        None,
        None,
    );
    let stability_since_raw: String = stability_since_raw.into_inner();

    // Write source tag
    //
    // When this item is part of a `crate use` in a downstream crate, the
    // source link in the downstream documentation will actually come back to
    // this page, and this link will be auto-clicked. The `id` attribute is
    // used to find the link to auto-click.
    let src_href =
        if cx.include_sources && !item.is_primitive() { cx.src_href(item) } else { None };

    let path_components = if item.is_primitive() || item.is_keyword() {
        vec![]
    } else {
        let cur = &cx.current;
        let amt = if item.is_mod() { cur.len() - 1 } else { cur.len() };
        cur.iter()
            .enumerate()
            .take(amt)
            .map(|(i, component)| PathComponent {
                path: "../".repeat(cur.len() - i - 1),
                name: *component,
            })
            .collect()
    };

    let item_vars = ItemVars {
        static_root_path: &page.get_static_root_path(),
        clipboard_svg: &static_files::STATIC_FILES.clipboard_svg,
        typ,
        name: item.name.as_ref().unwrap().as_str(),
        item_type: &item.type_().to_string(),
        path_components,
        stability_since_raw: &stability_since_raw,
        src_href: src_href.as_deref(),
    };

    item_vars.render_into(buf).unwrap();

    match &*item.kind {
        clean::ModuleItem(ref m) => item_module(buf, cx, item, &m.items),
        clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) => {
            item_function(buf, cx, item, f)
        }
        clean::TraitItem(ref t) => item_trait(buf, cx, item, t),
        clean::StructItem(ref s) => item_struct(buf, cx, item, s),
        clean::UnionItem(ref s) => item_union(buf, cx, item, s),
        clean::EnumItem(ref e) => item_enum(buf, cx, item, e),
        clean::TypedefItem(ref t) => item_typedef(buf, cx, item, t),
        clean::MacroItem(ref m) => item_macro(buf, cx, item, m),
        clean::ProcMacroItem(ref m) => item_proc_macro(buf, cx, item, m),
        clean::PrimitiveItem(_) => item_primitive(buf, cx, item),
        clean::StaticItem(ref i) | clean::ForeignStaticItem(ref i) => item_static(buf, cx, item, i),
        clean::ConstantItem(ref c) => item_constant(buf, cx, item, c),
        clean::ForeignTypeItem => item_foreign_type(buf, cx, item),
        clean::KeywordItem => item_keyword(buf, cx, item),
        clean::OpaqueTyItem(ref e) => item_opaque_ty(buf, cx, item, e),
        clean::TraitAliasItem(ref ta) => item_trait_alias(buf, cx, item, ta),
        _ => {
            // We don't generate pages for any other type.
            unreachable!();
        }
    }

    // Render notable-traits.js used for all methods in this module.
    if !cx.types_with_notable_traits.is_empty() {
        write!(
            buf,
            r#"<script type="text/json" id="notable-traits-data">{}</script>"#,
            notable_traits_json(cx.types_with_notable_traits.iter(), cx)
        );
        cx.types_with_notable_traits.clear();
    }
}

/// For large structs, enums, unions, etc, determine whether to hide their fields
fn should_hide_fields(n_fields: usize) -> bool {
    n_fields > 12
}

fn toggle_open(w: &mut Buffer, text: impl fmt::Display) {
    write!(
        w,
        "<details class=\"toggle type-contents-toggle\">\
            <summary class=\"hideme\">\
                <span>Show {}</span>\
            </summary>",
        text
    );
}

fn toggle_close(w: &mut Buffer) {
    w.write_str("</details>");
}

fn item_module(w: &mut Buffer, cx: &mut Context<'_>, item: &clean::Item, items: &[clean::Item]) {
    document(w, cx, item, None, HeadingOffset::H2);

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
        if item_ty_to_section(ty1) != item_ty_to_section(ty2)
            || (ty1 != ty2 && (ty1 == ItemType::ExternCrate || ty2 == ItemType::ExternCrate))
        {
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
        let lhs = i1.name.unwrap_or(kw::Empty);
        let rhs = i2.name.unwrap_or(kw::Empty);
        compare_names(lhs.as_str(), rhs.as_str())
    }

    match cx.shared.module_sorting {
        ModuleSorting::Alphabetical => {
            indices.sort_by(|&i1, &i2| cmp(&items[i1], &items[i2], i1, i2, cx.tcx()));
        }
        ModuleSorting::DeclarationOrder => {}
    }
    // This call is to remove re-export duplicates in cases such as:
    //
    // ```
    // pub(crate) mod foo {
    //     pub(crate) mod bar {
    //         pub(crate) trait Double { fn foo(); }
    //     }
    // }
    //
    // pub(crate) use foo::bar::*;
    // pub(crate) use foo::*;
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
            items[*i].item_id,
            if items[*i].name.is_some() { Some(full_path(cx, &items[*i])) } else { None },
            items[*i].type_(),
            if items[*i].is_import() { *i } else { 0 },
        )
    });

    debug!("{:?}", indices);
    let mut last_section = None;

    for &idx in &indices {
        let myitem = &items[idx];
        if myitem.is_stripped() {
            continue;
        }

        let my_section = item_ty_to_section(myitem.type_());
        if Some(my_section) != last_section {
            if last_section.is_some() {
                w.write_str(ITEM_TABLE_CLOSE);
            }
            last_section = Some(my_section);
            write!(
                w,
                "<h2 id=\"{id}\" class=\"small-section-header\">\
                    <a href=\"#{id}\">{name}</a>\
                 </h2>{}",
                ITEM_TABLE_OPEN,
                id = cx.derive_id(my_section.id().to_owned()),
                name = my_section.name(),
            );
        }

        let tcx = cx.tcx();
        match *myitem.kind {
            clean::ExternCrateItem { ref src } => {
                use crate::html::format::anchor;

                w.write_str(ITEM_TABLE_ROW_OPEN);
                match *src {
                    Some(src) => write!(
                        w,
                        "<div class=\"item-name\"><code>{}extern crate {} as {};",
                        visibility_print_with_space(myitem.visibility(tcx), myitem.item_id, cx),
                        anchor(myitem.item_id.expect_def_id(), src, cx),
                        myitem.name.unwrap(),
                    ),
                    None => write!(
                        w,
                        "<div class=\"item-name\"><code>{}extern crate {};",
                        visibility_print_with_space(myitem.visibility(tcx), myitem.item_id, cx),
                        anchor(myitem.item_id.expect_def_id(), myitem.name.unwrap(), cx),
                    ),
                }
                w.write_str("</code></div>");
                w.write_str(ITEM_TABLE_ROW_CLOSE);
            }

            clean::ImportItem(ref import) => {
                let stab_tags = if let Some(import_def_id) = import.source.did {
                    let ast_attrs = cx.tcx().get_attrs_unchecked(import_def_id);
                    let import_attrs = Box::new(clean::Attributes::from_ast(ast_attrs));

                    // Just need an item with the correct def_id and attrs
                    let import_item = clean::Item {
                        item_id: import_def_id.into(),
                        attrs: import_attrs,
                        cfg: ast_attrs.cfg(cx.tcx(), &cx.cache().hidden_cfg),
                        ..myitem.clone()
                    };

                    let stab_tags = Some(extra_info_tags(&import_item, item, cx.tcx()).to_string());
                    stab_tags
                } else {
                    None
                };

                w.write_str(ITEM_TABLE_ROW_OPEN);
                let id = match import.kind {
                    clean::ImportKind::Simple(s) => {
                        format!(" id=\"{}\"", cx.derive_id(format!("reexport.{}", s)))
                    }
                    clean::ImportKind::Glob => String::new(),
                };
                let stab_tags = stab_tags.unwrap_or_default();
                let (stab_tags_before, stab_tags_after) = if stab_tags.is_empty() {
                    ("", "")
                } else {
                    ("<div class=\"desc docblock-short\">", "</div>")
                };
                write!(
                    w,
                    "<div class=\"item-name\"{id}>\
                         <code>{vis}{imp}</code>\
                     </div>\
                     {stab_tags_before}{stab_tags}{stab_tags_after}",
                    vis = visibility_print_with_space(myitem.visibility(tcx), myitem.item_id, cx),
                    imp = import.print(cx),
                );
                w.write_str(ITEM_TABLE_ROW_CLOSE);
            }

            _ => {
                if myitem.name.is_none() {
                    continue;
                }

                let unsafety_flag = match *myitem.kind {
                    clean::FunctionItem(_) | clean::ForeignFunctionItem(_)
                        if myitem.fn_header(cx.tcx()).unwrap().unsafety
                            == hir::Unsafety::Unsafe =>
                    {
                        "<sup title=\"unsafe function\">⚠</sup>"
                    }
                    _ => "",
                };

                let visibility_emoji = match myitem.visibility(tcx) {
                    Some(ty::Visibility::Restricted(_)) => {
                        "<span title=\"Restricted Visibility\">&nbsp;🔒</span> "
                    }
                    _ => "",
                };

                let doc_value = myitem.doc_value().unwrap_or_default();
                w.write_str(ITEM_TABLE_ROW_OPEN);
                let docs = MarkdownSummaryLine(&doc_value, &myitem.links(cx)).into_string();
                let (docs_before, docs_after) = if docs.is_empty() {
                    ("", "")
                } else {
                    ("<div class=\"desc docblock-short\">", "</div>")
                };
                write!(
                    w,
                    "<div class=\"item-name\">\
                        <a class=\"{class}\" href=\"{href}\" title=\"{title}\">{name}</a>\
                        {visibility_emoji}\
                        {unsafety_flag}\
                        {stab_tags}\
                     </div>\
                     {docs_before}{docs}{docs_after}",
                    name = myitem.name.unwrap(),
                    visibility_emoji = visibility_emoji,
                    stab_tags = extra_info_tags(myitem, item, cx.tcx()),
                    class = myitem.type_(),
                    unsafety_flag = unsafety_flag,
                    href = item_path(myitem.type_(), myitem.name.unwrap().as_str()),
                    title = [myitem.type_().to_string(), full_path(cx, myitem)]
                        .iter()
                        .filter_map(|s| if !s.is_empty() { Some(s.as_str()) } else { None })
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                w.write_str(ITEM_TABLE_ROW_CLOSE);
            }
        }
    }

    if last_section.is_some() {
        w.write_str(ITEM_TABLE_CLOSE);
    }
}

/// Render the stability, deprecation and portability tags that are displayed in the item's summary
/// at the module level.
fn extra_info_tags<'a, 'tcx: 'a>(
    item: &'a clean::Item,
    parent: &'a clean::Item,
    tcx: TyCtxt<'tcx>,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    display_fn(move |f| {
        fn tag_html<'a>(
            class: &'a str,
            title: &'a str,
            contents: &'a str,
        ) -> impl fmt::Display + 'a {
            display_fn(move |f| {
                write!(
                    f,
                    r#"<span class="stab {}" title="{}">{}</span>"#,
                    class,
                    Escape(title),
                    contents
                )
            })
        }

        // The trailing space after each tag is to space it properly against the rest of the docs.
        if let Some(depr) = &item.deprecation(tcx) {
            let message = if stability::deprecation_in_effect(depr) {
                "Deprecated"
            } else {
                "Deprecation planned"
            };
            write!(f, "{}", tag_html("deprecated", "", message))?;
        }

        // The "rustc_private" crates are permanently unstable so it makes no sense
        // to render "unstable" everywhere.
        if item.stability(tcx).as_ref().map(|s| s.is_unstable() && s.feature != sym::rustc_private)
            == Some(true)
        {
            write!(f, "{}", tag_html("unstable", "", "Experimental"))?;
        }

        let cfg = match (&item.cfg, parent.cfg.as_ref()) {
            (Some(cfg), Some(parent_cfg)) => cfg.simplify_with(parent_cfg),
            (cfg, _) => cfg.as_deref().cloned(),
        };

        debug!("Portability name={:?} {:?} - {:?} = {:?}", item.name, item.cfg, parent.cfg, cfg);
        if let Some(ref cfg) = cfg {
            write!(
                f,
                "{}",
                tag_html("portability", &cfg.render_long_plain(), &cfg.render_short_html())
            )
        } else {
            Ok(())
        }
    })
}

fn item_function(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, f: &clean::Function) {
    let tcx = cx.tcx();
    let header = it.fn_header(tcx).expect("printing a function which isn't a function");
    let constness = print_constness_with_space(&header.constness, it.const_stability(tcx));
    let unsafety = header.unsafety.print_with_space();
    let abi = print_abi_with_space(header.abi).to_string();
    let asyncness = header.asyncness.print_with_space();
    let visibility = visibility_print_with_space(it.visibility(tcx), it.item_id, cx).to_string();
    let name = it.name.unwrap();

    let generics_len = format!("{:#}", f.generics.print(cx)).len();
    let header_len = "fn ".len()
        + visibility.len()
        + constness.len()
        + asyncness.len()
        + unsafety.len()
        + abi.len()
        + name.as_str().len()
        + generics_len;

    let notable_traits =
        f.decl.output.as_return().and_then(|output| notable_traits_button(output, cx));

    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        w.reserve(header_len);
        write!(
            w,
            "{vis}{constness}{asyncness}{unsafety}{abi}fn \
                {name}{generics}{decl}{notable_traits}{where_clause}",
            vis = visibility,
            constness = constness,
            asyncness = asyncness,
            unsafety = unsafety,
            abi = abi,
            name = name,
            generics = f.generics.print(cx),
            where_clause = print_where_clause(&f.generics, cx, 0, Ending::Newline),
            decl = f.decl.full_print(header_len, 0, cx),
            notable_traits = notable_traits.unwrap_or_default(),
        );
    });
    document(w, cx, it, None, HeadingOffset::H2);
}

fn item_trait(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, t: &clean::Trait) {
    let tcx = cx.tcx();
    let bounds = bounds(&t.bounds, false, cx);
    let required_types = t.items.iter().filter(|m| m.is_ty_associated_type()).collect::<Vec<_>>();
    let provided_types = t.items.iter().filter(|m| m.is_associated_type()).collect::<Vec<_>>();
    let required_consts = t.items.iter().filter(|m| m.is_ty_associated_const()).collect::<Vec<_>>();
    let provided_consts = t.items.iter().filter(|m| m.is_associated_const()).collect::<Vec<_>>();
    let required_methods = t.items.iter().filter(|m| m.is_ty_method()).collect::<Vec<_>>();
    let provided_methods = t.items.iter().filter(|m| m.is_method()).collect::<Vec<_>>();
    let count_types = required_types.len() + provided_types.len();
    let count_consts = required_consts.len() + provided_consts.len();
    let count_methods = required_methods.len() + provided_methods.len();
    let must_implement_one_of_functions = tcx.trait_def(t.def_id).must_implement_one_of.clone();

    // Output the trait definition
    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        write!(
            w,
            "{}{}{}trait {}{}{}",
            visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
            t.unsafety(tcx).print_with_space(),
            if t.is_auto(tcx) { "auto " } else { "" },
            it.name.unwrap(),
            t.generics.print(cx),
            bounds
        );

        if !t.generics.where_predicates.is_empty() {
            write!(w, "{}", print_where_clause(&t.generics, cx, 0, Ending::Newline));
        } else {
            w.write_str(" ");
        }

        if t.items.is_empty() {
            w.write_str("{ }");
        } else {
            // FIXME: we should be using a derived_id for the Anchors here
            w.write_str("{\n");
            let mut toggle = false;

            // If there are too many associated types, hide _everything_
            if should_hide_fields(count_types) {
                toggle = true;
                toggle_open(
                    w,
                    format_args!("{} associated items", count_types + count_consts + count_methods),
                );
            }
            for types in [&required_types, &provided_types] {
                for t in types {
                    render_assoc_item(
                        w,
                        t,
                        AssocItemLink::Anchor(None),
                        ItemType::Trait,
                        cx,
                        RenderMode::Normal,
                    );
                    w.write_str(";\n");
                }
            }
            // If there are too many associated constants, hide everything after them
            // We also do this if the types + consts is large because otherwise we could
            // render a bunch of types and _then_ a bunch of consts just because both were
            // _just_ under the limit
            if !toggle && should_hide_fields(count_types + count_consts) {
                toggle = true;
                toggle_open(
                    w,
                    format_args!(
                        "{} associated constant{} and {} method{}",
                        count_consts,
                        pluralize(count_consts),
                        count_methods,
                        pluralize(count_methods),
                    ),
                );
            }
            if count_types != 0 && (count_consts != 0 || count_methods != 0) {
                w.write_str("\n");
            }
            for consts in [&required_consts, &provided_consts] {
                for c in consts {
                    render_assoc_item(
                        w,
                        c,
                        AssocItemLink::Anchor(None),
                        ItemType::Trait,
                        cx,
                        RenderMode::Normal,
                    );
                    w.write_str(";\n");
                }
            }
            if !toggle && should_hide_fields(count_methods) {
                toggle = true;
                toggle_open(w, format_args!("{} methods", count_methods));
            }
            if count_consts != 0 && count_methods != 0 {
                w.write_str("\n");
            }

            if !required_methods.is_empty() {
                write!(w, "    // Required method{}\n", pluralize(required_methods.len()));
            }
            for (pos, m) in required_methods.iter().enumerate() {
                render_assoc_item(
                    w,
                    m,
                    AssocItemLink::Anchor(None),
                    ItemType::Trait,
                    cx,
                    RenderMode::Normal,
                );
                w.write_str(";\n");

                if pos < required_methods.len() - 1 {
                    w.write_str("<span class=\"item-spacer\"></span>");
                }
            }
            if !required_methods.is_empty() && !provided_methods.is_empty() {
                w.write_str("\n");
            }

            if !provided_methods.is_empty() {
                write!(w, "    // Provided method{}\n", pluralize(provided_methods.len()));
            }
            for (pos, m) in provided_methods.iter().enumerate() {
                render_assoc_item(
                    w,
                    m,
                    AssocItemLink::Anchor(None),
                    ItemType::Trait,
                    cx,
                    RenderMode::Normal,
                );

                w.write_str(" { ... }\n");

                if pos < provided_methods.len() - 1 {
                    w.write_str("<span class=\"item-spacer\"></span>");
                }
            }
            if toggle {
                toggle_close(w);
            }
            w.write_str("}");
        }
    });

    // Trait documentation
    document(w, cx, it, None, HeadingOffset::H2);

    fn write_small_section_header(w: &mut Buffer, id: &str, title: &str, extra_content: &str) {
        write!(
            w,
            "<h2 id=\"{0}\" class=\"small-section-header\">\
                {1}<a href=\"#{0}\" class=\"anchor\">§</a>\
             </h2>{2}",
            id, title, extra_content
        )
    }

    fn trait_item(w: &mut Buffer, cx: &mut Context<'_>, m: &clean::Item, t: &clean::Item) {
        let name = m.name.unwrap();
        info!("Documenting {} on {:?}", name, t.name);
        let item_type = m.type_();
        let id = cx.derive_id(format!("{}.{}", item_type, name));
        let mut content = Buffer::empty_from(w);
        document(&mut content, cx, m, Some(t), HeadingOffset::H5);
        let toggled = !content.is_empty();
        if toggled {
            let method_toggle_class = if item_type.is_method() { " method-toggle" } else { "" };
            write!(w, "<details class=\"toggle{method_toggle_class}\" open><summary>");
        }
        write!(w, "<section id=\"{}\" class=\"method\">", id);
        render_rightside(w, cx, m, t, RenderMode::Normal);
        write!(w, "<h4 class=\"code-header\">");
        render_assoc_item(
            w,
            m,
            AssocItemLink::Anchor(Some(&id)),
            ItemType::Impl,
            cx,
            RenderMode::Normal,
        );
        w.write_str("</h4>");
        w.write_str("</section>");
        if toggled {
            write!(w, "</summary>");
            w.push_buffer(content);
            write!(w, "</details>");
        }
    }

    if !required_types.is_empty() {
        write_small_section_header(
            w,
            "required-associated-types",
            "Required Associated Types",
            "<div class=\"methods\">",
        );
        for t in required_types {
            trait_item(w, cx, t, it);
        }
        w.write_str("</div>");
    }
    if !provided_types.is_empty() {
        write_small_section_header(
            w,
            "provided-associated-types",
            "Provided Associated Types",
            "<div class=\"methods\">",
        );
        for t in provided_types {
            trait_item(w, cx, t, it);
        }
        w.write_str("</div>");
    }

    if !required_consts.is_empty() {
        write_small_section_header(
            w,
            "required-associated-consts",
            "Required Associated Constants",
            "<div class=\"methods\">",
        );
        for t in required_consts {
            trait_item(w, cx, t, it);
        }
        w.write_str("</div>");
    }
    if !provided_consts.is_empty() {
        write_small_section_header(
            w,
            "provided-associated-consts",
            "Provided Associated Constants",
            "<div class=\"methods\">",
        );
        for t in provided_consts {
            trait_item(w, cx, t, it);
        }
        w.write_str("</div>");
    }

    // Output the documentation for each function individually
    if !required_methods.is_empty() || must_implement_one_of_functions.is_some() {
        write_small_section_header(
            w,
            "required-methods",
            "Required Methods",
            "<div class=\"methods\">",
        );

        if let Some(list) = must_implement_one_of_functions.as_deref() {
            write!(
                w,
                "<div class=\"stab must_implement\">At least one of the `{}` methods is required.</div>",
                list.iter().join("`, `")
            );
        }

        for m in required_methods {
            trait_item(w, cx, m, it);
        }
        w.write_str("</div>");
    }
    if !provided_methods.is_empty() {
        write_small_section_header(
            w,
            "provided-methods",
            "Provided Methods",
            "<div class=\"methods\">",
        );
        for m in provided_methods {
            trait_item(w, cx, m, it);
        }
        w.write_str("</div>");
    }

    // If there are methods directly on this trait object, render them here.
    render_assoc_items(w, cx, it, it.item_id.expect_def_id(), AssocItemRender::All);

    let cloned_shared = Rc::clone(&cx.shared);
    let cache = &cloned_shared.cache;
    let mut extern_crates = FxHashSet::default();
    if let Some(implementors) = cache.implementors.get(&it.item_id.expect_def_id()) {
        // The DefId is for the first Type found with that name. The bool is
        // if any Types with the same name but different DefId have been found.
        let mut implementor_dups: FxHashMap<Symbol, (DefId, bool)> = FxHashMap::default();
        for implementor in implementors {
            if let Some(did) = implementor.inner_impl().for_.without_borrowed_ref().def_id(cache) &&
                !did.is_local() {
                extern_crates.insert(did.krate);
            }
            match implementor.inner_impl().for_.without_borrowed_ref() {
                clean::Type::Path { ref path } if !path.is_assoc_ty() => {
                    let did = path.def_id();
                    let &mut (prev_did, ref mut has_duplicates) =
                        implementor_dups.entry(path.last()).or_insert((did, false));
                    if prev_did != did {
                        *has_duplicates = true;
                    }
                }
                _ => {}
            }
        }

        let (local, foreign) =
            implementors.iter().partition::<Vec<_>, _>(|i| i.is_on_local_type(cx));

        let (mut synthetic, mut concrete): (Vec<&&Impl>, Vec<&&Impl>) =
            local.iter().partition(|i| i.inner_impl().kind.is_auto());

        synthetic.sort_by(|a, b| compare_impl(a, b, cx));
        concrete.sort_by(|a, b| compare_impl(a, b, cx));

        if !foreign.is_empty() {
            write_small_section_header(w, "foreign-impls", "Implementations on Foreign Types", "");

            for implementor in foreign {
                let provided_methods = implementor.inner_impl().provided_trait_methods(cx.tcx());
                let assoc_link =
                    AssocItemLink::GotoSource(implementor.impl_item.item_id, &provided_methods);
                render_impl(
                    w,
                    cx,
                    implementor,
                    it,
                    assoc_link,
                    RenderMode::Normal,
                    None,
                    &[],
                    ImplRenderingParameters {
                        show_def_docs: false,
                        show_default_items: false,
                        show_non_assoc_items: true,
                        toggle_open_by_default: false,
                    },
                );
            }
        }

        write_small_section_header(
            w,
            "implementors",
            "Implementors",
            "<div id=\"implementors-list\">",
        );
        for implementor in concrete {
            render_implementor(cx, implementor, it, w, &implementor_dups, &[]);
        }
        w.write_str("</div>");

        if t.is_auto(cx.tcx()) {
            write_small_section_header(
                w,
                "synthetic-implementors",
                "Auto implementors",
                "<div id=\"synthetic-implementors-list\">",
            );
            for implementor in synthetic {
                render_implementor(
                    cx,
                    implementor,
                    it,
                    w,
                    &implementor_dups,
                    &collect_paths_for_type(implementor.inner_impl().for_.clone(), cache),
                );
            }
            w.write_str("</div>");
        }
    } else {
        // even without any implementations to write in, we still want the heading and list, so the
        // implementors javascript file pulled in below has somewhere to write the impls into
        write_small_section_header(
            w,
            "implementors",
            "Implementors",
            "<div id=\"implementors-list\"></div>",
        );

        if t.is_auto(cx.tcx()) {
            write_small_section_header(
                w,
                "synthetic-implementors",
                "Auto implementors",
                "<div id=\"synthetic-implementors-list\"></div>",
            );
        }
    }

    // Include implementors in crates that depend on the current crate.
    //
    // This is complicated by the way rustdoc is invoked, which is basically
    // the same way rustc is invoked: it gets called, one at a time, for each
    // crate. When building the rustdocs for the current crate, rustdoc can
    // see crate metadata for its dependencies, but cannot see metadata for its
    // dependents.
    //
    // To make this work, we generate a "hook" at this stage, and our
    // dependents can "plug in" to it when they build. For simplicity's sake,
    // it's [JSONP]: a JavaScript file with the data we need (and can parse),
    // surrounded by a tiny wrapper that the Rust side ignores, but allows the
    // JavaScript side to include without having to worry about Same Origin
    // Policy. The code for *that* is in `write_shared.rs`.
    //
    // This is further complicated by `#[doc(inline)]`. We want all copies
    // of an inlined trait to reference the same JS file, to address complex
    // dependency graphs like this one (lower crates depend on higher crates):
    //
    // ```text
    //  --------------------------------------------
    //  |            crate A: trait Foo            |
    //  --------------------------------------------
    //      |                               |
    //  --------------------------------    |
    //  | crate B: impl A::Foo for Bar |    |
    //  --------------------------------    |
    //      |                               |
    //  ---------------------------------------------
    //  | crate C: #[doc(inline)] use A::Foo as Baz |
    //  |          impl Baz for Quux                |
    //  ---------------------------------------------
    // ```
    //
    // Basically, we want `C::Baz` and `A::Foo` to show the same set of
    // impls, which is easier if they both treat `/implementors/A/trait.Foo.js`
    // as the Single Source of Truth.
    //
    // We also want the `impl Baz for Quux` to be written to
    // `trait.Foo.js`. However, when we generate plain HTML for `C::Baz`,
    // we're going to want to generate plain HTML for `impl Baz for Quux` too,
    // because that'll load faster, and it's better for SEO. And we don't want
    // the same impl to show up twice on the same page.
    //
    // To make this work, the implementors JS file has a structure kinda
    // like this:
    //
    // ```js
    // JSONP({
    // "B": {"impl A::Foo for Bar"},
    // "C": {"impl Baz for Quux"},
    // });
    // ```
    //
    // First of all, this means we can rebuild a crate, and it'll replace its own
    // data if something changes. That is, `rustdoc` is idempotent. The other
    // advantage is that we can list the crates that get included in the HTML,
    // and ignore them when doing the JavaScript-based part of rendering.
    // So C's HTML will have something like this:
    //
    // ```html
    // <script src="/implementors/A/trait.Foo.js"
    //     data-ignore-extern-crates="A,B" async></script>
    // ```
    //
    // And, when the JS runs, anything in data-ignore-extern-crates is known
    // to already be in the HTML, and will be ignored.
    //
    // [JSONP]: https://en.wikipedia.org/wiki/JSONP
    let mut js_src_path: UrlPartsBuilder = std::iter::repeat("..")
        .take(cx.current.len())
        .chain(std::iter::once("implementors"))
        .collect();
    if let Some(did) = it.item_id.as_def_id() &&
        let get_extern = { || cache.external_paths.get(&did).map(|s| &s.0) } &&
        let Some(fqp) = cache.exact_paths.get(&did).or_else(get_extern) {
        js_src_path.extend(fqp[..fqp.len() - 1].iter().copied());
        js_src_path.push_fmt(format_args!("{}.{}.js", it.type_(), fqp.last().unwrap()));
    } else {
        js_src_path.extend(cx.current.iter().copied());
        js_src_path.push_fmt(format_args!("{}.{}.js", it.type_(), it.name.unwrap()));
    }
    let extern_crates = extern_crates
        .into_iter()
        .map(|cnum| tcx.crate_name(cnum).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let (extern_before, extern_after) =
        if extern_crates.is_empty() { ("", "") } else { (" data-ignore-extern-crates=\"", "\"") };
    write!(
        w,
        "<script src=\"{src}\"{extern_before}{extern_crates}{extern_after} async></script>",
        src = js_src_path.finish(),
    );
}

fn item_trait_alias(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, t: &clean::TraitAlias) {
    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        write!(
            w,
            "trait {}{}{} = {};",
            it.name.unwrap(),
            t.generics.print(cx),
            print_where_clause(&t.generics, cx, 0, Ending::Newline),
            bounds(&t.bounds, true, cx)
        );
    });

    document(w, cx, it, None, HeadingOffset::H2);

    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, it.item_id.expect_def_id(), AssocItemRender::All)
}

fn item_opaque_ty(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, t: &clean::OpaqueTy) {
    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        write!(
            w,
            "type {}{}{where_clause} = impl {bounds};",
            it.name.unwrap(),
            t.generics.print(cx),
            where_clause = print_where_clause(&t.generics, cx, 0, Ending::Newline),
            bounds = bounds(&t.bounds, false, cx),
        );
    });

    document(w, cx, it, None, HeadingOffset::H2);

    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, it.item_id.expect_def_id(), AssocItemRender::All)
}

fn item_typedef(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, t: &clean::Typedef) {
    fn write_content(w: &mut Buffer, cx: &Context<'_>, it: &clean::Item, t: &clean::Typedef) {
        wrap_item(w, |w| {
            render_attributes_in_pre(w, it, "");
            write!(
                w,
                "{}type {}{}{where_clause} = {type_};",
                visibility_print_with_space(it.visibility(cx.tcx()), it.item_id, cx),
                it.name.unwrap(),
                t.generics.print(cx),
                where_clause = print_where_clause(&t.generics, cx, 0, Ending::Newline),
                type_ = t.type_.print(cx),
            );
        });
    }

    write_content(w, cx, it, t);

    document(w, cx, it, None, HeadingOffset::H2);

    let def_id = it.item_id.expect_def_id();
    // Render any items associated directly to this alias, as otherwise they
    // won't be visible anywhere in the docs. It would be nice to also show
    // associated items from the aliased type (see discussion in #32077), but
    // we need #14072 to make sense of the generics.
    render_assoc_items(w, cx, it, def_id, AssocItemRender::All);
    document_type_layout(w, cx, def_id);
}

fn item_union(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, s: &clean::Union) {
    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        render_union(w, it, Some(&s.generics), &s.fields, cx);
    });

    document(w, cx, it, None, HeadingOffset::H2);

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
            "<h2 id=\"fields\" class=\"fields small-section-header\">\
                Fields<a href=\"#fields\" class=\"anchor\">§</a>\
            </h2>"
        );
        for (field, ty) in fields {
            let name = field.name.expect("union field name");
            let id = format!("{}.{}", ItemType::StructField, name);
            write!(
                w,
                "<span id=\"{id}\" class=\"{shortty} small-section-header\">\
                     <a href=\"#{id}\" class=\"anchor field\">§</a>\
                     <code>{name}: {ty}</code>\
                 </span>",
                shortty = ItemType::StructField,
                ty = ty.print(cx),
            );
            if let Some(stability_class) = field.stability_class(cx.tcx()) {
                write!(w, "<span class=\"stab {stability_class}\"></span>");
            }
            document(w, cx, field, Some(it), HeadingOffset::H3);
        }
    }
    let def_id = it.item_id.expect_def_id();
    render_assoc_items(w, cx, it, def_id, AssocItemRender::All);
    document_type_layout(w, cx, def_id);
}

fn print_tuple_struct_fields(w: &mut Buffer, cx: &Context<'_>, s: &[clean::Item]) {
    for (i, ty) in s.iter().enumerate() {
        if i > 0 {
            w.write_str(", ");
        }
        match *ty.kind {
            clean::StrippedItem(box clean::StructFieldItem(_)) => w.write_str("_"),
            clean::StructFieldItem(ref ty) => write!(w, "{}", ty.print(cx)),
            _ => unreachable!(),
        }
    }
}

fn item_enum(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, e: &clean::Enum) {
    let tcx = cx.tcx();
    let count_variants = e.variants().count();
    wrap_item(w, |w| {
        render_attributes_in_pre(w, it, "");
        write!(
            w,
            "{}enum {}{}",
            visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
            it.name.unwrap(),
            e.generics.print(cx),
        );
        if !print_where_clause_and_check(w, &e.generics, cx) {
            // If there wasn't a `where` clause, we add a whitespace.
            w.write_str(" ");
        }

        let variants_stripped = e.has_stripped_entries();
        if count_variants == 0 && !variants_stripped {
            w.write_str("{}");
        } else {
            w.write_str("{\n");
            let toggle = should_hide_fields(count_variants);
            if toggle {
                toggle_open(w, format_args!("{} variants", count_variants));
            }
            for v in e.variants() {
                w.write_str("    ");
                let name = v.name.unwrap();
                match *v.kind {
                    // FIXME(#101337): Show discriminant
                    clean::VariantItem(ref var) => match var.kind {
                        clean::VariantKind::CLike => write!(w, "{}", name),
                        clean::VariantKind::Tuple(ref s) => {
                            write!(w, "{}(", name);
                            print_tuple_struct_fields(w, cx, s);
                            w.write_str(")");
                        }
                        clean::VariantKind::Struct(ref s) => {
                            render_struct(w, v, None, None, &s.fields, "    ", false, cx);
                        }
                    },
                    _ => unreachable!(),
                }
                w.write_str(",\n");
            }

            if variants_stripped {
                w.write_str("    // some variants omitted\n");
            }
            if toggle {
                toggle_close(w);
            }
            w.write_str("}");
        }
    });

    document(w, cx, it, None, HeadingOffset::H2);

    if count_variants != 0 {
        write!(
            w,
            "<h2 id=\"variants\" class=\"variants small-section-header\">\
                Variants{}<a href=\"#variants\" class=\"anchor\">§</a>\
            </h2>",
            document_non_exhaustive_header(it)
        );
        document_non_exhaustive(w, it);
        write!(w, "<div class=\"variants\">");
        for variant in e.variants() {
            let id = cx.derive_id(format!("{}.{}", ItemType::Variant, variant.name.unwrap()));
            write!(
                w,
                "<section id=\"{id}\" class=\"variant\">\
                    <a href=\"#{id}\" class=\"anchor\">§</a>",
            );
            render_stability_since_raw_with_extra(
                w,
                variant.stable_since(tcx),
                variant.const_stability(tcx),
                it.stable_since(tcx),
                it.const_stable_since(tcx),
                " rightside",
            );
            write!(w, "<h3 class=\"code-header\">{name}", name = variant.name.unwrap());

            let clean::VariantItem(variant_data) = &*variant.kind else { unreachable!() };

            if let clean::VariantKind::Tuple(ref s) = variant_data.kind {
                w.write_str("(");
                print_tuple_struct_fields(w, cx, s);
                w.write_str(")");
            }
            w.write_str("</h3></section>");

            let heading_and_fields = match &variant_data.kind {
                clean::VariantKind::Struct(s) => Some(("Fields", &s.fields)),
                clean::VariantKind::Tuple(fields) => {
                    // Documentation on tuple variant fields is rare, so to reduce noise we only emit
                    // the section if at least one field is documented.
                    if fields.iter().any(|f| f.doc_value().is_some()) {
                        Some(("Tuple Fields", fields))
                    } else {
                        None
                    }
                }
                clean::VariantKind::CLike => None,
            };

            if let Some((heading, fields)) = heading_and_fields {
                let variant_id =
                    cx.derive_id(format!("{}.{}.fields", ItemType::Variant, variant.name.unwrap()));
                write!(
                    w,
                    "<div class=\"sub-variant\" id=\"{variant_id}\">\
                        <h4>{heading}</h4>",
                );
                document_non_exhaustive(w, variant);
                for field in fields {
                    match *field.kind {
                        clean::StrippedItem(box clean::StructFieldItem(_)) => {}
                        clean::StructFieldItem(ref ty) => {
                            let id = cx.derive_id(format!(
                                "variant.{}.field.{}",
                                variant.name.unwrap(),
                                field.name.unwrap()
                            ));
                            write!(
                                w,
                                "<div class=\"sub-variant-field\">\
                                 <span id=\"{id}\" class=\"small-section-header\">\
                                     <a href=\"#{id}\" class=\"anchor field\">§</a>\
                                     <code>{f}: {t}</code>\
                                 </span>",
                                f = field.name.unwrap(),
                                t = ty.print(cx)
                            );
                            document(w, cx, field, Some(variant), HeadingOffset::H5);
                            write!(w, "</div>");
                        }
                        _ => unreachable!(),
                    }
                }
                w.write_str("</div>");
            }

            document(w, cx, variant, Some(it), HeadingOffset::H4);
        }
        write!(w, "</div>");
    }
    let def_id = it.item_id.expect_def_id();
    render_assoc_items(w, cx, it, def_id, AssocItemRender::All);
    document_type_layout(w, cx, def_id);
}

fn item_macro(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, t: &clean::Macro) {
    highlight::render_item_decl_with_highlighting(&t.source, w);
    document(w, cx, it, None, HeadingOffset::H2)
}

fn item_proc_macro(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, m: &clean::ProcMacro) {
    wrap_item(w, |w| {
        let name = it.name.expect("proc-macros always have names");
        match m.kind {
            MacroKind::Bang => {
                write!(w, "{}!() {{ /* proc-macro */ }}", name);
            }
            MacroKind::Attr => {
                write!(w, "#[{}]", name);
            }
            MacroKind::Derive => {
                write!(w, "#[derive({})]", name);
                if !m.helpers.is_empty() {
                    w.push_str("\n{\n");
                    w.push_str("    // Attributes available to this derive:\n");
                    for attr in &m.helpers {
                        writeln!(w, "    #[{}]", attr);
                    }
                    w.push_str("}\n");
                }
            }
        }
    });
    document(w, cx, it, None, HeadingOffset::H2)
}

fn item_primitive(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item) {
    let def_id = it.item_id.expect_def_id();
    document(w, cx, it, None, HeadingOffset::H2);
    if it.name.map(|n| n.as_str() != "reference").unwrap_or(false) {
        render_assoc_items(w, cx, it, def_id, AssocItemRender::All);
    } else {
        // We handle the "reference" primitive type on its own because we only want to list
        // implementations on generic types.
        let shared = Rc::clone(&cx.shared);
        let (concrete, synthetic, blanket_impl) = get_filtered_impls_for_reference(&shared, it);

        render_all_impls(w, cx, it, &concrete, &synthetic, &blanket_impl);
    }
}

fn item_constant(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, c: &clean::Constant) {
    wrap_item(w, |w| {
        let tcx = cx.tcx();
        render_attributes_in_code(w, it);

        write!(
            w,
            "{vis}const {name}: {typ}",
            vis = visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
            name = it.name.unwrap(),
            typ = c.type_.print(cx),
        );

        // FIXME: The code below now prints
        //            ` = _; // 100i32`
        //        if the expression is
        //            `50 + 50`
        //        which looks just wrong.
        //        Should we print
        //            ` = 100i32;`
        //        instead?

        let value = c.value(tcx);
        let is_literal = c.is_literal(tcx);
        let expr = c.expr(tcx);
        if value.is_some() || is_literal {
            write!(w, " = {expr};", expr = Escape(&expr));
        } else {
            w.write_str(";");
        }

        if !is_literal {
            if let Some(value) = &value {
                let value_lowercase = value.to_lowercase();
                let expr_lowercase = expr.to_lowercase();

                if value_lowercase != expr_lowercase
                    && value_lowercase.trim_end_matches("i32") != expr_lowercase
                {
                    write!(w, " // {value}", value = Escape(value));
                }
            }
        }
    });

    document(w, cx, it, None, HeadingOffset::H2)
}

fn item_struct(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, s: &clean::Struct) {
    wrap_item(w, |w| {
        render_attributes_in_code(w, it);
        render_struct(w, it, Some(&s.generics), s.ctor_kind, &s.fields, "", true, cx);
    });

    document(w, cx, it, None, HeadingOffset::H2);

    let mut fields = s
        .fields
        .iter()
        .filter_map(|f| match *f.kind {
            clean::StructFieldItem(ref ty) => Some((f, ty)),
            _ => None,
        })
        .peekable();
    if let None | Some(CtorKind::Fn) = s.ctor_kind {
        if fields.peek().is_some() {
            write!(
                w,
                "<h2 id=\"fields\" class=\"fields small-section-header\">\
                     {}{}<a href=\"#fields\" class=\"anchor\">§</a>\
                 </h2>",
                if s.ctor_kind.is_none() { "Fields" } else { "Tuple Fields" },
                document_non_exhaustive_header(it)
            );
            document_non_exhaustive(w, it);
            for (index, (field, ty)) in fields.enumerate() {
                let field_name =
                    field.name.map_or_else(|| index.to_string(), |sym| sym.as_str().to_string());
                let id = cx.derive_id(format!("{}.{}", ItemType::StructField, field_name));
                write!(
                    w,
                    "<span id=\"{id}\" class=\"{item_type} small-section-header\">\
                         <a href=\"#{id}\" class=\"anchor field\">§</a>\
                         <code>{field_name}: {ty}</code>\
                     </span>",
                    item_type = ItemType::StructField,
                    ty = ty.print(cx)
                );
                document(w, cx, field, Some(it), HeadingOffset::H3);
            }
        }
    }
    let def_id = it.item_id.expect_def_id();
    render_assoc_items(w, cx, it, def_id, AssocItemRender::All);
    document_type_layout(w, cx, def_id);
}

fn item_static(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item, s: &clean::Static) {
    wrap_item(w, |w| {
        render_attributes_in_code(w, it);
        write!(
            w,
            "{vis}static {mutability}{name}: {typ}",
            vis = visibility_print_with_space(it.visibility(cx.tcx()), it.item_id, cx),
            mutability = s.mutability.print_with_space(),
            name = it.name.unwrap(),
            typ = s.type_.print(cx)
        );
    });
    document(w, cx, it, None, HeadingOffset::H2)
}

fn item_foreign_type(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item) {
    wrap_item(w, |w| {
        w.write_str("extern {\n");
        render_attributes_in_code(w, it);
        write!(
            w,
            "    {}type {};\n}}",
            visibility_print_with_space(it.visibility(cx.tcx()), it.item_id, cx),
            it.name.unwrap(),
        );
    });

    document(w, cx, it, None, HeadingOffset::H2);

    render_assoc_items(w, cx, it, it.item_id.expect_def_id(), AssocItemRender::All)
}

fn item_keyword(w: &mut Buffer, cx: &mut Context<'_>, it: &clean::Item) {
    document(w, cx, it, None, HeadingOffset::H2)
}

/// Compare two strings treating multi-digit numbers as single units (i.e. natural sort order).
pub(crate) fn compare_names(mut lhs: &str, mut rhs: &str) -> Ordering {
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

pub(super) fn full_path(cx: &Context<'_>, item: &clean::Item) -> String {
    let mut s = join_with_double_colon(&cx.current);
    s.push_str("::");
    s.push_str(item.name.unwrap().as_str());
    s
}

pub(super) fn item_path(ty: ItemType, name: &str) -> String {
    match ty {
        ItemType::Module => format!("{}index.html", ensure_trailing_slash(name)),
        _ => format!("{}.{}.html", ty, name),
    }
}

fn bounds(t_bounds: &[clean::GenericBound], trait_alias: bool, cx: &Context<'_>) -> String {
    let mut bounds = String::new();
    if !t_bounds.is_empty() {
        if !trait_alias {
            bounds.push_str(": ");
        }
        for (i, p) in t_bounds.iter().enumerate() {
            if i > 0 {
                bounds.push_str(" + ");
            }
            bounds.push_str(&p.print(cx).to_string());
        }
    }
    bounds
}

fn wrap_item<F>(w: &mut Buffer, f: F)
where
    F: FnOnce(&mut Buffer),
{
    w.write_str(r#"<pre class="rust item-decl"><code>"#);
    f(w);
    w.write_str("</code></pre>");
}

fn compare_impl<'a, 'b>(lhs: &'a &&Impl, rhs: &'b &&Impl, cx: &Context<'_>) -> Ordering {
    let lhss = format!("{}", lhs.inner_impl().print(false, cx));
    let rhss = format!("{}", rhs.inner_impl().print(false, cx));

    // lhs and rhs are formatted as HTML, which may be unnecessary
    compare_names(&lhss, &rhss)
}

fn render_implementor(
    cx: &mut Context<'_>,
    implementor: &Impl,
    trait_: &clean::Item,
    w: &mut Buffer,
    implementor_dups: &FxHashMap<Symbol, (DefId, bool)>,
    aliases: &[String],
) {
    // If there's already another implementor that has the same abridged name, use the
    // full path, for example in `std::iter::ExactSizeIterator`
    let use_absolute = match implementor.inner_impl().for_ {
        clean::Type::Path { ref path, .. }
        | clean::BorrowedRef { type_: box clean::Type::Path { ref path, .. }, .. }
            if !path.is_assoc_ty() =>
        {
            implementor_dups[&path.last()].1
        }
        _ => false,
    };
    render_impl(
        w,
        cx,
        implementor,
        trait_,
        AssocItemLink::Anchor(None),
        RenderMode::Normal,
        Some(use_absolute),
        aliases,
        ImplRenderingParameters {
            show_def_docs: false,
            show_default_items: false,
            show_non_assoc_items: false,
            toggle_open_by_default: false,
        },
    );
}

fn render_union(
    w: &mut Buffer,
    it: &clean::Item,
    g: Option<&clean::Generics>,
    fields: &[clean::Item],
    cx: &Context<'_>,
) {
    let tcx = cx.tcx();
    write!(
        w,
        "{}union {}",
        visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
        it.name.unwrap(),
    );

    let where_displayed = g
        .map(|g| {
            write!(w, "{}", g.print(cx));
            print_where_clause_and_check(w, g, cx)
        })
        .unwrap_or(false);

    // If there wasn't a `where` clause, we add a whitespace.
    if !where_displayed {
        w.write_str(" ");
    }

    write!(w, "{{\n");
    let count_fields =
        fields.iter().filter(|f| matches!(*f.kind, clean::StructFieldItem(..))).count();
    let toggle = should_hide_fields(count_fields);
    if toggle {
        toggle_open(w, format_args!("{} fields", count_fields));
    }

    for field in fields {
        if let clean::StructFieldItem(ref ty) = *field.kind {
            write!(
                w,
                "    {}{}: {},\n",
                visibility_print_with_space(field.visibility(tcx), field.item_id, cx),
                field.name.unwrap(),
                ty.print(cx)
            );
        }
    }

    if it.has_stripped_entries().unwrap() {
        write!(w, "    /* private fields */\n");
    }
    if toggle {
        toggle_close(w);
    }
    w.write_str("}");
}

fn render_struct(
    w: &mut Buffer,
    it: &clean::Item,
    g: Option<&clean::Generics>,
    ty: Option<CtorKind>,
    fields: &[clean::Item],
    tab: &str,
    structhead: bool,
    cx: &Context<'_>,
) {
    let tcx = cx.tcx();
    write!(
        w,
        "{}{}{}",
        visibility_print_with_space(it.visibility(tcx), it.item_id, cx),
        if structhead { "struct " } else { "" },
        it.name.unwrap()
    );
    if let Some(g) = g {
        write!(w, "{}", g.print(cx))
    }
    match ty {
        None => {
            let where_diplayed = g.map(|g| print_where_clause_and_check(w, g, cx)).unwrap_or(false);

            // If there wasn't a `where` clause, we add a whitespace.
            if !where_diplayed {
                w.write_str(" {");
            } else {
                w.write_str("{");
            }
            let count_fields =
                fields.iter().filter(|f| matches!(*f.kind, clean::StructFieldItem(..))).count();
            let has_visible_fields = count_fields > 0;
            let toggle = should_hide_fields(count_fields);
            if toggle {
                toggle_open(w, format_args!("{} fields", count_fields));
            }
            for field in fields {
                if let clean::StructFieldItem(ref ty) = *field.kind {
                    write!(
                        w,
                        "\n{}    {}{}: {},",
                        tab,
                        visibility_print_with_space(field.visibility(tcx), field.item_id, cx),
                        field.name.unwrap(),
                        ty.print(cx),
                    );
                }
            }

            if has_visible_fields {
                if it.has_stripped_entries().unwrap() {
                    write!(w, "\n{}    /* private fields */", tab);
                }
                write!(w, "\n{}", tab);
            } else if it.has_stripped_entries().unwrap() {
                write!(w, " /* private fields */ ");
            }
            if toggle {
                toggle_close(w);
            }
            w.write_str("}");
        }
        Some(CtorKind::Fn) => {
            w.write_str("(");
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    w.write_str(", ");
                }
                match *field.kind {
                    clean::StrippedItem(box clean::StructFieldItem(..)) => write!(w, "_"),
                    clean::StructFieldItem(ref ty) => {
                        write!(
                            w,
                            "{}{}",
                            visibility_print_with_space(field.visibility(tcx), field.item_id, cx),
                            ty.print(cx),
                        )
                    }
                    _ => unreachable!(),
                }
            }
            w.write_str(")");
            if let Some(g) = g {
                write!(w, "{}", print_where_clause(g, cx, 0, Ending::NoNewline));
            }
            // We only want a ";" when we are displaying a tuple struct, not a variant tuple struct.
            if structhead {
                w.write_str(";");
            }
        }
        Some(CtorKind::Const) => {
            // Needed for PhantomData.
            if let Some(g) = g {
                write!(w, "{}", print_where_clause(g, cx, 0, Ending::NoNewline));
            }
            w.write_str(";");
        }
    }
}

fn document_non_exhaustive_header(item: &clean::Item) -> &str {
    if item.is_non_exhaustive() { " (Non-exhaustive)" } else { "" }
}

fn document_non_exhaustive(w: &mut Buffer, item: &clean::Item) {
    if item.is_non_exhaustive() {
        write!(
            w,
            "<details class=\"toggle non-exhaustive\">\
                 <summary class=\"hideme\"><span>{}</span></summary>\
                 <div class=\"docblock\">",
            {
                if item.is_struct() {
                    "This struct is marked as non-exhaustive"
                } else if item.is_enum() {
                    "This enum is marked as non-exhaustive"
                } else if item.is_variant() {
                    "This variant is marked as non-exhaustive"
                } else {
                    "This type is marked as non-exhaustive"
                }
            }
        );

        if item.is_struct() {
            w.write_str(
                "Non-exhaustive structs could have additional fields added in future. \
                 Therefore, non-exhaustive structs cannot be constructed in external crates \
                 using the traditional <code>Struct { .. }</code> syntax; cannot be \
                 matched against without a wildcard <code>..</code>; and \
                 struct update syntax will not work.",
            );
        } else if item.is_enum() {
            w.write_str(
                "Non-exhaustive enums could have additional variants added in future. \
                 Therefore, when matching against variants of non-exhaustive enums, an \
                 extra wildcard arm must be added to account for any future variants.",
            );
        } else if item.is_variant() {
            w.write_str(
                "Non-exhaustive enum variants could have additional fields added in future. \
                 Therefore, non-exhaustive enum variants cannot be constructed in external \
                 crates and cannot be matched against.",
            );
        } else {
            w.write_str(
                "This type will require a wildcard arm in any match statements or constructors.",
            );
        }

        w.write_str("</div></details>");
    }
}

fn document_type_layout(w: &mut Buffer, cx: &Context<'_>, ty_def_id: DefId) {
    fn write_size_of_layout(w: &mut Buffer, layout: &LayoutS, tag_size: u64) {
        if layout.abi.is_unsized() {
            write!(w, "(unsized)");
        } else {
            let size = layout.size.bytes() - tag_size;
            write!(w, "{size} byte{pl}", pl = if size == 1 { "" } else { "s" },);
            if layout.abi.is_uninhabited() {
                write!(
                    w,
                    " (<a href=\"https://doc.rust-lang.org/stable/reference/glossary.html#uninhabited\">uninhabited</a>)"
                );
            }
        }
    }

    if !cx.shared.show_type_layout {
        return;
    }

    writeln!(
        w,
        "<h2 id=\"layout\" class=\"small-section-header\"> \
        Layout<a href=\"#layout\" class=\"anchor\">§</a></h2>"
    );
    writeln!(w, "<div class=\"docblock\">");

    let tcx = cx.tcx();
    let param_env = tcx.param_env(ty_def_id);
    let ty = tcx.type_of(ty_def_id).subst_identity();
    match tcx.layout_of(param_env.and(ty)) {
        Ok(ty_layout) => {
            writeln!(
                w,
                "<div class=\"warning\"><p><strong>Note:</strong> Most layout information is \
                 <strong>completely unstable</strong> and may even differ between compilations. \
                 The only exception is types with certain <code>repr(...)</code> attributes. \
                 Please see the Rust Reference’s \
                 <a href=\"https://doc.rust-lang.org/reference/type-layout.html\">“Type Layout”</a> \
                 chapter for details on type layout guarantees.</p></div>"
            );
            w.write_str("<p><strong>Size:</strong> ");
            write_size_of_layout(w, &ty_layout.layout.0, 0);
            writeln!(w, "</p>");
            if let Variants::Multiple { variants, tag, tag_encoding, .. } =
                &ty_layout.layout.variants()
            {
                if !variants.is_empty() {
                    w.write_str(
                        "<p><strong>Size for each variant:</strong></p>\
                            <ul>",
                    );

                    let Adt(adt, _) = ty_layout.ty.kind() else {
                        span_bug!(tcx.def_span(ty_def_id), "not an adt")
                    };

                    let tag_size = if let TagEncoding::Niche { .. } = tag_encoding {
                        0
                    } else if let Primitive::Int(i, _) = tag.primitive() {
                        i.size().bytes()
                    } else {
                        span_bug!(tcx.def_span(ty_def_id), "tag is neither niche nor int")
                    };

                    for (index, layout) in variants.iter_enumerated() {
                        let name = adt.variant(index).name;
                        write!(w, "<li><code>{name}</code>: ");
                        write_size_of_layout(w, layout, tag_size);
                        writeln!(w, "</li>");
                    }
                    w.write_str("</ul>");
                }
            }
        }
        // This kind of layout error can occur with valid code, e.g. if you try to
        // get the layout of a generic type such as `Vec<T>`.
        Err(LayoutError::Unknown(_)) => {
            writeln!(
                w,
                "<p><strong>Note:</strong> Unable to compute type layout, \
                 possibly due to this type having generic parameters. \
                 Layout can only be computed for concrete, fully-instantiated types.</p>"
            );
        }
        // This kind of error probably can't happen with valid code, but we don't
        // want to panic and prevent the docs from building, so we just let the
        // user know that we couldn't compute the layout.
        Err(LayoutError::SizeOverflow(_)) => {
            writeln!(
                w,
                "<p><strong>Note:</strong> Encountered an error during type layout; \
                 the type was too big.</p>"
            );
        }
        Err(LayoutError::NormalizationFailure(_, _)) => {
            writeln!(
                w,
                "<p><strong>Note:</strong> Encountered an error during type layout; \
                the type failed to be normalized.</p>"
            )
        }
    }

    writeln!(w, "</div>");
}

fn pluralize(count: usize) -> &'static str {
    if count > 1 { "s" } else { "" }
}
