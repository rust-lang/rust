use std::os::windows::process;

use mbe::syntax_node_to_token_tree;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, Attr, FieldList, HasAttrs, RecordFieldList, TupleFieldList, Variant, VariantList},
    AstNode, SyntaxElement, SyntaxNode, T,
};
use tracing::info;

use crate::{db::ExpandDatabase, span_map::SpanMap, MacroCallLoc};

fn check_cfg_attr(
    attr: &Attr,
    loc: &MacroCallLoc,
    span_map: &SpanMap,
    db: &dyn ExpandDatabase,
) -> Option<bool> {
    attr.simple_name().as_deref().map(|v| v == "cfg")?;
    info!("Checking cfg attr {:?}", attr);
    let Some(tt) = attr.token_tree() else {
        info!("cfg attr has no expr {:?}", attr);
        return Some(true);
    };
    info!("Checking cfg {:?}", tt);
    let tt = tt.syntax().clone();
    // Convert to a tt::Subtree
    let tt = syntax_node_to_token_tree(&tt, span_map, loc.call_site);
    let cfg = cfg::CfgExpr::parse(&tt);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&cfg) != Some(false);
    Some(enabled)
}
enum CfgAttrResult {
    Enabled(Attr),
    Disabled,
}

fn check_cfg_attr_attr(
    attr: &Attr,
    loc: &MacroCallLoc,
    span_map: &SpanMap,
    db: &dyn ExpandDatabase,
) -> Option<CfgAttrResult> {
    attr.simple_name().as_deref().map(|v| v == "cfg_attr")?;
    info!("Checking cfg_attr attr {:?}", attr);
    let Some(tt) = attr.token_tree() else {
        info!("cfg_attr attr has no expr {:?}", attr);
        return None;
    };
    info!("Checking cfg_attr {:?}", tt);
    let tt = tt.syntax().clone();
    // Convert to a tt::Subtree
    let tt = syntax_node_to_token_tree(&tt, span_map, loc.call_site);
    let cfg = cfg::CfgExpr::parse(&tt);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&cfg) != Some(false);
    if enabled {
        // FIXME: Add the internal attribute
        Some(CfgAttrResult::Enabled(attr.clone()))
    } else {
        Some(CfgAttrResult::Disabled)
    }
}

fn process_has_attrs_with_possible_comma<I: HasAttrs>(
    items: impl Iterator<Item = I>,
    loc: &MacroCallLoc,
    span_map: &SpanMap,
    db: &dyn ExpandDatabase,
    res: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    for item in items {
        let field_attrs = item.attrs();
        'attrs: for attr in field_attrs {
            let Some(enabled) = check_cfg_attr(&attr, loc, span_map, db) else {
                continue;
            };
            if enabled {
                //FIXME: Should we remove the cfg_attr?
            } else {
                info!("censoring type {:?}", item.syntax());
                res.insert(item.syntax().clone().into());
                // We need to remove the , as well
                if let Some(comma) = item.syntax().next_sibling_or_token() {
                    if comma.kind() == T![,] {
                        res.insert(comma.into());
                    }
                }
                break 'attrs;
            }
            let Some(attr_result) = check_cfg_attr_attr(&attr, loc, span_map, db) else {
                continue;
            };
            match attr_result {
                CfgAttrResult::Enabled(attr) => {
                    //FIXME: Replace the attribute with the internal attribute
                }
                CfgAttrResult::Disabled => {
                    info!("censoring type {:?}", item.syntax());
                    res.insert(attr.syntax().clone().into());
                    continue;
                }
            }
        }
    }
    Some(())
}
fn process_enum(
    variants: VariantList,
    loc: &MacroCallLoc,
    span_map: &SpanMap,
    db: &dyn ExpandDatabase,
    res: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    for variant in variants.variants() {
        'attrs: for attr in variant.attrs() {
            if !check_cfg_attr(&attr, loc, span_map, db)? {
                info!("censoring variant {:?}", variant.syntax());
                res.insert(variant.syntax().clone().into());
                if let Some(comma) = variant.syntax().next_sibling_or_token() {
                    if comma.kind() == T![,] {
                        res.insert(comma.into());
                    }
                }
                break 'attrs;
            }
        }
        if let Some(fields) = variant.field_list() {
            match fields {
                ast::FieldList::RecordFieldList(fields) => {
                    process_has_attrs_with_possible_comma(fields.fields(), loc, span_map, db, res)?;
                }
                ast::FieldList::TupleFieldList(fields) => {
                    process_has_attrs_with_possible_comma(fields.fields(), loc, span_map, db, res)?;
                }
            }
        }
    }
    Some(())
}
/// Handle 
pub(crate) fn process_cfg_attrs(
    node: &SyntaxNode,
    loc: &MacroCallLoc,
    span_map: &SpanMap,
    db: &dyn ExpandDatabase,
) -> Option<FxHashSet<SyntaxElement>> {
    let mut res = FxHashSet::default();
    let item = ast::Item::cast(node.clone())?;
    match item {
        ast::Item::Struct(it) => match it.field_list()? {
            ast::FieldList::RecordFieldList(fields) => {
                process_has_attrs_with_possible_comma(
                    fields.fields(),
                    loc,
                    span_map,
                    db,
                    &mut res,
                )?;
            }
            ast::FieldList::TupleFieldList(fields) => {
                process_has_attrs_with_possible_comma(
                    fields.fields(),
                    loc,
                    span_map,
                    db,
                    &mut res,
                )?;
            }
        },
        ast::Item::Enum(it) => {
            process_enum(it.variant_list()?, loc, span_map, db, &mut res)?;
        }
        // FIXME: Implement for other items
        _ => {}
    }

    Some(res)
}
