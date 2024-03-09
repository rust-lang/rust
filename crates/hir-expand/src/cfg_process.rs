//! Processes out #[cfg] and #[cfg_attr] attributes from the input for the derive macro
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, Attr, HasAttrs, Meta, VariantList},
    AstNode, SyntaxElement, SyntaxNode, T,
};
use tracing::{info, warn};

use crate::{db::ExpandDatabase, MacroCallKind, MacroCallLoc};

fn check_cfg_attr(attr: &Attr, loc: &MacroCallLoc, db: &dyn ExpandDatabase) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg")? {
        return None;
    }
    info!("Evaluating cfg {}", attr);
    let cfg = cfg::CfgExpr::parse_from_attr_meta(attr.meta()?)?;
    info!("Checking cfg {:?}", cfg);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&cfg) != Some(false);
    Some(enabled)
}

fn check_cfg_attr_attr(attr: &Attr, loc: &MacroCallLoc, db: &dyn ExpandDatabase) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg_attr")? {
        return None;
    }
    info!("Evaluating cfg_attr {}", attr);

    let cfg_expr = cfg::CfgExpr::parse_from_attr_meta(attr.meta()?)?;
    info!("Checking cfg_attr {:?}", cfg_expr);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&cfg_expr) != Some(false);
    Some(enabled)
}

fn process_has_attrs_with_possible_comma<I: HasAttrs>(
    items: impl Iterator<Item = I>,
    loc: &MacroCallLoc,
    db: &dyn ExpandDatabase,
    remove: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    for item in items {
        let field_attrs = item.attrs();
        'attrs: for attr in field_attrs {
            if let Some(enabled) = check_cfg_attr(&attr, loc, db) {
                // Rustc does not strip the attribute if it is enabled. So we will will leave it
                if !enabled {
                    info!("censoring type {:?}", item.syntax());
                    remove.insert(item.syntax().clone().into());
                    // We need to remove the , as well
                    add_comma(&item, remove);
                    break 'attrs;
                }
            };

            if let Some(enabled) = check_cfg_attr_attr(&attr, loc, db) {
                if enabled {
                    info!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    info!("censoring type cfg_attr {:?}", item.syntax());
                    remove.insert(attr.syntax().clone().into());
                    continue;
                }
            }
        }
    }
    Some(())
}

fn remove_tokens_within_cfg_attr(meta: Meta) -> Option<FxHashSet<SyntaxElement>> {
    let mut remove: FxHashSet<SyntaxElement> = FxHashSet::default();
    info!("Enabling attribute {}", meta);
    let meta_path = meta.path()?;
    info!("Removing {:?}", meta_path.syntax());
    remove.insert(meta_path.syntax().clone().into());

    let meta_tt = meta.token_tree()?;
    info!("meta_tt {}", meta_tt);
    // Remove the left paren
    remove.insert(meta_tt.l_paren_token()?.into());
    let mut found_comma = false;
    for tt in meta_tt.token_trees_and_tokens().skip(1) {
        info!("Checking {:?}", tt);
        // Check if it is a subtree or a token. If it is a token check if it is a comma. If so, remove it and break.
        match tt {
            syntax::NodeOrToken::Node(node) => {
                // Remove the entire subtree
                remove.insert(node.syntax().clone().into());
            }
            syntax::NodeOrToken::Token(token) => {
                if token.kind() == T![,] {
                    found_comma = true;
                    remove.insert(token.into());
                    break;
                }
                remove.insert(token.into());
            }
        }
    }
    if !found_comma {
        warn!("No comma found in {}", meta_tt);
        return None;
    }
    // Remove the right paren
    remove.insert(meta_tt.r_paren_token()?.into());
    Some(remove)
}
fn add_comma(item: &impl AstNode, res: &mut FxHashSet<SyntaxElement>) {
    if let Some(comma) = item.syntax().next_sibling_or_token().filter(|it| it.kind() == T![,]) {
        res.insert(comma);
    }
}
fn process_enum(
    variants: VariantList,
    loc: &MacroCallLoc,
    db: &dyn ExpandDatabase,
    remove: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    'variant: for variant in variants.variants() {
        for attr in variant.attrs() {
            if let Some(enabled) = check_cfg_attr(&attr, loc, db) {
                // Rustc does not strip the attribute if it is enabled. So we will will leave it
                if !enabled {
                    info!("censoring type {:?}", variant.syntax());
                    remove.insert(variant.syntax().clone().into());
                    // We need to remove the , as well
                    add_comma(&variant, remove);
                    continue 'variant;
                }
            };

            if let Some(enabled) = check_cfg_attr_attr(&attr, loc, db) {
                if enabled {
                    info!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    info!("censoring type cfg_attr {:?}", variant.syntax());
                    remove.insert(attr.syntax().clone().into());
                    continue;
                }
            }
        }
        if let Some(fields) = variant.field_list() {
            match fields {
                ast::FieldList::RecordFieldList(fields) => {
                    process_has_attrs_with_possible_comma(fields.fields(), loc, db, remove)?;
                }
                ast::FieldList::TupleFieldList(fields) => {
                    process_has_attrs_with_possible_comma(fields.fields(), loc, db, remove)?;
                }
            }
        }
    }
    Some(())
}

pub(crate) fn process_cfg_attrs(
    node: &SyntaxNode,
    loc: &MacroCallLoc,
    db: &dyn ExpandDatabase,
) -> Option<FxHashSet<SyntaxElement>> {
    // FIXME: #[cfg_eval] is not implemented. But it is not stable yet
    if !matches!(loc.kind, MacroCallKind::Derive { .. }) {
        return None;
    }
    let mut res = FxHashSet::default();

    let item = ast::Item::cast(node.clone())?;
    match item {
        ast::Item::Struct(it) => match it.field_list()? {
            ast::FieldList::RecordFieldList(fields) => {
                process_has_attrs_with_possible_comma(fields.fields(), loc, db, &mut res)?;
            }
            ast::FieldList::TupleFieldList(fields) => {
                process_has_attrs_with_possible_comma(fields.fields(), loc, db, &mut res)?;
            }
        },
        ast::Item::Enum(it) => {
            process_enum(it.variant_list()?, loc, db, &mut res)?;
        }
        ast::Item::Union(it) => {
            process_has_attrs_with_possible_comma(
                it.record_field_list()?.fields(),
                loc,
                db,
                &mut res,
            )?;
        }
        // FIXME: Implement for other items if necessary. As we do not support #[cfg_eval] yet, we do not need to implement it for now
        _ => {}
    }
    Some(res)
}
