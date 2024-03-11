//! Processes out #[cfg] and #[cfg_attr] attributes from the input for the derive macro
use std::iter::Peekable;

use cfg::{CfgAtom, CfgExpr};
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, Attr, HasAttrs, Meta, VariantList},
    AstNode, NodeOrToken, SyntaxElement, SyntaxNode, T,
};
use tracing::{debug, warn};
use tt::SmolStr;

use crate::{db::ExpandDatabase, MacroCallKind, MacroCallLoc};

fn check_cfg_attr(attr: &Attr, loc: &MacroCallLoc, db: &dyn ExpandDatabase) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg")? {
        return None;
    }
    debug!("Evaluating cfg {}", attr);
    let cfg = parse_from_attr_meta(attr.meta()?)?;
    debug!("Checking cfg {:?}", cfg);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&cfg) != Some(false);
    Some(enabled)
}

fn check_cfg_attr_attr(attr: &Attr, loc: &MacroCallLoc, db: &dyn ExpandDatabase) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg_attr")? {
        return None;
    }
    debug!("Evaluating cfg_attr {}", attr);
    let cfg_expr = parse_from_attr_meta(attr.meta()?)?;
    debug!("Checking cfg_attr {:?}", cfg_expr);
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
            if check_cfg_attr(&attr, loc, db).map(|enabled| !enabled).unwrap_or_default() {
                debug!("censoring type {:?}", item.syntax());
                remove.insert(item.syntax().clone().into());
                // We need to remove the , as well
                remove_possible_comma(&item, remove);
                break 'attrs;
            }

            if let Some(enabled) = check_cfg_attr_attr(&attr, loc, db) {
                if enabled {
                    debug!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    debug!("censoring type cfg_attr {:?}", item.syntax());
                    remove.insert(attr.syntax().clone().into());
                    continue;
                }
            }
        }
    }
    Some(())
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum CfgExprStage {
    /// Stripping the CFGExpr part of the attribute
    StrippigCfgExpr,
    /// Found the comma after the CFGExpr. Will keep all tokens until the next comma or the end of the attribute
    FoundComma,
    /// Everything following the attribute. This could be another attribute or the end of the attribute.
    // FIXME: cfg_attr with multiple attributes will not be handled correctly. We will only keep the first attribute
    // Related Issue: https://github.com/rust-lang/rust-analyzer/issues/10110
    EverythingElse,
}
/// This function creates its own set of tokens to remove. To help prevent malformed syntax as input.
fn remove_tokens_within_cfg_attr(meta: Meta) -> Option<FxHashSet<SyntaxElement>> {
    let mut remove: FxHashSet<SyntaxElement> = FxHashSet::default();
    debug!("Enabling attribute {}", meta);
    let meta_path = meta.path()?;
    debug!("Removing {:?}", meta_path.syntax());
    remove.insert(meta_path.syntax().clone().into());

    let meta_tt = meta.token_tree()?;
    debug!("meta_tt {}", meta_tt);
    let mut stage = CfgExprStage::StrippigCfgExpr;
    for tt in meta_tt.token_trees_and_tokens() {
        debug!("Checking {:?}. Stage: {:?}", tt, stage);
        match (stage, tt) {
            (CfgExprStage::StrippigCfgExpr, syntax::NodeOrToken::Node(node)) => {
                remove.insert(node.syntax().clone().into());
            }
            (CfgExprStage::StrippigCfgExpr, syntax::NodeOrToken::Token(token)) => {
                if token.kind() == T![,] {
                    stage = CfgExprStage::FoundComma;
                }
                remove.insert(token.into());
            }
            (CfgExprStage::FoundComma, syntax::NodeOrToken::Token(token))
                if (token.kind() == T![,] || token.kind() == T![')']) =>
            {
                // The end of the attribute or separator for the next attribute
                stage = CfgExprStage::EverythingElse;
                remove.insert(token.into());
            }
            (CfgExprStage::EverythingElse, syntax::NodeOrToken::Node(node)) => {
                remove.insert(node.syntax().clone().into());
            }
            (CfgExprStage::EverythingElse, syntax::NodeOrToken::Token(token)) => {
                remove.insert(token.into());
            }
            // This is an actual attribute
            _ => {}
        }
    }
    if stage != CfgExprStage::EverythingElse {
        warn!("Invalid cfg_attr attribute. {:?}", meta_tt);
        return None;
    }
    Some(remove)
}
/// Removes a possible comma after the [AstNode]
fn remove_possible_comma(item: &impl AstNode, res: &mut FxHashSet<SyntaxElement>) {
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
            if check_cfg_attr(&attr, loc, db).map(|enabled| !enabled).unwrap_or_default() {
                // Rustc does not strip the attribute if it is enabled. So we will will leave it
                debug!("censoring type {:?}", variant.syntax());
                remove.insert(variant.syntax().clone().into());
                // We need to remove the , as well
                remove_possible_comma(&variant, remove);
                continue 'variant;
            };

            if let Some(enabled) = check_cfg_attr_attr(&attr, loc, db) {
                if enabled {
                    debug!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    debug!("censoring type cfg_attr {:?}", variant.syntax());
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
    let mut remove = FxHashSet::default();

    let item = ast::Item::cast(node.clone())?;
    for attr in item.attrs() {
        if let Some(enabled) = check_cfg_attr_attr(&attr, loc, db) {
            if enabled {
                debug!("Removing cfg_attr tokens {:?}", attr);
                let meta = attr.meta()?;
                let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                remove.extend(removes_from_cfg_attr);
            } else {
                debug!("censoring type cfg_attr {:?}", item.syntax());
                remove.insert(attr.syntax().clone().into());
                continue;
            }
        }
    }
    match item {
        ast::Item::Struct(it) => match it.field_list()? {
            ast::FieldList::RecordFieldList(fields) => {
                process_has_attrs_with_possible_comma(fields.fields(), loc, db, &mut remove)?;
            }
            ast::FieldList::TupleFieldList(fields) => {
                process_has_attrs_with_possible_comma(fields.fields(), loc, db, &mut remove)?;
            }
        },
        ast::Item::Enum(it) => {
            process_enum(it.variant_list()?, loc, db, &mut remove)?;
        }
        ast::Item::Union(it) => {
            process_has_attrs_with_possible_comma(
                it.record_field_list()?.fields(),
                loc,
                db,
                &mut remove,
            )?;
        }
        // FIXME: Implement for other items if necessary. As we do not support #[cfg_eval] yet, we do not need to implement it for now
        _ => {}
    }
    Some(remove)
}
/// Parses a `cfg` attribute from the meta
fn parse_from_attr_meta(meta: Meta) -> Option<CfgExpr> {
    let tt = meta.token_tree()?;
    let mut iter = tt.token_trees_and_tokens().skip(1).peekable();
    next_cfg_expr_from_syntax(&mut iter)
}

fn next_cfg_expr_from_syntax<I>(iter: &mut Peekable<I>) -> Option<CfgExpr>
where
    I: Iterator<Item = NodeOrToken<ast::TokenTree, syntax::SyntaxToken>>,
{
    let name = match iter.next() {
        None => return None,
        Some(NodeOrToken::Token(element)) => match element.kind() {
            syntax::T![ident] => SmolStr::new(element.text()),
            _ => return Some(CfgExpr::Invalid),
        },
        Some(_) => return Some(CfgExpr::Invalid),
    };
    let result = match name.as_str() {
        "all" | "any" | "not" => {
            let mut preds = Vec::new();
            let Some(NodeOrToken::Node(tree)) = iter.next() else {
                return Some(CfgExpr::Invalid);
            };
            let mut tree_iter = tree.token_trees_and_tokens().skip(1).peekable();
            while tree_iter
                .peek()
                .filter(
                    |element| matches!(element, NodeOrToken::Token(token) if (token.kind() != syntax::T![')'])),
                )
                .is_some()
            {
                let pred = next_cfg_expr_from_syntax(&mut tree_iter);
                if let Some(pred) = pred {
                    preds.push(pred);
                }
            }
            let group = match name.as_str() {
                "all" => CfgExpr::All(preds),
                "any" => CfgExpr::Any(preds),
                "not" => CfgExpr::Not(Box::new(preds.pop().unwrap_or(CfgExpr::Invalid))),
                _ => unreachable!(),
            };
            Some(group)
        }
        _ => match iter.peek() {
            Some(NodeOrToken::Token(element)) if (element.kind() == syntax::T![=]) => {
                iter.next();
                match iter.next() {
                    Some(NodeOrToken::Token(value_token))
                        if (value_token.kind() == syntax::SyntaxKind::STRING) =>
                    {
                        let value = value_token.text();
                        let value = SmolStr::new(value.trim_matches('"'));
                        Some(CfgExpr::Atom(CfgAtom::KeyValue { key: name, value }))
                    }
                    _ => None,
                }
            }
            _ => Some(CfgExpr::Atom(CfgAtom::Flag(name))),
        },
    };
    if let Some(NodeOrToken::Token(element)) = iter.peek() {
        if element.kind() == syntax::T![,] {
            iter.next();
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use cfg::DnfExpr;
    use expect_test::{expect, Expect};
    use syntax::{ast::Attr, AstNode, SourceFile};

    use crate::cfg_process::parse_from_attr_meta;

    fn check_dnf_from_syntax(input: &str, expect: Expect) {
        let parse = SourceFile::parse(input);
        let node = match parse.tree().syntax().descendants().find_map(Attr::cast) {
            Some(it) => it,
            None => {
                let node = std::any::type_name::<Attr>();
                panic!("Failed to make ast node `{node}` from text {input}")
            }
        };
        let node = node.clone_subtree();
        assert_eq!(node.syntax().text_range().start(), 0.into());

        let cfg = parse_from_attr_meta(node.meta().unwrap()).unwrap();
        let actual = format!("#![cfg({})]", DnfExpr::new(cfg));
        expect.assert_eq(&actual);
    }
    #[test]
    fn cfg_from_attr() {
        check_dnf_from_syntax(r#"#[cfg(test)]"#, expect![[r#"#![cfg(test)]"#]]);
        check_dnf_from_syntax(r#"#[cfg(not(never))]"#, expect![[r#"#![cfg(not(never))]"#]]);
    }
}
