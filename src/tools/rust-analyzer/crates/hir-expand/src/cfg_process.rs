//! Processes out #[cfg] and #[cfg_attr] attributes from the input for the derive macro
use std::iter::Peekable;

use base_db::Crate;
use cfg::{CfgAtom, CfgExpr};
use intern::{Symbol, sym};
use rustc_hash::FxHashSet;
use syntax::{
    AstNode, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxNode, T,
    ast::{self, Attr, HasAttrs, Meta, TokenTree, VariantList},
};
use tracing::{debug, warn};

use crate::{MacroCallLoc, MacroDefKind, db::ExpandDatabase, proc_macro::ProcMacroKind};

fn check_cfg(db: &dyn ExpandDatabase, attr: &Attr, krate: Crate) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg")? {
        return None;
    }
    let cfg = parse_from_attr_token_tree(&attr.meta()?.token_tree()?)?;
    let enabled = krate.cfg_options(db).check(&cfg) != Some(false);
    Some(enabled)
}

fn check_cfg_attr(db: &dyn ExpandDatabase, attr: &Attr, krate: Crate) -> Option<bool> {
    if !attr.simple_name().as_deref().map(|v| v == "cfg_attr")? {
        return None;
    }
    check_cfg_attr_value(db, &attr.token_tree()?, krate)
}

pub fn check_cfg_attr_value(
    db: &dyn ExpandDatabase,
    attr: &TokenTree,
    krate: Crate,
) -> Option<bool> {
    let cfg_expr = parse_from_attr_token_tree(attr)?;
    let enabled = krate.cfg_options(db).check(&cfg_expr) != Some(false);
    Some(enabled)
}

fn process_has_attrs_with_possible_comma<I: HasAttrs>(
    db: &dyn ExpandDatabase,
    items: impl Iterator<Item = I>,
    krate: Crate,
    remove: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    for item in items {
        let field_attrs = item.attrs();
        'attrs: for attr in field_attrs {
            if let Some(enabled) = check_cfg(db, &attr, krate) {
                if enabled {
                    debug!("censoring {:?}", attr.syntax());
                    remove.insert(attr.syntax().clone().into());
                } else {
                    debug!("censoring {:?}", item.syntax());
                    remove.insert(item.syntax().clone().into());
                    // We need to remove the , as well
                    remove_possible_comma(&item, remove);
                    break 'attrs;
                }
            }

            if let Some(enabled) = check_cfg_attr(db, &attr, krate) {
                if enabled {
                    debug!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    debug!("censoring type cfg_attr {:?}", item.syntax());
                    remove.insert(attr.syntax().clone().into());
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
    db: &dyn ExpandDatabase,
    variants: VariantList,
    krate: Crate,
    remove: &mut FxHashSet<SyntaxElement>,
) -> Option<()> {
    'variant: for variant in variants.variants() {
        for attr in variant.attrs() {
            if let Some(enabled) = check_cfg(db, &attr, krate) {
                if enabled {
                    debug!("censoring {:?}", attr.syntax());
                    remove.insert(attr.syntax().clone().into());
                } else {
                    // Rustc does not strip the attribute if it is enabled. So we will leave it
                    debug!("censoring type {:?}", variant.syntax());
                    remove.insert(variant.syntax().clone().into());
                    // We need to remove the , as well
                    remove_possible_comma(&variant, remove);
                    continue 'variant;
                }
            }

            if let Some(enabled) = check_cfg_attr(db, &attr, krate) {
                if enabled {
                    debug!("Removing cfg_attr tokens {:?}", attr);
                    let meta = attr.meta()?;
                    let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                    remove.extend(removes_from_cfg_attr);
                } else {
                    debug!("censoring type cfg_attr {:?}", variant.syntax());
                    remove.insert(attr.syntax().clone().into());
                }
            }
        }
        if let Some(fields) = variant.field_list() {
            match fields {
                ast::FieldList::RecordFieldList(fields) => {
                    process_has_attrs_with_possible_comma(db, fields.fields(), krate, remove)?;
                }
                ast::FieldList::TupleFieldList(fields) => {
                    process_has_attrs_with_possible_comma(db, fields.fields(), krate, remove)?;
                }
            }
        }
    }
    Some(())
}

pub(crate) fn process_cfg_attrs(
    db: &dyn ExpandDatabase,
    node: &SyntaxNode,
    loc: &MacroCallLoc,
) -> Option<FxHashSet<SyntaxElement>> {
    // FIXME: #[cfg_eval] is not implemented. But it is not stable yet
    let is_derive = match loc.def.kind {
        MacroDefKind::BuiltInDerive(..)
        | MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive) => true,
        MacroDefKind::BuiltInAttr(_, expander) => expander.is_derive(),
        _ => false,
    };
    let mut remove = FxHashSet::default();

    let item = ast::Item::cast(node.clone())?;
    for attr in item.attrs() {
        if let Some(enabled) = check_cfg_attr(db, &attr, loc.krate) {
            if enabled {
                debug!("Removing cfg_attr tokens {:?}", attr);
                let meta = attr.meta()?;
                let removes_from_cfg_attr = remove_tokens_within_cfg_attr(meta)?;
                remove.extend(removes_from_cfg_attr);
            } else {
                debug!("Removing type cfg_attr {:?}", item.syntax());
                remove.insert(attr.syntax().clone().into());
            }
        }
    }

    if is_derive {
        // Only derives get their code cfg-clean, normal attribute macros process only the cfg at their level
        // (cfg_attr is handled above, cfg is handled in the def map).
        match item {
            ast::Item::Struct(it) => match it.field_list()? {
                ast::FieldList::RecordFieldList(fields) => {
                    process_has_attrs_with_possible_comma(
                        db,
                        fields.fields(),
                        loc.krate,
                        &mut remove,
                    )?;
                }
                ast::FieldList::TupleFieldList(fields) => {
                    process_has_attrs_with_possible_comma(
                        db,
                        fields.fields(),
                        loc.krate,
                        &mut remove,
                    )?;
                }
            },
            ast::Item::Enum(it) => {
                process_enum(db, it.variant_list()?, loc.krate, &mut remove)?;
            }
            ast::Item::Union(it) => {
                process_has_attrs_with_possible_comma(
                    db,
                    it.record_field_list()?.fields(),
                    loc.krate,
                    &mut remove,
                )?;
            }
            // FIXME: Implement for other items if necessary. As we do not support #[cfg_eval] yet, we do not need to implement it for now
            _ => {}
        }
    }
    Some(remove)
}
/// Parses a `cfg` attribute from the meta
fn parse_from_attr_token_tree(tt: &TokenTree) -> Option<CfgExpr> {
    let mut iter = tt
        .token_trees_and_tokens()
        .filter(is_not_whitespace)
        .skip(1)
        .take_while(is_not_closing_paren)
        .peekable();
    next_cfg_expr_from_syntax(&mut iter)
}

fn is_not_closing_paren(element: &NodeOrToken<ast::TokenTree, syntax::SyntaxToken>) -> bool {
    !matches!(element, NodeOrToken::Token(token) if (token.kind() == syntax::T![')']))
}
fn is_not_whitespace(element: &NodeOrToken<ast::TokenTree, syntax::SyntaxToken>) -> bool {
    !matches!(element, NodeOrToken::Token(token) if (token.kind() == SyntaxKind::WHITESPACE))
}

fn next_cfg_expr_from_syntax<I>(iter: &mut Peekable<I>) -> Option<CfgExpr>
where
    I: Iterator<Item = NodeOrToken<ast::TokenTree, syntax::SyntaxToken>>,
{
    let name = match iter.next() {
        None => return None,
        Some(NodeOrToken::Token(element)) => match element.kind() {
            syntax::T![ident] => Symbol::intern(element.text()),
            _ => return Some(CfgExpr::Invalid),
        },
        Some(_) => return Some(CfgExpr::Invalid),
    };
    let result = match &name {
        s if [&sym::all, &sym::any, &sym::not].contains(&s) => {
            let mut preds = Vec::new();
            let Some(NodeOrToken::Node(tree)) = iter.next() else {
                return Some(CfgExpr::Invalid);
            };
            let mut tree_iter = tree
                .token_trees_and_tokens()
                .filter(is_not_whitespace)
                .skip(1)
                .take_while(is_not_closing_paren)
                .peekable();
            while tree_iter.peek().is_some() {
                let pred = next_cfg_expr_from_syntax(&mut tree_iter);
                if let Some(pred) = pred {
                    preds.push(pred);
                }
            }
            let group = match &name {
                s if *s == sym::all => CfgExpr::All(preds.into_boxed_slice()),
                s if *s == sym::any => CfgExpr::Any(preds.into_boxed_slice()),
                s if *s == sym::not => {
                    CfgExpr::Not(Box::new(preds.pop().unwrap_or(CfgExpr::Invalid)))
                }
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
                        Some(CfgExpr::Atom(CfgAtom::KeyValue {
                            key: name,
                            value: Symbol::intern(value.trim_matches('"')),
                        }))
                    }
                    _ => None,
                }
            }
            _ => Some(CfgExpr::Atom(CfgAtom::Flag(name))),
        },
    };
    if let Some(NodeOrToken::Token(element)) = iter.peek()
        && element.kind() == syntax::T![,]
    {
        iter.next();
    }
    result
}
#[cfg(test)]
mod tests {
    use cfg::DnfExpr;
    use expect_test::{Expect, expect};
    use syntax::{AstNode, SourceFile, ast::Attr};

    use crate::cfg_process::parse_from_attr_token_tree;

    fn check_dnf_from_syntax(input: &str, expect: Expect) {
        let parse = SourceFile::parse(input, span::Edition::CURRENT);
        let node = match parse.tree().syntax().descendants().find_map(Attr::cast) {
            Some(it) => it,
            None => {
                let node = std::any::type_name::<Attr>();
                panic!("Failed to make ast node `{node}` from text {input}")
            }
        };
        let node = node.clone_subtree();
        assert_eq!(node.syntax().text_range().start(), 0.into());

        let cfg = parse_from_attr_token_tree(&node.meta().unwrap().token_tree().unwrap()).unwrap();
        let actual = format!("#![cfg({})]", DnfExpr::new(&cfg));
        expect.assert_eq(&actual);
    }
    #[test]
    fn cfg_from_attr() {
        check_dnf_from_syntax(r#"#[cfg(test)]"#, expect![[r#"#![cfg(test)]"#]]);
        check_dnf_from_syntax(r#"#[cfg(not(never))]"#, expect![[r#"#![cfg(not(never))]"#]]);
    }
}
