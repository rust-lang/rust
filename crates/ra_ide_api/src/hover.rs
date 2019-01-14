use ra_db::{Cancelable, SyntaxDatabase};
use ra_syntax::{
    AstNode, SyntaxNode, TreeArc,
    ast::{self, NameOwner, VisibilityOwner},
    algo::{find_covering_node, find_node_at_offset, find_leaf_at_offset, visit::{visitor, Visitor}},
};

use crate::{db::RootDatabase, RangeInfo, FilePosition, FileRange, NavigationTarget};

pub(crate) fn hover(
    db: &RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<RangeInfo<String>>> {
    let file = db.source_file(position.file_id);
    let mut res = Vec::new();

    let mut range = None;
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset) {
        use crate::goto_definition::{ReferenceResult::*, reference_definition};
        let ref_result = reference_definition(db, position.file_id, name_ref)?;
        match ref_result {
            Exact(nav) => res.extend(doc_text_for(db, nav)?),
            Approximate(navs) => {
                let mut msg = String::from("Failed to exactly resolve the symbol. This is probably because rust_analyzer does not yet support glob imports or traits.");
                if !navs.is_empty() {
                    msg.push_str("  \nThese items were found instead:");
                }
                res.push(msg);
                for nav in navs {
                    res.extend(doc_text_for(db, nav)?)
                }
            }
        }
        if !res.is_empty() {
            range = Some(name_ref.syntax().range())
        }
    }
    if range.is_none() {
        let node = find_leaf_at_offset(file.syntax(), position.offset).find_map(|leaf| {
            leaf.ancestors()
                .find(|n| ast::Expr::cast(*n).is_some() || ast::Pat::cast(*n).is_some())
        });
        let node = ctry!(node);
        let frange = FileRange {
            file_id: position.file_id,
            range: node.range(),
        };
        res.extend(type_of(db, frange)?.map(Into::into));
        range = Some(node.range());
    };

    let range = ctry!(range);
    if res.is_empty() {
        return Ok(None);
    }
    let res = RangeInfo::new(range, res.join("\n\n---\n"));
    Ok(Some(res))
}

pub(crate) fn type_of(db: &RootDatabase, frange: FileRange) -> Cancelable<Option<String>> {
    let file = db.source_file(frange.file_id);
    let syntax = file.syntax();
    let leaf_node = find_covering_node(syntax, frange.range);
    // if we picked identifier, expand to pattern/expression
    let node = leaf_node
        .ancestors()
        .take_while(|it| it.range() == leaf_node.range())
        .find(|&it| ast::Expr::cast(it).is_some() || ast::Pat::cast(it).is_some())
        .unwrap_or(leaf_node);
    let parent_fn = ctry!(node.ancestors().find_map(ast::FnDef::cast));
    let function = ctry!(hir::source_binder::function_from_source(
        db,
        frange.file_id,
        parent_fn
    )?);
    let infer = function.infer(db)?;
    let syntax_mapping = function.body_syntax_mapping(db)?;
    if let Some(expr) = ast::Expr::cast(node).and_then(|e| syntax_mapping.node_expr(e)) {
        Ok(Some(infer[expr].to_string()))
    } else if let Some(pat) = ast::Pat::cast(node).and_then(|p| syntax_mapping.node_pat(p)) {
        Ok(Some(infer[pat].to_string()))
    } else {
        Ok(None)
    }
}

// FIXME: this should not really use navigation target. Rather, approximatelly
// resovled symbol should return a `DefId`.
fn doc_text_for(db: &RootDatabase, nav: NavigationTarget) -> Cancelable<Option<String>> {
    let result = match (nav.description(db), nav.docs(db)) {
        (Some(desc), Some(docs)) => Some("```rust\n".to_string() + &*desc + "\n```\n\n" + &*docs),
        (Some(desc), None) => Some("```rust\n".to_string() + &*desc + "\n```"),
        (None, Some(docs)) => Some(docs),
        _ => None,
    };

    Ok(result)
}

impl NavigationTarget {
    fn node(&self, db: &RootDatabase) -> Option<TreeArc<SyntaxNode>> {
        let source_file = db.source_file(self.file_id());
        let source_file = source_file.syntax();
        let node = source_file
            .descendants()
            .find(|node| node.kind() == self.kind() && node.range() == self.full_range())?
            .to_owned();
        Some(node)
    }

    fn docs(&self, db: &RootDatabase) -> Option<String> {
        let node = self.node(db)?;
        fn doc_comments<N: ast::DocCommentsOwner>(node: &N) -> Option<String> {
            let comments = node.doc_comment_text();
            if comments.is_empty() {
                None
            } else {
                Some(comments)
            }
        }

        visitor()
            .visit(doc_comments::<ast::FnDef>)
            .visit(doc_comments::<ast::StructDef>)
            .visit(doc_comments::<ast::EnumDef>)
            .visit(doc_comments::<ast::TraitDef>)
            .visit(doc_comments::<ast::Module>)
            .visit(doc_comments::<ast::TypeDef>)
            .visit(doc_comments::<ast::ConstDef>)
            .visit(doc_comments::<ast::StaticDef>)
            .accept(&node)?
    }

    /// Get a description of this node.
    ///
    /// e.g. `struct Name`, `enum Name`, `fn Name`
    fn description(&self, db: &RootDatabase) -> Option<String> {
        // TODO: After type inference is done, add type information to improve the output
        let node = self.node(db)?;
        // TODO: Refactor to be have less repetition
        visitor()
            .visit(|node: &ast::FnDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("fn ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::StructDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("struct ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::EnumDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("enum ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::TraitDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("trait ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::Module| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("mod ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::TypeDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("type ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::ConstDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("const ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .visit(|node: &ast::StaticDef| {
                let mut string = node
                    .visibility()
                    .map(|v| format!("{} ", v.syntax().text()))
                    .unwrap_or_default();
                string.push_str("static ");
                node.name()?.syntax().text().push_to(&mut string);
                Some(string)
            })
            .accept(&node)?
    }
}

#[cfg(test)]
mod tests {
    use ra_syntax::TextRange;
    use crate::mock_analysis::{single_file_with_position, single_file_with_range};

    #[test]
    fn hover_shows_type_of_an_expression() {
        let (analysis, position) = single_file_with_position(
            "
            pub fn foo() -> u32 { 1 }

            fn main() {
                let foo_test = foo()<|>;
            }
            ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.range, TextRange::from_to(95.into(), 100.into()));
        assert_eq!(hover.info, "u32");
    }

    #[test]
    fn hover_for_local_variable() {
        let (analysis, position) = single_file_with_position("fn func(foo: i32) { fo<|>o; }");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.info, "i32");
    }

    #[test]
    fn hover_for_local_variable_pat() {
        let (analysis, position) = single_file_with_position("fn func(fo<|>o: i32) {}");
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.info, "i32");
    }

    #[test]
    fn test_type_of_for_function() {
        let (analysis, range) = single_file_with_range(
            "
            pub fn foo() -> u32 { 1 };

            fn main() {
                let foo_test = <|>foo()<|>;
            }
            ",
        );

        let type_name = analysis.type_of(range).unwrap().unwrap();
        assert_eq!("u32", &type_name);
    }

    // FIXME: improve type_of to make this work
    #[test]
    fn test_type_of_for_expr_1() {
        let (analysis, range) = single_file_with_range(
            "
            fn main() {
                let foo = <|>1 + foo_test<|>;
            }
            ",
        );

        let type_name = analysis.type_of(range).unwrap().unwrap();
        assert_eq!("[unknown]", &type_name);
    }

    // FIXME: improve type_of to make this work
    #[test]
    fn test_type_of_for_expr_2() {
        let (analysis, range) = single_file_with_range(
            "
            fn main() {
                let foo: usize = 1;
                let bar = <|>1 + foo_test<|>;
            }
            ",
        );

        let type_name = analysis.type_of(range).unwrap().unwrap();
        assert_eq!("[unknown]", &type_name);
    }

}
