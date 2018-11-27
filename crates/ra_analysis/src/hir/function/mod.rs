mod scope;

use std::{
    cmp::{max, min},
    sync::Arc,
};

use ra_syntax::{
    ast::{self, AstNode, DocCommentsOwner, NameOwner},
    TextRange, TextUnit,
};

use crate::{
    hir::HirDatabase,
    syntax_ptr::SyntaxPtr, FileId,
};

pub(crate) use self::scope::{resolve_local_name, FnScopes};
pub(crate) use crate::loc2id::FnId;

impl FnId {
    pub(crate) fn get(db: &impl HirDatabase, file_id: FileId, fn_def: ast::FnDef) -> FnId {
        let ptr = SyntaxPtr::new(file_id, fn_def.syntax());
        db.id_maps().fn_id(ptr)
    }
}

pub(crate) struct FunctionDescriptor {
    fn_id: FnId,
}

impl FunctionDescriptor {
    pub(crate) fn guess_from_source(
        db: &impl HirDatabase,
        file_id: FileId,
        fn_def: ast::FnDef,
    ) -> FunctionDescriptor {
        let fn_id = FnId::get(db, file_id, fn_def);
        FunctionDescriptor { fn_id }
    }

    pub(crate) fn scope(&self, db: &impl HirDatabase) -> Arc<FnScopes> {
        db.fn_scopes(self.fn_id)
    }
}

#[derive(Debug, Clone)]
pub struct FnDescriptor {
    pub name: String,
    pub label: String,
    pub ret_type: Option<String>,
    pub params: Vec<String>,
    pub doc: Option<String>,
}

impl FnDescriptor {
    pub fn new(node: ast::FnDef) -> Option<Self> {
        let name = node.name()?.text().to_string();

        let mut doc = None;

        // Strip the body out for the label.
        let mut label: String = if let Some(body) = node.body() {
            let body_range = body.syntax().range();
            let label: String = node
                .syntax()
                .children()
                .filter(|child| !child.range().is_subrange(&body_range))
                .map(|node| node.text().to_string())
                .collect();
            label
        } else {
            node.syntax().text().to_string()
        };

        if let Some((comment_range, docs)) = FnDescriptor::extract_doc_comments(node) {
            let comment_range = comment_range
                .checked_sub(node.syntax().range().start())
                .unwrap();
            let start = comment_range.start().to_usize();
            let end = comment_range.end().to_usize();

            // Remove the comment from the label
            label.replace_range(start..end, "");

            // Massage markdown
            let mut processed_lines = Vec::new();
            let mut in_code_block = false;
            for line in docs.lines() {
                if line.starts_with("```") {
                    in_code_block = !in_code_block;
                }

                let line = if in_code_block && line.starts_with("```") && !line.contains("rust") {
                    "```rust".into()
                } else {
                    line.to_string()
                };

                processed_lines.push(line);
            }

            if !processed_lines.is_empty() {
                doc = Some(processed_lines.join("\n"));
            }
        }

        let params = FnDescriptor::param_list(node);
        let ret_type = node.ret_type().map(|r| r.syntax().text().to_string());

        Some(FnDescriptor {
            name,
            ret_type,
            params,
            label: label.trim().to_owned(),
            doc,
        })
    }

    fn extract_doc_comments(node: ast::FnDef) -> Option<(TextRange, String)> {
        if node.doc_comments().count() == 0 {
            return None;
        }

        let comment_text = node.doc_comment_text();

        let (begin, end) = node
            .doc_comments()
            .map(|comment| comment.syntax().range())
            .map(|range| (range.start().to_usize(), range.end().to_usize()))
            .fold((std::usize::MAX, std::usize::MIN), |acc, range| {
                (min(acc.0, range.0), max(acc.1, range.1))
            });

        let range = TextRange::from_to(TextUnit::from_usize(begin), TextUnit::from_usize(end));

        Some((range, comment_text))
    }

    fn param_list(node: ast::FnDef) -> Vec<String> {
        let mut res = vec![];
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                res.push(self_param.syntax().text().to_string())
            }

            // Maybe use param.pat here? See if we can just extract the name?
            //res.extend(param_list.params().map(|p| p.syntax().text().to_string()));
            res.extend(
                param_list
                    .params()
                    .filter_map(|p| p.pat())
                    .map(|pat| pat.syntax().text().to_string()),
            );
        }
        res
    }
}
