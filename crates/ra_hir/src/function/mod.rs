mod scope;

use std::{
    cmp::{max, min},
    sync::Arc,
};

use ra_syntax::{
    TextRange, TextUnit, SyntaxNodeRef,
    ast::{self, AstNode, DocCommentsOwner, NameOwner},
};
use ra_db::FileId;

use crate::{
    Cancelable,
    DefLoc, DefKind, DefId, HirDatabase, SourceItemId,
    Module,
};

pub use self::scope::FnScopes;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId(pub(crate) DefId);

pub struct Function {
    fn_id: FnId,
}

impl Function {
    pub(crate) fn new(def_id: DefId) -> Function {
        let fn_id = FnId(def_id);
        Function { fn_id }
    }

    pub fn guess_from_source(
        db: &impl HirDatabase,
        file_id: FileId,
        fn_def: ast::FnDef,
    ) -> Cancelable<Option<Function>> {
        let module = ctry!(Module::guess_from_child_node(db, file_id, fn_def.syntax())?);
        let file_items = db.file_items(file_id);
        let item_id = file_items.id_of(fn_def.syntax());
        let source_item_id = SourceItemId { file_id, item_id };
        let def_loc = DefLoc {
            kind: DefKind::Function,
            source_root_id: module.source_root_id,
            module_id: module.module_id,
            source_item_id,
        };
        Ok(Some(Function::new(def_loc.id(db))))
    }

    pub fn guess_for_name_ref(
        db: &impl HirDatabase,
        file_id: FileId,
        name_ref: ast::NameRef,
    ) -> Cancelable<Option<Function>> {
        Function::guess_for_node(db, file_id, name_ref.syntax())
    }

    pub fn guess_for_bind_pat(
        db: &impl HirDatabase,
        file_id: FileId,
        bind_pat: ast::BindPat,
    ) -> Cancelable<Option<Function>> {
        Function::guess_for_node(db, file_id, bind_pat.syntax())
    }

    fn guess_for_node(
        db: &impl HirDatabase,
        file_id: FileId,
        node: SyntaxNodeRef,
    ) -> Cancelable<Option<Function>> {
        let fn_def = ctry!(node.ancestors().find_map(ast::FnDef::cast));
        Function::guess_from_source(db, file_id, fn_def)
    }

    pub fn scope(&self, db: &impl HirDatabase) -> Arc<FnScopes> {
        db.fn_scopes(self.fn_id)
    }

    pub fn signature_info(&self, db: &impl HirDatabase) -> Option<FnSignatureInfo> {
        let syntax = db.fn_syntax(self.fn_id);
        FnSignatureInfo::new(syntax.borrowed())
    }
}

#[derive(Debug, Clone)]
pub struct FnSignatureInfo {
    pub name: String,
    pub label: String,
    pub ret_type: Option<String>,
    pub params: Vec<String>,
    pub doc: Option<String>,
}

impl FnSignatureInfo {
    fn new(node: ast::FnDef) -> Option<Self> {
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

        if let Some((comment_range, docs)) = FnSignatureInfo::extract_doc_comments(node) {
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

        let params = FnSignatureInfo::param_list(node);
        let ret_type = node.ret_type().map(|r| r.syntax().text().to_string());

        Some(FnSignatureInfo {
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
