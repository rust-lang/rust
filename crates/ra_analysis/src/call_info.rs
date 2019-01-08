use ra_db::{SyntaxDatabase, Cancelable};
use ra_syntax::{
    AstNode, SyntaxNode, TextUnit, TextRange,
    SyntaxKind::FN_DEF,
    ast::{self, ArgListOwner},
};
use ra_editor::find_node_at_offset;
use hir::FnSignatureInfo;

use crate::{FilePosition, db::RootDatabase};

/// Computes parameter information for the given call expression.
pub(crate) fn call_info(
    db: &RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<(FnSignatureInfo, Option<usize>)>> {
    let file = db.source_file(position.file_id);
    let syntax = file.syntax();

    // Find the calling expression and it's NameRef
    let calling_node = ctry!(FnCallNode::with_node(syntax, position.offset));
    let name_ref = ctry!(calling_node.name_ref());

    // Resolve the function's NameRef (NOTE: this isn't entirely accurate).
    let file_symbols = db.index_resolve(name_ref)?;
    for symbol in file_symbols {
        if symbol.ptr.kind() == FN_DEF {
            let fn_file = db.source_file(symbol.file_id);
            let fn_def = symbol.ptr.resolve(&fn_file);
            let fn_def = ast::FnDef::cast(&fn_def).unwrap();
            let descr = ctry!(hir::source_binder::function_from_source(
                db,
                symbol.file_id,
                fn_def
            )?);
            if let Some(descriptor) = descr.signature_info(db) {
                // If we have a calling expression let's find which argument we are on
                let mut current_parameter = None;

                let num_params = descriptor.params.len();
                let has_self = fn_def.param_list().and_then(|l| l.self_param()).is_some();

                if num_params == 1 {
                    if !has_self {
                        current_parameter = Some(0);
                    }
                } else if num_params > 1 {
                    // Count how many parameters into the call we are.
                    // TODO: This is best effort for now and should be fixed at some point.
                    // It may be better to see where we are in the arg_list and then check
                    // where offset is in that list (or beyond).
                    // Revisit this after we get documentation comments in.
                    if let Some(ref arg_list) = calling_node.arg_list() {
                        let start = arg_list.syntax().range().start();

                        let range_search = TextRange::from_to(start, position.offset);
                        let mut commas: usize = arg_list
                            .syntax()
                            .text()
                            .slice(range_search)
                            .to_string()
                            .matches(',')
                            .count();

                        // If we have a method call eat the first param since it's just self.
                        if has_self {
                            commas += 1;
                        }

                        current_parameter = Some(commas);
                    }
                }

                return Ok(Some((descriptor, current_parameter)));
            }
        }
    }

    Ok(None)
}

enum FnCallNode<'a> {
    CallExpr(&'a ast::CallExpr),
    MethodCallExpr(&'a ast::MethodCallExpr),
}

impl<'a> FnCallNode<'a> {
    pub fn with_node(syntax: &'a SyntaxNode, offset: TextUnit) -> Option<FnCallNode<'a>> {
        if let Some(expr) = find_node_at_offset::<ast::CallExpr>(syntax, offset) {
            return Some(FnCallNode::CallExpr(expr));
        }
        if let Some(expr) = find_node_at_offset::<ast::MethodCallExpr>(syntax, offset) {
            return Some(FnCallNode::MethodCallExpr(expr));
        }
        None
    }

    pub fn name_ref(&self) -> Option<&'a ast::NameRef> {
        match *self {
            FnCallNode::CallExpr(call_expr) => Some(match call_expr.expr()?.kind() {
                ast::ExprKind::PathExpr(path_expr) => path_expr.path()?.segment()?.name_ref()?,
                _ => return None,
            }),

            FnCallNode::MethodCallExpr(call_expr) => call_expr
                .syntax()
                .children()
                .filter_map(ast::NameRef::cast)
                .nth(0),
        }
    }

    pub fn arg_list(&self) -> Option<&'a ast::ArgList> {
        match *self {
            FnCallNode::CallExpr(expr) => expr.arg_list(),
            FnCallNode::MethodCallExpr(expr) => expr.arg_list(),
        }
    }
}
