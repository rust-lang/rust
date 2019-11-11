//! Builtin macro
use crate::db::AstDatabase;
use crate::{
    ast::{self, AstNode},
    name, AstId, CrateId, HirFileId, MacroCallId, MacroDefId, MacroDefKind, MacroFileKind,
    TextUnit,
};

use crate::quote;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinExpander {
    Line,
}

impl BuiltinExpander {
    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        match self {
            BuiltinExpander::Line => line_expand(db, id, tt),
        }
    }
}

pub fn find_builtin_macro(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::MacroCall>,
) -> Option<MacroDefId> {
    // FIXME: Better registering method
    if ident == &name::LINE_MACRO {
        Some(MacroDefId { krate, ast_id, kind: MacroDefKind::BuiltIn(BuiltinExpander::Line) })
    } else {
        None
    }
}

fn to_line_number(db: &dyn AstDatabase, file: HirFileId, pos: TextUnit) -> usize {
    // FIXME: Use expansion info
    let file_id = file.original_file(db);
    let text = db.file_text(file_id);
    let mut line_num = 1;

    // Count line end
    for (i, c) in text.chars().enumerate() {
        if i == pos.to_usize() {
            break;
        }
        if c == '\n' {
            line_num += 1;
        }
    }

    line_num
}

fn line_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = loc.ast_id.to_node(db);

    let arg = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
    let arg_start = arg.syntax().text_range().start();

    let file = id.as_file(MacroFileKind::Expr);
    let line_num = to_line_number(db, file, arg_start);

    let expanded = quote! {
        #line_num
    };

    Ok(expanded)
}
