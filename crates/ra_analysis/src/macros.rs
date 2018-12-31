/// Begining of macro expansion.
///
/// This code should be moved out of ra_analysis into hir (?) ideally.
use std::sync::Arc;

use ra_syntax::{ast, AstNode, TextUnit};
use hir::MacroDatabase;

use crate::{db::RootDatabase, FileId};

pub(crate) fn expand(
    db: &RootDatabase,
    _file_id: FileId,
    macro_call: ast::MacroCall,
) -> Option<(TextUnit, Arc<hir::MacroExpansion>)> {
    let path = macro_call.path()?;
    if path.qualifier().is_some() {
        return None;
    }
    let name_ref = path.segment()?.name_ref()?;
    if name_ref.text() != "ctry" {
        return None;
    }
    let arg = macro_call.token_tree()?.syntax();

    let def = hir::MacroDef::CTry;
    let input = hir::MacroInput {
        text: arg.text().to_string(),
    };
    let exp = db.expand_macro(def, input)?;
    Some((arg.range().start(), exp))
}
