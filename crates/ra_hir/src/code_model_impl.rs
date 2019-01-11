mod krate; // `crate` is invalid ident :(
mod module;
pub(crate) mod function;

use ra_syntax::{AstNode, TreeArc};

use crate::{HirDatabase, DefId, HirFileId};

pub(crate) fn def_id_to_ast<N: AstNode>(
    db: &impl HirDatabase,
    def_id: DefId,
) -> (HirFileId, TreeArc<N>) {
    let (file_id, syntax) = def_id.source(db);
    let ast = N::cast(&syntax)
        .unwrap_or_else(|| panic!("def points to wrong source {:?} {:?}", def_id, syntax))
        .to_owned();
    (file_id, ast)
}
