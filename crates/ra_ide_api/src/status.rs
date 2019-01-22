use ra_db::{
    LocationIntener, SourceFileQuery,
    salsa::{Database, debug::DebugQueryTable},
};

use crate::db::RootDatabase;

pub(crate) fn status(db: &RootDatabase) -> String {
    let n_parsed_files = db.query(SourceFileQuery).keys::<Vec<_>>().len();
    let n_defs = {
        let interner: &LocationIntener<hir::DefLoc, hir::DefId> = db.as_ref();
        interner.len()
    };
    format!("#n_parsed_files {}\n#n_defs {}\n", n_parsed_files, n_defs)
}
