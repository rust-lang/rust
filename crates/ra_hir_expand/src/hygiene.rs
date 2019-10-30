//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use ra_db::CrateId;
use ra_syntax::ast;

use crate::{
    db::AstDatabase,
    either::Either,
    name::{AsName, Name},
    HirFileId, HirFileIdRepr,
};

#[derive(Debug)]
pub struct Hygiene {
    // This is what `$crate` expands to
    def_crate: Option<CrateId>,
}

impl Hygiene {
    pub fn new(db: &impl AstDatabase, file_id: HirFileId) -> Hygiene {
        let def_crate = match file_id.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc = db.lookup_intern_macro(macro_file.macro_call_id);
                Some(loc.def.krate)
            }
        };
        Hygiene { def_crate }
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene { def_crate: None }
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(&self, name_ref: ast::NameRef) -> Either<Name, CrateId> {
        if let Some(def_crate) = self.def_crate {
            if name_ref.text() == "$crate" {
                return Either::B(def_crate);
            }
        }
        Either::A(name_ref.as_name())
    }
}
