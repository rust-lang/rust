//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
// Should this be moved to `hir_expand`? Seems like it.

use hir_expand::{db::AstDatabase, HirFileId};
use ra_db::CrateId;
use ra_syntax::ast;

use crate::{
    either::Either,
    name::{AsName, Name},
};

#[derive(Debug)]
pub struct Hygiene {
    // This is what `$crate` expands to
    def_crate: Option<CrateId>,
}

impl Hygiene {
    pub fn new(db: &impl AstDatabase, file_id: HirFileId) -> Hygiene {
        Hygiene { def_crate: file_id.macro_crate(db) }
    }

    pub(crate) fn new_unhygienic() -> Hygiene {
        Hygiene { def_crate: None }
    }

    // FIXME: this should just return name
    pub(crate) fn name_ref_to_name(&self, name_ref: ast::NameRef) -> Either<Name, CrateId> {
        if let Some(def_crate) = self.def_crate {
            if name_ref.text() == "$crate" {
                return Either::B(def_crate);
            }
        }
        Either::A(name_ref.as_name())
    }
}
