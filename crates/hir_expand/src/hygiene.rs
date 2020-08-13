//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use base_db::CrateId;
use either::Either;
use syntax::ast;

use crate::{
    db::AstDatabase,
    name::{AsName, Name},
    HirFileId, HirFileIdRepr, MacroCallId, MacroDefKind,
};

#[derive(Clone, Debug)]
pub struct Hygiene {
    // This is what `$crate` expands to
    def_crate: Option<CrateId>,

    // Indicate this is a local inner macro
    local_inner: bool,
}

impl Hygiene {
    pub fn new(db: &dyn AstDatabase, file_id: HirFileId) -> Hygiene {
        let (def_crate, local_inner) = match file_id.0 {
            HirFileIdRepr::FileId(_) => (None, false),
            HirFileIdRepr::MacroFile(macro_file) => match macro_file.macro_call_id {
                MacroCallId::LazyMacro(id) => {
                    let loc = db.lookup_intern_macro(id);
                    match loc.def.kind {
                        MacroDefKind::Declarative => (loc.def.krate, loc.def.local_inner),
                        MacroDefKind::BuiltIn(_) => (None, false),
                        MacroDefKind::BuiltInDerive(_) => (None, false),
                        MacroDefKind::BuiltInEager(_) => (None, false),
                        MacroDefKind::CustomDerive(_) => (None, false),
                    }
                }
                MacroCallId::EagerMacro(_id) => (None, false),
            },
        };
        Hygiene { def_crate, local_inner }
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene { def_crate: None, local_inner: false }
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(&self, name_ref: ast::NameRef) -> Either<Name, CrateId> {
        if let Some(def_crate) = self.def_crate {
            if name_ref.text() == "$crate" {
                return Either::Right(def_crate);
            }
        }
        Either::Left(name_ref.as_name())
    }

    pub fn local_inner_macros(&self) -> Option<CrateId> {
        if self.local_inner {
            self.def_crate
        } else {
            None
        }
    }
}
