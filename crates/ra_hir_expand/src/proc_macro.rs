//! Proc Macro Expander stub

use crate::{db::AstDatabase, LazyMacroId, MacroCallKind, MacroCallLoc};
use ra_db::CrateId;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    krate: CrateId,
}

impl ProcMacroExpander {
    pub fn new(krate: CrateId) -> ProcMacroExpander {
        ProcMacroExpander { krate }
    }

    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        id: LazyMacroId,
        _tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        let loc: MacroCallLoc = db.lookup_intern_macro(id);
        let name = match loc.kind {
            MacroCallKind::FnLike(_) => return Err(mbe::ExpandError::ConversionError),
            MacroCallKind::Attr(_, name) => name,
        };

        dbg!(name);

        unimplemented!()
    }
}
