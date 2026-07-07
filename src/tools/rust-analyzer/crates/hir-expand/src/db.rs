//! Defines database & queries for macro expansion.

use base_db::{Crate, SourceDatabase};
use syntax::ast;

use crate::{AstId, declarative::DeclarativeMacroExpander};

#[query_group::query_group]
pub trait ExpandDatabase: SourceDatabase {
    /// Fetches (and compiles) the expander of this decl macro.
    #[salsa::invoke(DeclarativeMacroExpander::expander)]
    #[salsa::transparent]
    fn decl_macro_expander(
        &self,
        def_crate: Crate,
        id: AstId<ast::Macro>,
    ) -> &DeclarativeMacroExpander;
}
