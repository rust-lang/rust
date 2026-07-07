//! Defines database & queries for macro expansion.

use base_db::{Crate, SourceDatabase};
use syntax::ast;

use crate::{
    AstId, BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerExpander,
    MacroDefId, MacroDefKind, declarative::DeclarativeMacroExpander,
    proc_macro::CustomProcMacroExpander,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenExpander<'db> {
    /// Old-style `macro_rules` or the new macros 2.0
    DeclarativeMacro(&'db DeclarativeMacroExpander),
    /// Stuff like `line!` and `file!`.
    BuiltIn(BuiltinFnLikeExpander),
    /// Built-in eagerly expanded fn-like macros (`include!`, `concat!`, etc.)
    BuiltInEager(EagerExpander),
    /// `global_allocator` and such.
    BuiltInAttr(BuiltinAttrExpander),
    /// `derive(Copy)` and such.
    BuiltInDerive(BuiltinDeriveExpander),
    UnimplementedBuiltIn,
    /// The thing we love the most here in rust-analyzer -- procedural macros.
    ProcMacro(CustomProcMacroExpander),
}

#[query_group::query_group]
pub trait ExpandDatabase: SourceDatabase {
    /// Fetches the expander for this macro.
    #[salsa::transparent]
    #[salsa::invoke(TokenExpander::macro_expander)]
    fn macro_expander(&self, id: MacroDefId) -> TokenExpander<'_>;

    /// Fetches (and compiles) the expander of this decl macro.
    #[salsa::invoke(DeclarativeMacroExpander::expander)]
    #[salsa::transparent]
    fn decl_macro_expander(
        &self,
        def_crate: Crate,
        id: AstId<ast::Macro>,
    ) -> &DeclarativeMacroExpander;
}

impl<'db> TokenExpander<'db> {
    fn macro_expander(db: &'db dyn ExpandDatabase, id: MacroDefId) -> TokenExpander<'db> {
        match id.kind {
            MacroDefKind::Declarative(ast_id, _) => {
                TokenExpander::DeclarativeMacro(db.decl_macro_expander(id.krate, ast_id))
            }
            MacroDefKind::BuiltIn(_, expander) => TokenExpander::BuiltIn(expander),
            MacroDefKind::BuiltInAttr(_, expander) => TokenExpander::BuiltInAttr(expander),
            MacroDefKind::BuiltInDerive(_, expander) => TokenExpander::BuiltInDerive(expander),
            MacroDefKind::BuiltInEager(_, expander) => TokenExpander::BuiltInEager(expander),
            MacroDefKind::ProcMacro(_, expander, _) => TokenExpander::ProcMacro(expander),
            MacroDefKind::UnimplementedBuiltIn(_) => TokenExpander::UnimplementedBuiltIn,
        }
    }
}
