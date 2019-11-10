//! Builtin macro
use crate::{ast, name, AstId, BuiltinMacro, CrateId, MacroDefId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinExpander {
    Line
}

impl BuiltinExpander {
    pub fn expand(&self, _tt: &tt::Subtree) -> Result<tt::Subtree, mbe::ExpandError> {
        Err(mbe::ExpandError::UnexpectedToken)
    }
}

pub fn find_builtin_macro(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::MacroCall>,
) -> Option<MacroDefId> {
    // FIXME: Better registering method
    if ident == &name::LINE {
        Some(MacroDefId::BuiltinMacro(BuiltinMacro { expander: BuiltinExpander::Line, krate, ast_id }))
    } else {
        None
    }
}
