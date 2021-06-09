//! Builtin derives.

use syntax::ast;

use crate::{db::AstDatabase, name, AstId, CrateId, MacroCallId, MacroDefId, MacroDefKind};

macro_rules! register_builtin {
    ( $(($name:ident, $variant:ident) => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinAttrExpander {
            $($variant),*
        }

        impl BuiltinAttrExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> Result<tt::Subtree, mbe::ExpandError> {
                let expander = match *self {
                    $( BuiltinAttrExpander::$variant => $expand, )*
                };
                expander(db, id, tt)
            }

            fn find_by_name(name: &name::Name) -> Option<Self> {
                match name {
                    $( id if id == &name::name![$name] => Some(BuiltinAttrExpander::$variant), )*
                     _ => None,
                }
            }
        }

    };
}

register_builtin! {
    (bench, Bench) => bench_expand,
    (cfg_accessible, CfgAccessible) => cfg_accessible_expand,
    (cfg_eval, CfgEval) => cfg_eval_expand,
    (derive, Derive) => derive_expand,
    (global_allocator, GlobalAllocator) => global_allocator_expand,
    (test, Test) => test_expand,
    (test_case, TestCase) => test_case_expand
}

pub fn find_builtin_attr(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::Macro>,
) -> Option<MacroDefId> {
    let expander = BuiltinAttrExpander::find_by_name(ident)?;
    Some(MacroDefId {
        krate,
        kind: MacroDefKind::BuiltInAttr(expander, ast_id),
        local_inner: false,
    })
}

fn bench_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn cfg_accessible_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn cfg_eval_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn derive_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn global_allocator_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn test_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}

fn test_case_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    Ok(tt.clone())
}
