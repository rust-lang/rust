//! Builtin attributes.

use mbe::ExpandResult;
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
            ) -> ExpandResult<tt::Subtree> {
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

impl BuiltinAttrExpander {
    pub fn is_derive(self) -> bool {
        matches!(self, BuiltinAttrExpander::Derive)
    }
    pub fn is_test(self) -> bool {
        matches!(self, BuiltinAttrExpander::Test)
    }
    pub fn is_bench(self) -> bool {
        matches!(self, BuiltinAttrExpander::Bench)
    }
}

register_builtin! {
    (bench, Bench) => dummy_attr_expand,
    (cfg_accessible, CfgAccessible) => dummy_attr_expand,
    (cfg_eval, CfgEval) => dummy_attr_expand,
    (derive, Derive) => dummy_attr_expand,
    (global_allocator, GlobalAllocator) => dummy_attr_expand,
    (test, Test) => dummy_attr_expand,
    (test_case, TestCase) => dummy_attr_expand
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

fn dummy_attr_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(tt.clone())
}
