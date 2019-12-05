//! Builtin derives.
use crate::db::AstDatabase;
use crate::{name, MacroCallId, MacroDefId, MacroDefKind};

use crate::quote;

macro_rules! register_builtin {
    ( $(($name:ident, $kind: ident) => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveExpander {
            $($kind),*
        }

        impl BuiltinDeriveExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> Result<tt::Subtree, mbe::ExpandError> {
                let expander = match *self {
                    $( BuiltinDeriveExpander::$kind => $expand, )*
                };
                expander(db, id, tt)
            }
        }

        pub fn find_builtin_derive(ident: &name::Name) -> Option<MacroDefId> {
            let kind = match ident {
                 $( id if id == &name::$name => BuiltinDeriveExpander::$kind, )*
                 _ => return None,
            };

            Some(MacroDefId { krate: None, ast_id: None, kind: MacroDefKind::BuiltInDerive(kind) })
        }
    };
}

register_builtin! {
    (COPY_TRAIT, Copy) => copy_expand,
    (CLONE_TRAIT, Clone) => clone_expand
}

fn copy_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let expanded = quote! {
        impl Copy for Foo {}
    };
    Ok(expanded)
}

fn clone_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let expanded = quote! {
        impl Clone for Foo {}
    };
    Ok(expanded)
}
