//! Builtin derives.

use tracing::debug;

use mbe::ExpandResult;
use syntax::{
    ast::{self, AstNode, HasGenericParams, HasModuleItem, HasName},
    match_ast,
};

use crate::{db::AstDatabase, name, quote, AstId, CrateId, MacroCallId, MacroDefId, MacroDefKind};

macro_rules! register_builtin {
    ( $($trait:ident => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveExpander {
            $($trait),*
        }

        impl BuiltinDeriveExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> ExpandResult<tt::Subtree> {
                let expander = match *self {
                    $( BuiltinDeriveExpander::$trait => $expand, )*
                };
                expander(db, id, tt)
            }

            fn find_by_name(name: &name::Name) -> Option<Self> {
                match name {
                    $( id if id == &name::name![$trait] => Some(BuiltinDeriveExpander::$trait), )*
                     _ => None,
                }
            }
        }

    };
}

register_builtin! {
    Copy => copy_expand,
    Clone => clone_expand,
    Default => default_expand,
    Debug => debug_expand,
    Hash => hash_expand,
    Ord => ord_expand,
    PartialOrd => partial_ord_expand,
    Eq => eq_expand,
    PartialEq => partial_eq_expand
}

pub fn find_builtin_derive(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::Macro>,
) -> Option<MacroDefId> {
    let expander = BuiltinDeriveExpander::find_by_name(ident)?;
    Some(MacroDefId {
        krate,
        kind: MacroDefKind::BuiltInDerive(expander, ast_id),
        local_inner: false,
    })
}

struct BasicAdtInfo {
    name: tt::Ident,
    type_params: usize,
}

fn parse_adt(tt: &tt::Subtree) -> Result<BasicAdtInfo, mbe::ExpandError> {
    let (parsed, token_map) = mbe::token_tree_to_syntax_node(tt, mbe::TopEntryPoint::MacroItems)?; // FragmentKind::Items doesn't parse attrs?
    let macro_items = ast::MacroItems::cast(parsed.syntax_node()).ok_or_else(|| {
        debug!("derive node didn't parse");
        mbe::ExpandError::UnexpectedToken
    })?;
    let item = macro_items.items().next().ok_or_else(|| {
        debug!("no module item parsed");
        mbe::ExpandError::NoMatchingRule
    })?;
    let node = item.syntax();
    let (name, params) = match_ast! {
        match node {
            ast::Struct(it) => (it.name(), it.generic_param_list()),
            ast::Enum(it) => (it.name(), it.generic_param_list()),
            ast::Union(it) => (it.name(), it.generic_param_list()),
            _ => {
                debug!("unexpected node is {:?}", node);
                return Err(mbe::ExpandError::ConversionError)
            },
        }
    };
    let name = name.ok_or_else(|| {
        debug!("parsed item has no name");
        mbe::ExpandError::NoMatchingRule
    })?;
    let name_token_id = token_map.token_by_range(name.syntax().text_range()).ok_or_else(|| {
        debug!("name token not found");
        mbe::ExpandError::ConversionError
    })?;
    let name_token = tt::Ident { id: name_token_id, text: name.text().into() };
    let type_params = params.map_or(0, |type_param_list| type_param_list.type_params().count());
    Ok(BasicAdtInfo { name: name_token, type_params })
}

fn make_type_args(n: usize, bound: Vec<tt::TokenTree>) -> Vec<tt::TokenTree> {
    let mut result = Vec::<tt::TokenTree>::with_capacity(n * 2);
    result.push(
        tt::Leaf::Punct(tt::Punct {
            char: '<',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        })
        .into(),
    );
    for i in 0..n {
        if i > 0 {
            result.push(
                tt::Leaf::Punct(tt::Punct {
                    char: ',',
                    spacing: tt::Spacing::Alone,
                    id: tt::TokenId::unspecified(),
                })
                .into(),
            );
        }
        result.push(
            tt::Leaf::Ident(tt::Ident {
                id: tt::TokenId::unspecified(),
                text: format!("T{}", i).into(),
            })
            .into(),
        );
        result.extend(bound.iter().cloned());
    }
    result.push(
        tt::Leaf::Punct(tt::Punct {
            char: '>',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        })
        .into(),
    );
    result
}

fn expand_simple_derive(tt: &tt::Subtree, trait_path: tt::Subtree) -> ExpandResult<tt::Subtree> {
    let info = match parse_adt(tt) {
        Ok(info) => info,
        Err(e) => return ExpandResult::only_err(e),
    };
    let name = info.name;
    let trait_path_clone = trait_path.token_trees.clone();
    let bound = (quote! { : ##trait_path_clone }).token_trees;
    let type_params = make_type_args(info.type_params, bound);
    let type_args = make_type_args(info.type_params, Vec::new());
    let trait_path = trait_path.token_trees;
    let expanded = quote! {
        impl ##type_params ##trait_path for #name ##type_args {}
    };
    ExpandResult::ok(expanded)
}

fn find_builtin_crate(db: &dyn AstDatabase, id: MacroCallId) -> tt::TokenTree {
    // FIXME: make hygiene works for builtin derive macro
    // such that $crate can be used here.
    let cg = db.crate_graph();
    let krate = db.lookup_intern_macro_call(id).krate;

    // XXX
    //  All crates except core itself should have a dependency on core,
    //  We detect `core` by seeing whether it doesn't have such a dependency.
    let tt = if cg[krate].dependencies.iter().any(|dep| &*dep.name == "core") {
        quote! { core }
    } else {
        cov_mark::hit!(test_copy_expand_in_core);
        quote! { crate }
    };

    tt.token_trees[0].clone()
}

fn copy_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::marker::Copy })
}

fn clone_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::clone::Clone })
}

fn default_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::default::Default })
}

fn debug_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::fmt::Debug })
}

fn hash_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::hash::Hash })
}

fn eq_expand(db: &dyn AstDatabase, id: MacroCallId, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Eq })
}

fn partial_eq_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialEq })
}

fn ord_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Ord })
}

fn partial_ord_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialOrd })
}
