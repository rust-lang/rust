//! Builtin derives.

use base_db::{CrateOrigin, LangCrateOrigin};
use std::collections::HashSet;
use tracing::debug;

use crate::tt::{self, TokenId};
use syntax::{
    ast::{self, AstNode, HasGenericParams, HasModuleItem, HasName, HasTypeBounds, PathType},
    match_ast,
};

use crate::{db::ExpandDatabase, name, quote, ExpandError, ExpandResult, MacroCallId};

macro_rules! register_builtin {
    ( $($trait:ident => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveExpander {
            $($trait),*
        }

        impl BuiltinDeriveExpander {
            pub fn expand(
                &self,
                db: &dyn ExpandDatabase,
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

pub fn find_builtin_derive(ident: &name::Name) -> Option<BuiltinDeriveExpander> {
    BuiltinDeriveExpander::find_by_name(ident)
}

struct BasicAdtInfo {
    name: tt::Ident,
    /// first field is the name, and
    /// second field is `Some(ty)` if it's a const param of type `ty`, `None` if it's a type param.
    /// third fields is where bounds, if any
    param_types: Vec<(tt::Subtree, Option<tt::Subtree>, Option<tt::Subtree>)>,
    associated_types: Vec<tt::Subtree>,
}

fn parse_adt(tt: &tt::Subtree) -> Result<BasicAdtInfo, ExpandError> {
    let (parsed, token_map) = mbe::token_tree_to_syntax_node(tt, mbe::TopEntryPoint::MacroItems);
    let macro_items = ast::MacroItems::cast(parsed.syntax_node()).ok_or_else(|| {
        debug!("derive node didn't parse");
        ExpandError::Other("invalid item definition".into())
    })?;
    let item = macro_items.items().next().ok_or_else(|| {
        debug!("no module item parsed");
        ExpandError::Other("no item found".into())
    })?;
    let node = item.syntax();
    let (name, params) = match_ast! {
        match node {
            ast::Struct(it) => (it.name(), it.generic_param_list()),
            ast::Enum(it) => (it.name(), it.generic_param_list()),
            ast::Union(it) => (it.name(), it.generic_param_list()),
            _ => {
                debug!("unexpected node is {:?}", node);
                return Err(ExpandError::Other("expected struct, enum or union".into()))
            },
        }
    };
    let mut param_type_set: HashSet<String> = HashSet::new();
    let param_types = params
        .into_iter()
        .flat_map(|param_list| param_list.type_or_const_params())
        .map(|param| {
            let name = {
                let this = param.name();
                match this {
                    Some(x) => {
                        param_type_set.insert(x.to_string());
                        mbe::syntax_node_to_token_tree(x.syntax()).0
                    }
                    None => tt::Subtree::empty(),
                }
            };
            let bounds = match &param {
                ast::TypeOrConstParam::Type(x) => {
                    x.type_bound_list().map(|x| mbe::syntax_node_to_token_tree(x.syntax()).0)
                }
                ast::TypeOrConstParam::Const(_) => None,
            };
            let ty = if let ast::TypeOrConstParam::Const(param) = param {
                let ty = param
                    .ty()
                    .map(|ty| mbe::syntax_node_to_token_tree(ty.syntax()).0)
                    .unwrap_or_else(tt::Subtree::empty);
                Some(ty)
            } else {
                None
            };
            (name, ty, bounds)
        })
        .collect();
    let is_associated_type = |p: &PathType| {
        if let Some(p) = p.path() {
            if let Some(parent) = p.qualifier() {
                if let Some(x) = parent.segment() {
                    if let Some(x) = x.path_type() {
                        if let Some(x) = x.path() {
                            if let Some(pname) = x.as_single_name_ref() {
                                if param_type_set.contains(&pname.to_string()) {
                                    // <T as Trait>::Assoc
                                    return true;
                                }
                            }
                        }
                    }
                }
                if let Some(pname) = parent.as_single_name_ref() {
                    if param_type_set.contains(&pname.to_string()) {
                        // T::Assoc
                        return true;
                    }
                }
            }
        }
        false
    };
    let associated_types = node
        .descendants()
        .filter_map(PathType::cast)
        .filter(is_associated_type)
        .map(|x| mbe::syntax_node_to_token_tree(x.syntax()).0)
        .collect::<Vec<_>>();
    let name = name.ok_or_else(|| {
        debug!("parsed item has no name");
        ExpandError::Other("missing name".into())
    })?;
    let name_token_id =
        token_map.token_by_range(name.syntax().text_range()).unwrap_or_else(TokenId::unspecified);
    let name_token = tt::Ident { span: name_token_id, text: name.text().into() };
    Ok(BasicAdtInfo { name: name_token, param_types, associated_types })
}

/// Given that we are deriving a trait `DerivedTrait` for a type like:
///
/// ```ignore (only-for-syntax-highlight)
/// struct Struct<'a, ..., 'z, A, B: DeclaredTrait, C, ..., Z> where C: WhereTrait {
///     a: A,
///     b: B::Item,
///     b1: <B as DeclaredTrait>::Item,
///     c1: <C as WhereTrait>::Item,
///     c2: Option<<C as WhereTrait>::Item>,
///     ...
/// }
/// ```
///
/// create an impl like:
///
/// ```ignore (only-for-syntax-highlight)
/// impl<'a, ..., 'z, A, B: DeclaredTrait, C, ... Z> where
///     C:                       WhereTrait,
///     A: DerivedTrait + B1 + ... + BN,
///     B: DerivedTrait + B1 + ... + BN,
///     C: DerivedTrait + B1 + ... + BN,
///     B::Item:                 DerivedTrait + B1 + ... + BN,
///     <C as WhereTrait>::Item: DerivedTrait + B1 + ... + BN,
///     ...
/// {
///     ...
/// }
/// ```
///
/// where B1, ..., BN are the bounds given by `bounds_paths`.'. Z is a phantom type, and
/// therefore does not get bound by the derived trait.
fn expand_simple_derive(tt: &tt::Subtree, trait_path: tt::Subtree) -> ExpandResult<tt::Subtree> {
    let info = match parse_adt(tt) {
        Ok(info) => info,
        Err(e) => return ExpandResult::with_err(tt::Subtree::empty(), e),
    };
    let mut where_block = vec![];
    let (params, args): (Vec<_>, Vec<_>) = info
        .param_types
        .into_iter()
        .map(|(ident, param_ty, bound)| {
            let ident_ = ident.clone();
            if let Some(b) = bound {
                let ident = ident.clone();
                where_block.push(quote! { #ident : #b , });
            }
            if let Some(ty) = param_ty {
                (quote! { const #ident : #ty , }, quote! { #ident_ , })
            } else {
                let bound = trait_path.clone();
                (quote! { #ident : #bound , }, quote! { #ident_ , })
            }
        })
        .unzip();

    where_block.extend(info.associated_types.iter().map(|x| {
        let x = x.clone();
        let bound = trait_path.clone();
        quote! { #x : #bound , }
    }));

    let name = info.name;
    let expanded = quote! {
        impl < ##params > #trait_path for #name < ##args > where ##where_block {}
    };
    ExpandResult::ok(expanded)
}

fn find_builtin_crate(db: &dyn ExpandDatabase, id: MacroCallId) -> tt::TokenTree {
    // FIXME: make hygiene works for builtin derive macro
    // such that $crate can be used here.
    let cg = db.crate_graph();
    let krate = db.lookup_intern_macro_call(id).krate;

    let tt = if matches!(cg[krate].origin, CrateOrigin::Lang(LangCrateOrigin::Core)) {
        cov_mark::hit!(test_copy_expand_in_core);
        quote! { crate }
    } else {
        quote! { core }
    };

    tt.token_trees[0].clone()
}

fn copy_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::marker::Copy })
}

fn clone_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::clone::Clone })
}

fn default_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::default::Default })
}

fn debug_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::fmt::Debug })
}

fn hash_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::hash::Hash })
}

fn eq_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Eq })
}

fn partial_eq_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialEq })
}

fn ord_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Ord })
}

fn partial_ord_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialOrd })
}
