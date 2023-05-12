//! Builtin derives.

use ::tt::Ident;
use base_db::{CrateOrigin, LangCrateOrigin};
use itertools::izip;
use mbe::TokenMap;
use std::collections::HashSet;
use stdx::never;
use tracing::debug;

use crate::tt::{self, TokenId};
use syntax::{
    ast::{
        self, AstNode, FieldList, HasAttrs, HasGenericParams, HasModuleItem, HasName,
        HasTypeBounds, PathType,
    },
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

enum VariantShape {
    Struct(Vec<tt::Ident>),
    Tuple(usize),
    Unit,
}

fn tuple_field_iterator(n: usize) -> impl Iterator<Item = tt::Ident> {
    (0..n).map(|x| Ident::new(format!("f{x}"), tt::TokenId::unspecified()))
}

impl VariantShape {
    fn as_pattern(&self, path: tt::Subtree) -> tt::Subtree {
        self.as_pattern_map(path, |x| quote!(#x))
    }

    fn field_names(&self) -> Vec<tt::Ident> {
        match self {
            VariantShape::Struct(s) => s.clone(),
            VariantShape::Tuple(n) => tuple_field_iterator(*n).collect(),
            VariantShape::Unit => vec![],
        }
    }

    fn as_pattern_map(
        &self,
        path: tt::Subtree,
        field_map: impl Fn(&tt::Ident) -> tt::Subtree,
    ) -> tt::Subtree {
        match self {
            VariantShape::Struct(fields) => {
                let fields = fields.iter().map(|x| {
                    let mapped = field_map(x);
                    quote! { #x : #mapped , }
                });
                quote! {
                    #path { ##fields }
                }
            }
            &VariantShape::Tuple(n) => {
                let fields = tuple_field_iterator(n).map(|x| {
                    let mapped = field_map(&x);
                    quote! {
                        #mapped ,
                    }
                });
                quote! {
                    #path ( ##fields )
                }
            }
            VariantShape::Unit => path,
        }
    }

    fn from(value: Option<FieldList>, token_map: &TokenMap) -> Result<Self, ExpandError> {
        let r = match value {
            None => VariantShape::Unit,
            Some(FieldList::RecordFieldList(x)) => VariantShape::Struct(
                x.fields()
                    .map(|x| x.name())
                    .map(|x| name_to_token(token_map, x))
                    .collect::<Result<_, _>>()?,
            ),
            Some(FieldList::TupleFieldList(x)) => VariantShape::Tuple(x.fields().count()),
        };
        Ok(r)
    }
}

enum AdtShape {
    Struct(VariantShape),
    Enum { variants: Vec<(tt::Ident, VariantShape)>, default_variant: Option<usize> },
    Union,
}

impl AdtShape {
    fn as_pattern(&self, name: &tt::Ident) -> Vec<tt::Subtree> {
        self.as_pattern_map(name, |x| quote!(#x))
    }

    fn field_names(&self) -> Vec<Vec<tt::Ident>> {
        match self {
            AdtShape::Struct(s) => {
                vec![s.field_names()]
            }
            AdtShape::Enum { variants, .. } => {
                variants.iter().map(|(_, fields)| fields.field_names()).collect()
            }
            AdtShape::Union => {
                never!("using fields of union in derive is always wrong");
                vec![]
            }
        }
    }

    fn as_pattern_map(
        &self,
        name: &tt::Ident,
        field_map: impl Fn(&tt::Ident) -> tt::Subtree,
    ) -> Vec<tt::Subtree> {
        match self {
            AdtShape::Struct(s) => {
                vec![s.as_pattern_map(quote! { #name }, field_map)]
            }
            AdtShape::Enum { variants, .. } => variants
                .iter()
                .map(|(v, fields)| fields.as_pattern_map(quote! { #name :: #v }, &field_map))
                .collect(),
            AdtShape::Union => {
                never!("pattern matching on union is always wrong");
                vec![quote! { un }]
            }
        }
    }
}

struct BasicAdtInfo {
    name: tt::Ident,
    shape: AdtShape,
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
    let (name, params, shape) = match_ast! {
        match node {
            ast::Struct(it) => (it.name(), it.generic_param_list(), AdtShape::Struct(VariantShape::from(it.field_list(), &token_map)?)),
            ast::Enum(it) => {
                let default_variant = it.variant_list().into_iter().flat_map(|x| x.variants()).position(|x| x.attrs().any(|x| x.simple_name() == Some("default".into())));
                (
                    it.name(),
                    it.generic_param_list(),
                    AdtShape::Enum {
                        default_variant,
                        variants: it.variant_list()
                            .into_iter()
                            .flat_map(|x| x.variants())
                            .map(|x| Ok((name_to_token(&token_map,x.name())?, VariantShape::from(x.field_list(), &token_map)?))).collect::<Result<_, ExpandError>>()?
                    }
                )
            },
            ast::Union(it) => (it.name(), it.generic_param_list(), AdtShape::Union),
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
    let name_token = name_to_token(&token_map, name)?;
    Ok(BasicAdtInfo { name: name_token, shape, param_types, associated_types })
}

fn name_to_token(token_map: &TokenMap, name: Option<ast::Name>) -> Result<tt::Ident, ExpandError> {
    let name = name.ok_or_else(|| {
        debug!("parsed item has no name");
        ExpandError::Other("missing name".into())
    })?;
    let name_token_id =
        token_map.token_by_range(name.syntax().text_range()).unwrap_or_else(TokenId::unspecified);
    let name_token = tt::Ident { span: name_token_id, text: name.text().into() };
    Ok(name_token)
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
fn expand_simple_derive(
    tt: &tt::Subtree,
    trait_path: tt::Subtree,
    trait_body: impl FnOnce(&BasicAdtInfo) -> tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let info = match parse_adt(tt) {
        Ok(info) => info,
        Err(e) => return ExpandResult::new(tt::Subtree::empty(), e),
    };
    let trait_body = trait_body(&info);
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
        impl < ##params > #trait_path for #name < ##args > where ##where_block { #trait_body }
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
    expand_simple_derive(tt, quote! { #krate::marker::Copy }, |_| quote! {})
}

fn clone_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::clone::Clone }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            let star = tt::Punct {
                char: '*',
                spacing: ::tt::Spacing::Alone,
                span: tt::TokenId::unspecified(),
            };
            return quote! {
                fn clone(&self) -> Self {
                    #star self
                }
            };
        }
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct {
                char: '*',
                spacing: ::tt::Spacing::Alone,
                span: tt::TokenId::unspecified(),
            };
            return quote! {
                fn clone(&self) -> Self {
                    match #star self {}
                }
            };
        }
        let name = &adt.name;
        let patterns = adt.shape.as_pattern(name);
        let exprs = adt.shape.as_pattern_map(name, |x| quote! { #x .clone() });
        let arms = patterns.into_iter().zip(exprs.into_iter()).map(|(pat, expr)| {
            let fat_arrow = fat_arrow();
            quote! {
                #pat #fat_arrow #expr,
            }
        });

        quote! {
            fn clone(&self) -> Self {
                match self {
                    ##arms
                }
            }
        }
    })
}

/// This function exists since `quote! { => }` doesn't work.
fn fat_arrow() -> ::tt::Subtree<TokenId> {
    let eq =
        tt::Punct { char: '=', spacing: ::tt::Spacing::Joint, span: tt::TokenId::unspecified() };
    quote! { #eq> }
}

/// This function exists since `quote! { && }` doesn't work.
fn and_and() -> ::tt::Subtree<TokenId> {
    let and =
        tt::Punct { char: '&', spacing: ::tt::Spacing::Joint, span: tt::TokenId::unspecified() };
    quote! { #and& }
}

fn default_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = &find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::default::Default }, |adt| {
        let body = match &adt.shape {
            AdtShape::Struct(fields) => {
                let name = &adt.name;
                fields
                    .as_pattern_map(quote!(#name), |_| quote!(#krate::default::Default::default()))
            }
            AdtShape::Enum { default_variant, variants } => {
                if let Some(d) = default_variant {
                    let (name, fields) = &variants[*d];
                    let adt_name = &adt.name;
                    fields.as_pattern_map(
                        quote!(#adt_name :: #name),
                        |_| quote!(#krate::default::Default::default()),
                    )
                } else {
                    // FIXME: Return expand error here
                    quote!()
                }
            }
            AdtShape::Union => {
                // FIXME: Return expand error here
                quote!()
            }
        };
        quote! {
            fn default() -> Self {
                #body
            }
        }
    })
}

fn debug_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = &find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::fmt::Debug }, |adt| {
        let for_variant = |name: String, v: &VariantShape| match v {
            VariantShape::Struct(fields) => {
                let for_fields = fields.iter().map(|x| {
                    let x_string = x.to_string();
                    quote! {
                        .field(#x_string, & #x)
                    }
                });
                quote! {
                    f.debug_struct(#name) ##for_fields .finish()
                }
            }
            VariantShape::Tuple(n) => {
                let for_fields = tuple_field_iterator(*n).map(|x| {
                    quote! {
                        .field( & #x)
                    }
                });
                quote! {
                    f.debug_tuple(#name) ##for_fields .finish()
                }
            }
            VariantShape::Unit => quote! {
                f.write_str(#name)
            },
        };
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct {
                char: '*',
                spacing: ::tt::Spacing::Alone,
                span: tt::TokenId::unspecified(),
            };
            return quote! {
                fn fmt(&self, f: &mut #krate::fmt::Formatter) -> #krate::fmt::Result {
                    match #star self {}
                }
            };
        }
        let arms = match &adt.shape {
            AdtShape::Struct(fields) => {
                let fat_arrow = fat_arrow();
                let name = &adt.name;
                let pat = fields.as_pattern(quote!(#name));
                let expr = for_variant(name.to_string(), fields);
                vec![quote! { #pat #fat_arrow #expr }]
            }
            AdtShape::Enum { variants, .. } => variants
                .iter()
                .map(|(name, v)| {
                    let fat_arrow = fat_arrow();
                    let adt_name = &adt.name;
                    let pat = v.as_pattern(quote!(#adt_name :: #name));
                    let expr = for_variant(name.to_string(), v);
                    quote! {
                        #pat #fat_arrow #expr ,
                    }
                })
                .collect(),
            AdtShape::Union => {
                // FIXME: Return expand error here
                vec![]
            }
        };
        quote! {
            fn fmt(&self, f: &mut #krate::fmt::Formatter) -> #krate::fmt::Result {
                match self {
                    ##arms
                }
            }
        }
    })
}

fn hash_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = &find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::hash::Hash }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote! {};
        }
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct {
                char: '*',
                spacing: ::tt::Spacing::Alone,
                span: tt::TokenId::unspecified(),
            };
            return quote! {
                fn hash<H: #krate::hash::Hasher>(&self, state: &mut H) {
                    match #star self {}
                }
            };
        }
        let arms = adt.shape.as_pattern(&adt.name).into_iter().zip(adt.shape.field_names()).map(
            |(pat, names)| {
                let expr = {
                    let it = names.iter().map(|x| quote! { #x . hash(state); });
                    quote! { {
                        ##it
                    } }
                };
                let fat_arrow = fat_arrow();
                quote! {
                    #pat #fat_arrow #expr ,
                }
            },
        );
        quote! {
            fn hash<H: #krate::hash::Hasher>(&self, state: &mut H) {
                #krate::mem::discriminant(self).hash(state);
                match self {
                    ##arms
                }
            }
        }
    })
}

fn eq_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Eq }, |_| quote! {})
}

fn partial_eq_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialEq }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote! {};
        }
        let name = &adt.name;

        let (self_patterns, other_patterns) = self_and_other_patterns(adt, name);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names()).map(
            |(pat1, pat2, names)| {
                let fat_arrow = fat_arrow();
                let body = match &*names {
                    [] => {
                        quote!(true)
                    }
                    [first, rest @ ..] => {
                        let rest = rest.iter().map(|x| {
                            let t1 = Ident::new(format!("{}_self", x.text), x.span);
                            let t2 = Ident::new(format!("{}_other", x.text), x.span);
                            let and_and = and_and();
                            quote!(#and_and #t1 .eq( #t2 ))
                        });
                        let first = {
                            let t1 = Ident::new(format!("{}_self", first.text), first.span);
                            let t2 = Ident::new(format!("{}_other", first.text), first.span);
                            quote!(#t1 .eq( #t2 ))
                        };
                        quote!(#first ##rest)
                    }
                };
                quote! { ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );

        let fat_arrow = fat_arrow();
        quote! {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    ##arms
                    _unused #fat_arrow false
                }
            }
        }
    })
}

fn self_and_other_patterns(
    adt: &BasicAdtInfo,
    name: &tt::Ident,
) -> (Vec<tt::Subtree>, Vec<tt::Subtree>) {
    let self_patterns = adt.shape.as_pattern_map(name, |x| {
        let t = Ident::new(format!("{}_self", x.text), x.span);
        quote!(#t)
    });
    let other_patterns = adt.shape.as_pattern_map(name, |x| {
        let t = Ident::new(format!("{}_other", x.text), x.span);
        quote!(#t)
    });
    (self_patterns, other_patterns)
}

fn ord_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = &find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::Ord }, |adt| {
        fn compare(
            krate: &tt::TokenTree,
            left: tt::Subtree,
            right: tt::Subtree,
            rest: tt::Subtree,
        ) -> tt::Subtree {
            let fat_arrow1 = fat_arrow();
            let fat_arrow2 = fat_arrow();
            quote! {
                match #left.cmp(&#right) {
                    #krate::cmp::Ordering::Equal #fat_arrow1 {
                        #rest
                    }
                    c #fat_arrow2 return c,
                }
            }
        }
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote!();
        }
        let left = quote!(#krate::intrinsics::discriminant_value(self));
        let right = quote!(#krate::intrinsics::discriminant_value(other));

        let (self_patterns, other_patterns) = self_and_other_patterns(adt, &adt.name);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names()).map(
            |(pat1, pat2, fields)| {
                let mut body = quote!(#krate::cmp::Ordering::Equal);
                for f in fields.into_iter().rev() {
                    let t1 = Ident::new(format!("{}_self", f.text), f.span);
                    let t2 = Ident::new(format!("{}_other", f.text), f.span);
                    body = compare(krate, quote!(#t1), quote!(#t2), body);
                }
                let fat_arrow = fat_arrow();
                quote! { ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );
        let fat_arrow = fat_arrow();
        let body = compare(
            krate,
            left,
            right,
            quote! {
                match (self, other) {
                    ##arms
                    _unused #fat_arrow #krate::cmp::Ordering::Equal
                }
            },
        );
        quote! {
            fn cmp(&self, other: &Self) -> #krate::cmp::Ordering {
                #body
            }
        }
    })
}

fn partial_ord_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = &find_builtin_crate(db, id);
    expand_simple_derive(tt, quote! { #krate::cmp::PartialOrd }, |adt| {
        fn compare(
            krate: &tt::TokenTree,
            left: tt::Subtree,
            right: tt::Subtree,
            rest: tt::Subtree,
        ) -> tt::Subtree {
            let fat_arrow1 = fat_arrow();
            let fat_arrow2 = fat_arrow();
            quote! {
                match #left.partial_cmp(&#right) {
                    #krate::option::Option::Some(#krate::cmp::Ordering::Equal) #fat_arrow1 {
                        #rest
                    }
                    c #fat_arrow2 return c,
                }
            }
        }
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote!();
        }
        let left = quote!(#krate::intrinsics::discriminant_value(self));
        let right = quote!(#krate::intrinsics::discriminant_value(other));

        let (self_patterns, other_patterns) = self_and_other_patterns(adt, &adt.name);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names()).map(
            |(pat1, pat2, fields)| {
                let mut body = quote!(#krate::option::Option::Some(#krate::cmp::Ordering::Equal));
                for f in fields.into_iter().rev() {
                    let t1 = Ident::new(format!("{}_self", f.text), f.span);
                    let t2 = Ident::new(format!("{}_other", f.text), f.span);
                    body = compare(krate, quote!(#t1), quote!(#t2), body);
                }
                let fat_arrow = fat_arrow();
                quote! { ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );
        let fat_arrow = fat_arrow();
        let body = compare(
            krate,
            left,
            right,
            quote! {
                match (self, other) {
                    ##arms
                    _unused #fat_arrow #krate::option::Option::Some(#krate::cmp::Ordering::Equal)
                }
            },
        );
        quote! {
            fn partial_cmp(&self, other: &Self) -> #krate::option::Option::Option<#krate::cmp::Ordering> {
                #body
            }
        }
    })
}
