//! Builtin derives.

use either::Either;
use intern::sym;
use itertools::{Itertools, izip};
use parser::SyntaxKind;
use rustc_hash::FxHashSet;
use span::{Edition, Span, SyntaxContext};
use stdx::never;
use syntax_bridge::DocCommentDesugarMode;
use tracing::debug;

use crate::{
    ExpandError, ExpandResult, MacroCallId,
    builtin::quote::{dollar_crate, quote},
    db::ExpandDatabase,
    hygiene::span_with_def_site_ctxt,
    name::{self, AsName, Name},
    span_map::ExpansionSpanMap,
    tt,
};
use syntax::{
    ast::{
        self, AstNode, FieldList, HasAttrs, HasGenericArgs, HasGenericParams, HasModuleItem,
        HasName, HasTypeBounds, edit_in_place::GenericParamsOwnerEdit, make,
    },
    ted,
};

macro_rules! register_builtin {
    ( $($trait:ident => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveExpander {
            $($trait),*
        }

        impl BuiltinDeriveExpander {
            pub fn expander(&self) -> fn(&dyn ExpandDatabase, Span, &tt::TopSubtree) -> ExpandResult<tt::TopSubtree>  {
                match *self {
                    $( BuiltinDeriveExpander::$trait => $expand, )*
                }
            }

            fn find_by_name(name: &name::Name) -> Option<Self> {
                match name {
                    $( id if id == &sym::$trait => Some(BuiltinDeriveExpander::$trait), )*
                     _ => None,
                }
            }
        }

    };
}

impl BuiltinDeriveExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::TopSubtree,
        span: Span,
    ) -> ExpandResult<tt::TopSubtree> {
        let span = span_with_def_site_ctxt(db, span, id.into(), Edition::CURRENT);
        self.expander()(db, span, tt)
    }
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
    PartialEq => partial_eq_expand,
    CoercePointee => coerce_pointee_expand
}

pub fn find_builtin_derive(ident: &name::Name) -> Option<BuiltinDeriveExpander> {
    BuiltinDeriveExpander::find_by_name(ident)
}

#[derive(Clone)]
enum VariantShape {
    Struct(Vec<tt::Ident>),
    Tuple(usize),
    Unit,
}

fn tuple_field_iterator(span: Span, n: usize) -> impl Iterator<Item = tt::Ident> {
    (0..n).map(move |it| tt::Ident::new(&format!("f{it}"), span))
}

impl VariantShape {
    fn as_pattern(&self, path: tt::TopSubtree, span: Span) -> tt::TopSubtree {
        self.as_pattern_map(path, span, |it| quote!(span => #it))
    }

    fn field_names(&self, span: Span) -> Vec<tt::Ident> {
        match self {
            VariantShape::Struct(s) => s.clone(),
            VariantShape::Tuple(n) => tuple_field_iterator(span, *n).collect(),
            VariantShape::Unit => vec![],
        }
    }

    fn as_pattern_map(
        &self,
        path: tt::TopSubtree,
        span: Span,
        field_map: impl Fn(&tt::Ident) -> tt::TopSubtree,
    ) -> tt::TopSubtree {
        match self {
            VariantShape::Struct(fields) => {
                let fields = fields.iter().map(|it| {
                    let mapped = field_map(it);
                    quote! {span => #it : #mapped , }
                });
                quote! {span =>
                    #path { # #fields }
                }
            }
            &VariantShape::Tuple(n) => {
                let fields = tuple_field_iterator(span, n).map(|it| {
                    let mapped = field_map(&it);
                    quote! {span =>
                        #mapped ,
                    }
                });
                quote! {span =>
                    #path ( # #fields )
                }
            }
            VariantShape::Unit => path,
        }
    }

    fn from(
        call_site: Span,
        tm: &ExpansionSpanMap,
        value: Option<FieldList>,
    ) -> Result<Self, ExpandError> {
        let r = match value {
            None => VariantShape::Unit,
            Some(FieldList::RecordFieldList(it)) => VariantShape::Struct(
                it.fields()
                    .map(|it| it.name())
                    .map(|it| name_to_token(call_site, tm, it))
                    .collect::<Result<_, _>>()?,
            ),
            Some(FieldList::TupleFieldList(it)) => VariantShape::Tuple(it.fields().count()),
        };
        Ok(r)
    }
}

#[derive(Clone)]
enum AdtShape {
    Struct(VariantShape),
    Enum { variants: Vec<(tt::Ident, VariantShape)>, default_variant: Option<usize> },
    Union,
}

impl AdtShape {
    fn as_pattern(&self, span: Span, name: &tt::Ident) -> Vec<tt::TopSubtree> {
        self.as_pattern_map(name, |it| quote!(span =>#it), span)
    }

    fn field_names(&self, span: Span) -> Vec<Vec<tt::Ident>> {
        match self {
            AdtShape::Struct(s) => {
                vec![s.field_names(span)]
            }
            AdtShape::Enum { variants, .. } => {
                variants.iter().map(|(_, fields)| fields.field_names(span)).collect()
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
        field_map: impl Fn(&tt::Ident) -> tt::TopSubtree,
        span: Span,
    ) -> Vec<tt::TopSubtree> {
        match self {
            AdtShape::Struct(s) => {
                vec![s.as_pattern_map(quote! {span => #name }, span, field_map)]
            }
            AdtShape::Enum { variants, .. } => variants
                .iter()
                .map(|(v, fields)| {
                    fields.as_pattern_map(quote! {span => #name :: #v }, span, &field_map)
                })
                .collect(),
            AdtShape::Union => {
                never!("pattern matching on union is always wrong");
                vec![quote! {span => un }]
            }
        }
    }
}

#[derive(Clone)]
struct BasicAdtInfo {
    name: tt::Ident,
    shape: AdtShape,
    /// first field is the name, and
    /// second field is `Some(ty)` if it's a const param of type `ty`, `None` if it's a type param.
    /// third fields is where bounds, if any
    param_types: Vec<AdtParam>,
    where_clause: Vec<tt::TopSubtree>,
    associated_types: Vec<tt::TopSubtree>,
}

#[derive(Clone)]
struct AdtParam {
    name: tt::TopSubtree,
    /// `None` if this is a type parameter.
    const_ty: Option<tt::TopSubtree>,
    bounds: Option<tt::TopSubtree>,
}

// FIXME: This whole thing needs a refactor. Each derive requires its special values, and the result is a mess.
fn parse_adt(
    db: &dyn ExpandDatabase,
    tt: &tt::TopSubtree,
    call_site: Span,
) -> Result<BasicAdtInfo, ExpandError> {
    let (adt, tm) = to_adt_syntax(db, tt, call_site)?;
    parse_adt_from_syntax(&adt, &tm, call_site)
}

fn parse_adt_from_syntax(
    adt: &ast::Adt,
    tm: &span::SpanMap<SyntaxContext>,
    call_site: Span,
) -> Result<BasicAdtInfo, ExpandError> {
    let (name, generic_param_list, where_clause, shape) = match &adt {
        ast::Adt::Struct(it) => (
            it.name(),
            it.generic_param_list(),
            it.where_clause(),
            AdtShape::Struct(VariantShape::from(call_site, tm, it.field_list())?),
        ),
        ast::Adt::Enum(it) => {
            let default_variant = it
                .variant_list()
                .into_iter()
                .flat_map(|it| it.variants())
                .position(|it| it.attrs().any(|it| it.simple_name() == Some("default".into())));
            (
                it.name(),
                it.generic_param_list(),
                it.where_clause(),
                AdtShape::Enum {
                    default_variant,
                    variants: it
                        .variant_list()
                        .into_iter()
                        .flat_map(|it| it.variants())
                        .map(|it| {
                            Ok((
                                name_to_token(call_site, tm, it.name())?,
                                VariantShape::from(call_site, tm, it.field_list())?,
                            ))
                        })
                        .collect::<Result<_, ExpandError>>()?,
                },
            )
        }
        ast::Adt::Union(it) => {
            (it.name(), it.generic_param_list(), it.where_clause(), AdtShape::Union)
        }
    };

    let mut param_type_set: FxHashSet<Name> = FxHashSet::default();
    let param_types = generic_param_list
        .into_iter()
        .flat_map(|param_list| param_list.type_or_const_params())
        .map(|param| {
            let name = {
                let this = param.name();
                match this {
                    Some(it) => {
                        param_type_set.insert(it.as_name());
                        syntax_bridge::syntax_node_to_token_tree(
                            it.syntax(),
                            tm,
                            call_site,
                            DocCommentDesugarMode::ProcMacro,
                        )
                    }
                    None => {
                        tt::TopSubtree::empty(::tt::DelimSpan { open: call_site, close: call_site })
                    }
                }
            };
            let bounds = match &param {
                ast::TypeOrConstParam::Type(it) => it.type_bound_list().map(|it| {
                    syntax_bridge::syntax_node_to_token_tree(
                        it.syntax(),
                        tm,
                        call_site,
                        DocCommentDesugarMode::ProcMacro,
                    )
                }),
                ast::TypeOrConstParam::Const(_) => None,
            };
            let const_ty = if let ast::TypeOrConstParam::Const(param) = param {
                let ty = param
                    .ty()
                    .map(|ty| {
                        syntax_bridge::syntax_node_to_token_tree(
                            ty.syntax(),
                            tm,
                            call_site,
                            DocCommentDesugarMode::ProcMacro,
                        )
                    })
                    .unwrap_or_else(|| {
                        tt::TopSubtree::empty(::tt::DelimSpan { open: call_site, close: call_site })
                    });
                Some(ty)
            } else {
                None
            };
            AdtParam { name, const_ty, bounds }
        })
        .collect();

    let where_clause = if let Some(w) = where_clause {
        w.predicates()
            .map(|it| {
                syntax_bridge::syntax_node_to_token_tree(
                    it.syntax(),
                    tm,
                    call_site,
                    DocCommentDesugarMode::ProcMacro,
                )
            })
            .collect()
    } else {
        vec![]
    };

    // For a generic parameter `T`, when shorthand associated type `T::Assoc` appears in field
    // types (of any variant for enums), we generate trait bound for it. It sounds reasonable to
    // also generate trait bound for qualified associated type `<T as Trait>::Assoc`, but rustc
    // does not do that for some unknown reason.
    //
    // See the analogous function in rustc [find_type_parameters()] and rust-lang/rust#50730.
    // [find_type_parameters()]: https://github.com/rust-lang/rust/blob/1.70.0/compiler/rustc_builtin_macros/src/deriving/generic/mod.rs#L378

    // It's cumbersome to deal with the distinct structures of ADTs, so let's just get untyped
    // `SyntaxNode` that contains fields and look for descendant `ast::PathType`s. Of note is that
    // we should not inspect `ast::PathType`s in parameter bounds and where clauses.
    let field_list = match adt {
        ast::Adt::Enum(it) => it.variant_list().map(|list| list.syntax().clone()),
        ast::Adt::Struct(it) => it.field_list().map(|list| list.syntax().clone()),
        ast::Adt::Union(it) => it.record_field_list().map(|list| list.syntax().clone()),
    };
    let associated_types = field_list
        .into_iter()
        .flat_map(|it| it.descendants())
        .filter_map(ast::PathType::cast)
        .filter_map(|p| {
            let name = p.path()?.qualifier()?.as_single_name_ref()?.as_name();
            param_type_set.contains(&name).then_some(p)
        })
        .map(|it| {
            syntax_bridge::syntax_node_to_token_tree(
                it.syntax(),
                tm,
                call_site,
                DocCommentDesugarMode::ProcMacro,
            )
        })
        .collect();
    let name_token = name_to_token(call_site, tm, name)?;
    Ok(BasicAdtInfo { name: name_token, shape, param_types, where_clause, associated_types })
}

fn to_adt_syntax(
    db: &dyn ExpandDatabase,
    tt: &tt::TopSubtree,
    call_site: Span,
) -> Result<(ast::Adt, span::SpanMap<SyntaxContext>), ExpandError> {
    let (parsed, tm) = crate::db::token_tree_to_syntax_node(
        db,
        tt,
        crate::ExpandTo::Items,
        parser::Edition::CURRENT_FIXME,
    );
    let macro_items = ast::MacroItems::cast(parsed.syntax_node())
        .ok_or_else(|| ExpandError::other(call_site, "invalid item definition"))?;
    let item =
        macro_items.items().next().ok_or_else(|| ExpandError::other(call_site, "no item found"))?;
    let adt = ast::Adt::cast(item.syntax().clone())
        .ok_or_else(|| ExpandError::other(call_site, "expected struct, enum or union"))?;
    Ok((adt, tm))
}

fn name_to_token(
    call_site: Span,
    token_map: &ExpansionSpanMap,
    name: Option<ast::Name>,
) -> Result<tt::Ident, ExpandError> {
    let name = name.ok_or_else(|| {
        debug!("parsed item has no name");
        ExpandError::other(call_site, "missing name")
    })?;
    let span = token_map.span_at(name.syntax().text_range().start());

    let name_token = tt::Ident::new(name.text().as_ref(), span);
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
/// where B1, ..., BN are the bounds given by `bounds_paths`. Z is a phantom type, and
/// therefore does not get bound by the derived trait.
fn expand_simple_derive(
    db: &dyn ExpandDatabase,
    invoc_span: Span,
    tt: &tt::TopSubtree,
    trait_path: tt::TopSubtree,
    make_trait_body: impl FnOnce(&BasicAdtInfo) -> tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let info = match parse_adt(db, tt, invoc_span) {
        Ok(info) => info,
        Err(e) => {
            return ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: invoc_span, close: invoc_span }),
                e,
            );
        }
    };
    ExpandResult::ok(expand_simple_derive_with_parsed(
        invoc_span,
        info,
        trait_path,
        make_trait_body,
        true,
        tt::TopSubtree::empty(tt::DelimSpan::from_single(invoc_span)),
    ))
}

fn expand_simple_derive_with_parsed(
    invoc_span: Span,
    info: BasicAdtInfo,
    trait_path: tt::TopSubtree,
    make_trait_body: impl FnOnce(&BasicAdtInfo) -> tt::TopSubtree,
    constrain_to_trait: bool,
    extra_impl_params: tt::TopSubtree,
) -> tt::TopSubtree {
    let trait_body = make_trait_body(&info);
    let mut where_block: Vec<_> =
        info.where_clause.into_iter().map(|w| quote! {invoc_span => #w , }).collect();
    let (params, args): (Vec<_>, Vec<_>) = info
        .param_types
        .into_iter()
        .map(|param| {
            let ident = param.name;
            if let Some(b) = param.bounds {
                let ident2 = ident.clone();
                where_block.push(quote! {invoc_span => #ident2 : #b , });
            }
            if let Some(ty) = param.const_ty {
                let ident2 = ident.clone();
                (quote! {invoc_span => const #ident : #ty , }, quote! {invoc_span => #ident2 , })
            } else {
                let bound = trait_path.clone();
                let ident2 = ident.clone();
                let param = if constrain_to_trait {
                    quote! {invoc_span => #ident : #bound , }
                } else {
                    quote! {invoc_span => #ident , }
                };
                (param, quote! {invoc_span => #ident2 , })
            }
        })
        .unzip();

    if constrain_to_trait {
        where_block.extend(info.associated_types.iter().map(|it| {
            let it = it.clone();
            let bound = trait_path.clone();
            quote! {invoc_span => #it : #bound , }
        }));
    }

    let name = info.name;
    quote! {invoc_span =>
        impl < # #params #extra_impl_params > #trait_path for #name < # #args > where # #where_block { #trait_body }
    }
}

fn copy_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::marker::Copy }, |_| quote! {span =>})
}

fn clone_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::clone::Clone }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            let star = tt::Punct { char: '*', spacing: ::tt::Spacing::Alone, span };
            return quote! {span =>
                fn clone(&self) -> Self {
                    #star self
                }
            };
        }
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct { char: '*', spacing: ::tt::Spacing::Alone, span };
            return quote! {span =>
                fn clone(&self) -> Self {
                    match #star self {}
                }
            };
        }
        let name = &adt.name;
        let patterns = adt.shape.as_pattern(span, name);
        let exprs = adt.shape.as_pattern_map(name, |it| quote! {span => #it .clone() }, span);
        let arms = patterns.into_iter().zip(exprs).map(|(pat, expr)| {
            let fat_arrow = fat_arrow(span);
            quote! {span =>
                #pat #fat_arrow #expr,
            }
        });

        quote! {span =>
            fn clone(&self) -> Self {
                match self {
                    # #arms
                }
            }
        }
    })
}

/// This function exists since `quote! {span => => }` doesn't work.
fn fat_arrow(span: Span) -> tt::TopSubtree {
    let eq = tt::Punct { char: '=', spacing: ::tt::Spacing::Joint, span };
    quote! {span => #eq> }
}

/// This function exists since `quote! {span => && }` doesn't work.
fn and_and(span: Span) -> tt::TopSubtree {
    let and = tt::Punct { char: '&', spacing: ::tt::Spacing::Joint, span };
    quote! {span => #and& }
}

fn default_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::default::Default }, |adt| {
        let body = match &adt.shape {
            AdtShape::Struct(fields) => {
                let name = &adt.name;
                fields.as_pattern_map(
                    quote!(span =>#name),
                    span,
                    |_| quote!(span =>#krate::default::Default::default()),
                )
            }
            AdtShape::Enum { default_variant, variants } => {
                if let Some(d) = default_variant {
                    let (name, fields) = &variants[*d];
                    let adt_name = &adt.name;
                    fields.as_pattern_map(
                        quote!(span =>#adt_name :: #name),
                        span,
                        |_| quote!(span =>#krate::default::Default::default()),
                    )
                } else {
                    // FIXME: Return expand error here
                    quote!(span =>)
                }
            }
            AdtShape::Union => {
                // FIXME: Return expand error here
                quote!(span =>)
            }
        };
        quote! {span =>
            fn default() -> Self {
                #body
            }
        }
    })
}

fn debug_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::fmt::Debug }, |adt| {
        let for_variant = |name: String, v: &VariantShape| match v {
            VariantShape::Struct(fields) => {
                let for_fields = fields.iter().map(|it| {
                    let x_string = it.to_string();
                    quote! {span =>
                        .field(#x_string, & #it)
                    }
                });
                quote! {span =>
                    f.debug_struct(#name) # #for_fields .finish()
                }
            }
            VariantShape::Tuple(n) => {
                let for_fields = tuple_field_iterator(span, *n).map(|it| {
                    quote! {span =>
                        .field( & #it)
                    }
                });
                quote! {span =>
                    f.debug_tuple(#name) # #for_fields .finish()
                }
            }
            VariantShape::Unit => quote! {span =>
                f.write_str(#name)
            },
        };
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct { char: '*', spacing: ::tt::Spacing::Alone, span };
            return quote! {span =>
                fn fmt(&self, f: &mut #krate::fmt::Formatter) -> #krate::fmt::Result {
                    match #star self {}
                }
            };
        }
        let arms = match &adt.shape {
            AdtShape::Struct(fields) => {
                let fat_arrow = fat_arrow(span);
                let name = &adt.name;
                let pat = fields.as_pattern(quote!(span =>#name), span);
                let expr = for_variant(name.to_string(), fields);
                vec![quote! {span => #pat #fat_arrow #expr }]
            }
            AdtShape::Enum { variants, .. } => variants
                .iter()
                .map(|(name, v)| {
                    let fat_arrow = fat_arrow(span);
                    let adt_name = &adt.name;
                    let pat = v.as_pattern(quote!(span =>#adt_name :: #name), span);
                    let expr = for_variant(name.to_string(), v);
                    quote! {span =>
                        #pat #fat_arrow #expr ,
                    }
                })
                .collect(),
            AdtShape::Union => {
                // FIXME: Return expand error here
                vec![]
            }
        };
        quote! {span =>
            fn fmt(&self, f: &mut #krate::fmt::Formatter) -> #krate::fmt::Result {
                match self {
                    # #arms
                }
            }
        }
    })
}

fn hash_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::hash::Hash }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote! {span =>};
        }
        if matches!(&adt.shape, AdtShape::Enum { variants, .. } if variants.is_empty()) {
            let star = tt::Punct { char: '*', spacing: ::tt::Spacing::Alone, span };
            return quote! {span =>
                fn hash<H: #krate::hash::Hasher>(&self, ra_expand_state: &mut H) {
                    match #star self {}
                }
            };
        }
        let arms =
            adt.shape.as_pattern(span, &adt.name).into_iter().zip(adt.shape.field_names(span)).map(
                |(pat, names)| {
                    let expr = {
                        let it =
                            names.iter().map(|it| quote! {span => #it . hash(ra_expand_state); });
                        quote! {span => {
                            # #it
                        } }
                    };
                    let fat_arrow = fat_arrow(span);
                    quote! {span =>
                        #pat #fat_arrow #expr ,
                    }
                },
            );
        let check_discriminant = if matches!(&adt.shape, AdtShape::Enum { .. }) {
            quote! {span => #krate::mem::discriminant(self).hash(ra_expand_state); }
        } else {
            quote! {span =>}
        };
        quote! {span =>
            fn hash<H: #krate::hash::Hasher>(&self, ra_expand_state: &mut H) {
                #check_discriminant
                match self {
                    # #arms
                }
            }
        }
    })
}

fn eq_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::cmp::Eq }, |_| quote! {span =>})
}

fn partial_eq_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::cmp::PartialEq }, |adt| {
        if matches!(adt.shape, AdtShape::Union) {
            // FIXME: Return expand error here
            return quote! {span =>};
        }
        let name = &adt.name;

        let (self_patterns, other_patterns) = self_and_other_patterns(adt, name, span);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names(span)).map(
            |(pat1, pat2, names)| {
                let fat_arrow = fat_arrow(span);
                let body = match &*names {
                    [] => {
                        quote!(span =>true)
                    }
                    [first, rest @ ..] => {
                        let rest = rest.iter().map(|it| {
                            let t1 = tt::Ident::new(&format!("{}_self", it.sym), it.span);
                            let t2 = tt::Ident::new(&format!("{}_other", it.sym), it.span);
                            let and_and = and_and(span);
                            quote!(span =>#and_and #t1 .eq( #t2 ))
                        });
                        let first = {
                            let t1 = tt::Ident::new(&format!("{}_self", first.sym), first.span);
                            let t2 = tt::Ident::new(&format!("{}_other", first.sym), first.span);
                            quote!(span =>#t1 .eq( #t2 ))
                        };
                        quote!(span =>#first # #rest)
                    }
                };
                quote! {span => ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );

        let fat_arrow = fat_arrow(span);
        quote! {span =>
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    # #arms
                    _unused #fat_arrow false
                }
            }
        }
    })
}

fn self_and_other_patterns(
    adt: &BasicAdtInfo,
    name: &tt::Ident,
    span: Span,
) -> (Vec<tt::TopSubtree>, Vec<tt::TopSubtree>) {
    let self_patterns = adt.shape.as_pattern_map(
        name,
        |it| {
            let t = tt::Ident::new(&format!("{}_self", it.sym), it.span);
            quote!(span =>#t)
        },
        span,
    );
    let other_patterns = adt.shape.as_pattern_map(
        name,
        |it| {
            let t = tt::Ident::new(&format!("{}_other", it.sym), it.span);
            quote!(span =>#t)
        },
        span,
    );
    (self_patterns, other_patterns)
}

fn ord_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::cmp::Ord }, |adt| {
        fn compare(
            krate: &tt::Ident,
            left: tt::TopSubtree,
            right: tt::TopSubtree,
            rest: tt::TopSubtree,
            span: Span,
        ) -> tt::TopSubtree {
            let fat_arrow1 = fat_arrow(span);
            let fat_arrow2 = fat_arrow(span);
            quote! {span =>
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
            return quote!(span =>);
        }
        let (self_patterns, other_patterns) = self_and_other_patterns(adt, &adt.name, span);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names(span)).map(
            |(pat1, pat2, fields)| {
                let mut body = quote!(span =>#krate::cmp::Ordering::Equal);
                for f in fields.into_iter().rev() {
                    let t1 = tt::Ident::new(&format!("{}_self", f.sym), f.span);
                    let t2 = tt::Ident::new(&format!("{}_other", f.sym), f.span);
                    body = compare(krate, quote!(span =>#t1), quote!(span =>#t2), body, span);
                }
                let fat_arrow = fat_arrow(span);
                quote! {span => ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );
        let fat_arrow = fat_arrow(span);
        let mut body = quote! {span =>
            match (self, other) {
                # #arms
                _unused #fat_arrow #krate::cmp::Ordering::Equal
            }
        };
        if matches!(&adt.shape, AdtShape::Enum { .. }) {
            let left = quote!(span =>#krate::intrinsics::discriminant_value(self));
            let right = quote!(span =>#krate::intrinsics::discriminant_value(other));
            body = compare(krate, left, right, body, span);
        }
        quote! {span =>
            fn cmp(&self, other: &Self) -> #krate::cmp::Ordering {
                #body
            }
        }
    })
}

fn partial_ord_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(db, span, tt, quote! {span => #krate::cmp::PartialOrd }, |adt| {
        fn compare(
            krate: &tt::Ident,
            left: tt::TopSubtree,
            right: tt::TopSubtree,
            rest: tt::TopSubtree,
            span: Span,
        ) -> tt::TopSubtree {
            let fat_arrow1 = fat_arrow(span);
            let fat_arrow2 = fat_arrow(span);
            quote! {span =>
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
            return quote!(span =>);
        }
        let left = quote!(span =>#krate::intrinsics::discriminant_value(self));
        let right = quote!(span =>#krate::intrinsics::discriminant_value(other));

        let (self_patterns, other_patterns) = self_and_other_patterns(adt, &adt.name, span);
        let arms = izip!(self_patterns, other_patterns, adt.shape.field_names(span)).map(
            |(pat1, pat2, fields)| {
                let mut body =
                    quote!(span =>#krate::option::Option::Some(#krate::cmp::Ordering::Equal));
                for f in fields.into_iter().rev() {
                    let t1 = tt::Ident::new(&format!("{}_self", f.sym), f.span);
                    let t2 = tt::Ident::new(&format!("{}_other", f.sym), f.span);
                    body = compare(krate, quote!(span =>#t1), quote!(span =>#t2), body, span);
                }
                let fat_arrow = fat_arrow(span);
                quote! {span => ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );
        let fat_arrow = fat_arrow(span);
        let body = compare(
            krate,
            left,
            right,
            quote! {span =>
                match (self, other) {
                    # #arms
                    _unused #fat_arrow #krate::option::Option::Some(#krate::cmp::Ordering::Equal)
                }
            },
            span,
        );
        quote! {span =>
            fn partial_cmp(&self, other: &Self) -> #krate::option::Option<#krate::cmp::Ordering> {
                #body
            }
        }
    })
}

fn coerce_pointee_expand(
    db: &dyn ExpandDatabase,
    span: Span,
    tt: &tt::TopSubtree,
) -> ExpandResult<tt::TopSubtree> {
    let (adt, _span_map) = match to_adt_syntax(db, tt, span) {
        Ok(it) => it,
        Err(err) => {
            return ExpandResult::new(tt::TopSubtree::empty(tt::DelimSpan::from_single(span)), err);
        }
    };
    let adt = adt.clone_for_update();
    let ast::Adt::Struct(strukt) = &adt else {
        return ExpandResult::new(
            tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
            ExpandError::other(span, "`CoercePointee` can only be derived on `struct`s"),
        );
    };
    let has_at_least_one_field = strukt
        .field_list()
        .map(|it| match it {
            ast::FieldList::RecordFieldList(it) => it.fields().next().is_some(),
            ast::FieldList::TupleFieldList(it) => it.fields().next().is_some(),
        })
        .unwrap_or(false);
    if !has_at_least_one_field {
        return ExpandResult::new(
            tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
            ExpandError::other(
                span,
                "`CoercePointee` can only be derived on `struct`s with at least one field",
            ),
        );
    }
    let is_repr_transparent = strukt.attrs().any(|attr| {
        attr.as_simple_call().is_some_and(|(name, tt)| {
            name == "repr"
                && tt.syntax().children_with_tokens().any(|it| {
                    it.into_token().is_some_and(|it| {
                        it.kind() == SyntaxKind::IDENT && it.text() == "transparent"
                    })
                })
        })
    });
    if !is_repr_transparent {
        return ExpandResult::new(
            tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
            ExpandError::other(
                span,
                "`CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`",
            ),
        );
    }
    let type_params = strukt
        .generic_param_list()
        .into_iter()
        .flat_map(|generics| {
            generics.generic_params().filter_map(|param| match param {
                ast::GenericParam::TypeParam(param) => Some(param),
                _ => None,
            })
        })
        .collect_vec();
    if type_params.is_empty() {
        return ExpandResult::new(
            tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
            ExpandError::other(
                span,
                "`CoercePointee` can only be derived on `struct`s that are generic over at least one type",
            ),
        );
    }
    let (pointee_param, pointee_param_idx) = if type_params.len() == 1 {
        // Regardless of the only type param being designed as `#[pointee]` or not, we can just use it as such.
        (type_params[0].clone(), 0)
    } else {
        let mut pointees = type_params.iter().cloned().enumerate().filter(|(_, param)| {
            param.attrs().any(|attr| {
                let is_pointee = attr.as_simple_atom().is_some_and(|name| name == "pointee");
                if is_pointee {
                    // Remove the `#[pointee]` attribute so it won't be present in the generated
                    // impls (where we cannot resolve it).
                    ted::remove(attr.syntax());
                }
                is_pointee
            })
        });
        match (pointees.next(), pointees.next()) {
            (Some((pointee_idx, pointee)), None) => (pointee, pointee_idx),
            (None, _) => {
                return ExpandResult::new(
                    tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
                    ExpandError::other(
                        span,
                        "exactly one generic type parameter must be marked \
                                as `#[pointee]` to derive `CoercePointee` traits",
                    ),
                );
            }
            (Some(_), Some(_)) => {
                return ExpandResult::new(
                    tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
                    ExpandError::other(
                        span,
                        "only one type parameter can be marked as `#[pointee]` \
                                when deriving `CoercePointee` traits",
                    ),
                );
            }
        }
    };
    let (Some(struct_name), Some(pointee_param_name)) = (strukt.name(), pointee_param.name())
    else {
        return ExpandResult::new(
            tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
            ExpandError::other(span, "invalid item"),
        );
    };

    {
        let mut pointee_has_maybe_sized_bound = false;
        if let Some(bounds) = pointee_param.type_bound_list() {
            pointee_has_maybe_sized_bound |= bounds.bounds().any(is_maybe_sized_bound);
        }
        if let Some(where_clause) = strukt.where_clause() {
            pointee_has_maybe_sized_bound |= where_clause.predicates().any(|pred| {
                let Some(ast::Type::PathType(ty)) = pred.ty() else { return false };
                let is_not_pointee = ty.path().is_none_or(|path| {
                    let is_pointee = path
                        .as_single_name_ref()
                        .is_some_and(|name| name.text() == pointee_param_name.text());
                    !is_pointee
                });
                if is_not_pointee {
                    return false;
                }
                pred.type_bound_list()
                    .is_some_and(|bounds| bounds.bounds().any(is_maybe_sized_bound))
            })
        }
        if !pointee_has_maybe_sized_bound {
            return ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
                ExpandError::other(
                    span,
                    format!(
                        "`derive(CoercePointee)` requires `{pointee_param_name}` to be marked `?Sized`"
                    ),
                ),
            );
        }
    }

    const ADDED_PARAM: &str = "__S";

    let where_clause = strukt.get_or_create_where_clause();

    {
        let mut new_predicates = Vec::new();

        // # Rewrite generic parameter bounds
        // For each bound `U: ..` in `struct<U: ..>`, make a new bound with `__S` in place of `#[pointee]`
        // Example:
        // ```
        // struct<
        //     U: Trait<T>,
        //     #[pointee] T: Trait<T> + ?Sized,
        //     V: Trait<T>> ...
        // ```
        // ... generates this `impl` generic parameters
        // ```
        // impl<
        //     U: Trait<T>,
        //     T: Trait<T> + ?Sized,
        //     V: Trait<T>
        // >
        // where
        //     U: Trait<__S>,
        //     __S: Trait<__S> + ?Sized,
        //     V: Trait<__S> ...
        // ```
        for param in &type_params {
            let Some(param_name) = param.name() else { continue };
            if let Some(bounds) = param.type_bound_list() {
                // If the target type is the pointee, duplicate the bound as whole.
                // Otherwise, duplicate only bounds that mention the pointee.
                let is_pointee = param_name.text() == pointee_param_name.text();
                let new_bounds = bounds
                    .bounds()
                    .map(|bound| bound.clone_subtree().clone_for_update())
                    .filter(|bound| {
                        bound.ty().is_some_and(|ty| {
                            substitute_type_in_bound(ty, &pointee_param_name.text(), ADDED_PARAM)
                                || is_pointee
                        })
                    });
                let new_bounds_target = if is_pointee {
                    make::name_ref(ADDED_PARAM)
                } else {
                    make::name_ref(&param_name.text())
                };
                new_predicates.push(
                    make::where_pred(
                        Either::Right(make::ty_path(make::path_from_segments(
                            [make::path_segment(new_bounds_target)],
                            false,
                        ))),
                        new_bounds,
                    )
                    .clone_for_update(),
                );
            }
        }

        // # Rewrite `where` clauses
        //
        // Move on to `where` clauses.
        // Example:
        // ```
        // struct MyPointer<#[pointee] T, ..>
        // where
        //   U: Trait<V> + Trait<T>,
        //   Companion<T>: Trait<T>,
        //   T: Trait<T> + ?Sized,
        // { .. }
        // ```
        // ... will have a impl prelude like so
        // ```
        // impl<..> ..
        // where
        //   U: Trait<V> + Trait<T>,
        //   U: Trait<__S>,
        //   Companion<T>: Trait<T>,
        //   Companion<__S>: Trait<__S>,
        //   T: Trait<T> + ?Sized,
        //   __S: Trait<__S> + ?Sized,
        // ```
        //
        // We should also write a few new `where` bounds from `#[pointee] T` to `__S`
        // as well as any bound that indirectly involves the `#[pointee] T` type.
        for predicate in where_clause.predicates() {
            let predicate = predicate.clone_subtree().clone_for_update();
            let Some(pred_target) = predicate.ty() else { continue };

            // If the target type references the pointee, duplicate the bound as whole.
            // Otherwise, duplicate only bounds that mention the pointee.
            if substitute_type_in_bound(
                pred_target.clone(),
                &pointee_param_name.text(),
                ADDED_PARAM,
            ) {
                if let Some(bounds) = predicate.type_bound_list() {
                    for bound in bounds.bounds() {
                        if let Some(ty) = bound.ty() {
                            substitute_type_in_bound(ty, &pointee_param_name.text(), ADDED_PARAM);
                        }
                    }
                }

                new_predicates.push(predicate);
            } else if let Some(bounds) = predicate.type_bound_list() {
                let new_bounds = bounds
                    .bounds()
                    .map(|bound| bound.clone_subtree().clone_for_update())
                    .filter(|bound| {
                        bound.ty().is_some_and(|ty| {
                            substitute_type_in_bound(ty, &pointee_param_name.text(), ADDED_PARAM)
                        })
                    });
                new_predicates.push(
                    make::where_pred(Either::Right(pred_target), new_bounds).clone_for_update(),
                );
            }
        }

        for new_predicate in new_predicates {
            where_clause.add_predicate(new_predicate);
        }
    }

    {
        // # Add `Unsize<__S>` bound to `#[pointee]` at the generic parameter location
        //
        // Find the `#[pointee]` parameter and add an `Unsize<__S>` bound to it.
        where_clause.add_predicate(
            make::where_pred(
                Either::Right(make::ty_path(make::path_from_segments(
                    [make::path_segment(make::name_ref(&pointee_param_name.text()))],
                    false,
                ))),
                [make::type_bound(make::ty_path(make::path_from_segments(
                    [
                        make::path_segment(make::name_ref("core")),
                        make::path_segment(make::name_ref("marker")),
                        make::generic_ty_path_segment(
                            make::name_ref("Unsize"),
                            [make::type_arg(make::ty_path(make::path_from_segments(
                                [make::path_segment(make::name_ref(ADDED_PARAM))],
                                false,
                            )))
                            .into()],
                        ),
                    ],
                    true,
                )))],
            )
            .clone_for_update(),
        );
    }

    let self_for_traits = {
        // Replace the `#[pointee]` with `__S`.
        let mut type_param_idx = 0;
        let self_params_for_traits = strukt
            .generic_param_list()
            .into_iter()
            .flat_map(|params| params.generic_params())
            .filter_map(|param| {
                Some(match param {
                    ast::GenericParam::ConstParam(param) => {
                        ast::GenericArg::ConstArg(make::expr_const_value(&param.name()?.text()))
                    }
                    ast::GenericParam::LifetimeParam(param) => {
                        make::lifetime_arg(param.lifetime()?).into()
                    }
                    ast::GenericParam::TypeParam(param) => {
                        let name = if pointee_param_idx == type_param_idx {
                            make::name_ref(ADDED_PARAM)
                        } else {
                            make::name_ref(&param.name()?.text())
                        };
                        type_param_idx += 1;
                        make::type_arg(make::ty_path(make::path_from_segments(
                            [make::path_segment(name)],
                            false,
                        )))
                        .into()
                    }
                })
            });

        make::path_from_segments(
            [make::generic_ty_path_segment(
                make::name_ref(&struct_name.text()),
                self_params_for_traits,
            )],
            false,
        )
        .clone_for_update()
    };

    let mut span_map = span::SpanMap::empty();
    // One span for them all.
    span_map.push(adt.syntax().text_range().end(), span);

    let self_for_traits = syntax_bridge::syntax_node_to_token_tree(
        self_for_traits.syntax(),
        &span_map,
        span,
        DocCommentDesugarMode::ProcMacro,
    );
    let info = match parse_adt_from_syntax(&adt, &span_map, span) {
        Ok(it) => it,
        Err(err) => {
            return ExpandResult::new(tt::TopSubtree::empty(tt::DelimSpan::from_single(span)), err);
        }
    };

    let self_for_traits2 = self_for_traits.clone();
    let krate = dollar_crate(span);
    let krate2 = krate.clone();
    let dispatch_from_dyn = expand_simple_derive_with_parsed(
        span,
        info.clone(),
        quote! {span => #krate2::ops::DispatchFromDyn<#self_for_traits2> },
        |_adt| quote! {span => },
        false,
        quote! {span => __S },
    );
    let coerce_unsized = expand_simple_derive_with_parsed(
        span,
        info,
        quote! {span => #krate::ops::CoerceUnsized<#self_for_traits> },
        |_adt| quote! {span => },
        false,
        quote! {span => __S },
    );
    return ExpandResult::ok(quote! {span => #dispatch_from_dyn #coerce_unsized });

    fn is_maybe_sized_bound(bound: ast::TypeBound) -> bool {
        if bound.question_mark_token().is_none() {
            return false;
        }
        let Some(ast::Type::PathType(ty)) = bound.ty() else {
            return false;
        };
        let Some(path) = ty.path() else {
            return false;
        };
        return segments_eq(&path, &["Sized"])
            || segments_eq(&path, &["core", "marker", "Sized"])
            || segments_eq(&path, &["std", "marker", "Sized"]);

        fn segments_eq(path: &ast::Path, expected: &[&str]) -> bool {
            path.segments().zip_longest(expected.iter().copied()).all(|value| {
                value.both().is_some_and(|(segment, expected)| {
                    segment.name_ref().is_some_and(|name| name.text() == expected)
                })
            })
        }
    }

    /// Returns true if any substitution was performed.
    fn substitute_type_in_bound(ty: ast::Type, param_name: &str, replacement: &str) -> bool {
        return match ty {
            ast::Type::ArrayType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::DynTraitType(ty) => go_bounds(ty.type_bound_list(), param_name, replacement),
            ast::Type::FnPtrType(ty) => any_long(
                ty.param_list()
                    .into_iter()
                    .flat_map(|params| params.params().filter_map(|param| param.ty()))
                    .chain(ty.ret_type().and_then(|it| it.ty())),
                |ty| substitute_type_in_bound(ty, param_name, replacement),
            ),
            ast::Type::ForType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::ImplTraitType(ty) => {
                go_bounds(ty.type_bound_list(), param_name, replacement)
            }
            ast::Type::ParenType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::PathType(ty) => ty.path().is_some_and(|path| {
                if path.as_single_name_ref().is_some_and(|name| name.text() == param_name) {
                    ted::replace(
                        path.syntax(),
                        make::path_from_segments(
                            [make::path_segment(make::name_ref(replacement))],
                            false,
                        )
                        .clone_for_update()
                        .syntax(),
                    );
                    return true;
                }

                any_long(
                    path.segments()
                        .filter_map(|segment| segment.generic_arg_list())
                        .flat_map(|it| it.generic_args())
                        .filter_map(|generic_arg| match generic_arg {
                            ast::GenericArg::TypeArg(ty) => ty.ty(),
                            _ => None,
                        }),
                    |ty| substitute_type_in_bound(ty, param_name, replacement),
                )
            }),
            ast::Type::PtrType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::RefType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::SliceType(ty) => {
                ty.ty().is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::TupleType(ty) => {
                any_long(ty.fields(), |ty| substitute_type_in_bound(ty, param_name, replacement))
            }
            ast::Type::InferType(_) | ast::Type::MacroType(_) | ast::Type::NeverType(_) => false,
        };

        fn go_bounds(
            bounds: Option<ast::TypeBoundList>,
            param_name: &str,
            replacement: &str,
        ) -> bool {
            bounds.is_some_and(|bounds| {
                any_long(bounds.bounds(), |bound| {
                    bound
                        .ty()
                        .is_some_and(|ty| substitute_type_in_bound(ty, param_name, replacement))
                })
            })
        }

        /// Like [`Iterator::any()`], but not short-circuiting.
        fn any_long<I: Iterator, F: FnMut(I::Item) -> bool>(iter: I, mut f: F) -> bool {
            let mut result = false;
            iter.for_each(|item| result |= f(item));
            result
        }
    }
}
