//! Builtin derives.

use itertools::izip;
use rustc_hash::FxHashSet;
use span::{MacroCallId, Span};
use stdx::never;
use tracing::debug;

use crate::{
    hygiene::span_with_def_site_ctxt,
    name::{AsName, Name},
    quote::dollar_crate,
    span_map::ExpansionSpanMap,
    tt,
};
use syntax::ast::{
    self, AstNode, FieldList, HasAttrs, HasGenericParams, HasModuleItem, HasName, HasTypeBounds,
};

use crate::{db::ExpandDatabase, name, quote, ExpandError, ExpandResult};

macro_rules! register_builtin {
    ( $($trait:ident => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveExpander {
            $($trait),*
        }

        impl BuiltinDeriveExpander {
            pub fn expander(&self) -> fn(Span, &tt::Subtree) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( BuiltinDeriveExpander::$trait => $expand, )*
                }
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

impl BuiltinDeriveExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> ExpandResult<tt::Subtree> {
        let span = db.lookup_intern_macro_call(id).call_site;
        let span = span_with_def_site_ctxt(db, span, id);
        self.expander()(span, tt)
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

fn tuple_field_iterator(span: Span, n: usize) -> impl Iterator<Item = tt::Ident> {
    (0..n).map(move |it| tt::Ident::new(format!("f{it}"), span))
}

impl VariantShape {
    fn as_pattern(&self, path: tt::Subtree, span: Span) -> tt::Subtree {
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
        path: tt::Subtree,
        span: Span,
        field_map: impl Fn(&tt::Ident) -> tt::Subtree,
    ) -> tt::Subtree {
        match self {
            VariantShape::Struct(fields) => {
                let fields = fields.iter().map(|it| {
                    let mapped = field_map(it);
                    quote! {span => #it : #mapped , }
                });
                quote! {span =>
                    #path { ##fields }
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
                    #path ( ##fields )
                }
            }
            VariantShape::Unit => path,
        }
    }

    fn from(tm: &ExpansionSpanMap, value: Option<FieldList>) -> Result<Self, ExpandError> {
        let r = match value {
            None => VariantShape::Unit,
            Some(FieldList::RecordFieldList(it)) => VariantShape::Struct(
                it.fields()
                    .map(|it| it.name())
                    .map(|it| name_to_token(tm, it))
                    .collect::<Result<_, _>>()?,
            ),
            Some(FieldList::TupleFieldList(it)) => VariantShape::Tuple(it.fields().count()),
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
    fn as_pattern(&self, span: Span, name: &tt::Ident) -> Vec<tt::Subtree> {
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
        field_map: impl Fn(&tt::Ident) -> tt::Subtree,
        span: Span,
    ) -> Vec<tt::Subtree> {
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

struct BasicAdtInfo {
    name: tt::Ident,
    shape: AdtShape,
    /// first field is the name, and
    /// second field is `Some(ty)` if it's a const param of type `ty`, `None` if it's a type param.
    /// third fields is where bounds, if any
    param_types: Vec<(tt::Subtree, Option<tt::Subtree>, Option<tt::Subtree>)>,
    where_clause: Vec<tt::Subtree>,
    associated_types: Vec<tt::Subtree>,
}

fn parse_adt(tt: &tt::Subtree, call_site: Span) -> Result<BasicAdtInfo, ExpandError> {
    let (parsed, tm) = &mbe::token_tree_to_syntax_node(tt, mbe::TopEntryPoint::MacroItems);
    let macro_items = ast::MacroItems::cast(parsed.syntax_node())
        .ok_or_else(|| ExpandError::other("invalid item definition"))?;
    let item = macro_items.items().next().ok_or_else(|| ExpandError::other("no item found"))?;
    let adt = &ast::Adt::cast(item.syntax().clone())
        .ok_or_else(|| ExpandError::other("expected struct, enum or union"))?;
    let (name, generic_param_list, where_clause, shape) = match adt {
        ast::Adt::Struct(it) => (
            it.name(),
            it.generic_param_list(),
            it.where_clause(),
            AdtShape::Struct(VariantShape::from(tm, it.field_list())?),
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
                                name_to_token(tm, it.name())?,
                                VariantShape::from(tm, it.field_list())?,
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
                        mbe::syntax_node_to_token_tree(it.syntax(), tm, call_site)
                    }
                    None => {
                        tt::Subtree::empty(::tt::DelimSpan { open: call_site, close: call_site })
                    }
                }
            };
            let bounds = match &param {
                ast::TypeOrConstParam::Type(it) => it
                    .type_bound_list()
                    .map(|it| mbe::syntax_node_to_token_tree(it.syntax(), tm, call_site)),
                ast::TypeOrConstParam::Const(_) => None,
            };
            let ty = if let ast::TypeOrConstParam::Const(param) = param {
                let ty = param
                    .ty()
                    .map(|ty| mbe::syntax_node_to_token_tree(ty.syntax(), tm, call_site))
                    .unwrap_or_else(|| {
                        tt::Subtree::empty(::tt::DelimSpan { open: call_site, close: call_site })
                    });
                Some(ty)
            } else {
                None
            };
            (name, ty, bounds)
        })
        .collect();

    let where_clause = if let Some(w) = where_clause {
        w.predicates()
            .map(|it| mbe::syntax_node_to_token_tree(it.syntax(), tm, call_site))
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
        .map(|it| mbe::syntax_node_to_token_tree(it.syntax(), tm, call_site))
        .collect();
    let name_token = name_to_token(tm, name)?;
    Ok(BasicAdtInfo { name: name_token, shape, param_types, where_clause, associated_types })
}

fn name_to_token(
    token_map: &ExpansionSpanMap,
    name: Option<ast::Name>,
) -> Result<tt::Ident, ExpandError> {
    let name = name.ok_or_else(|| {
        debug!("parsed item has no name");
        ExpandError::other("missing name")
    })?;
    let span = token_map.span_at(name.syntax().text_range().start());
    let name_token = tt::Ident { span, text: name.text().into() };
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
    invoc_span: Span,
    tt: &tt::Subtree,
    trait_path: tt::Subtree,
    make_trait_body: impl FnOnce(&BasicAdtInfo) -> tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let info = match parse_adt(tt, invoc_span) {
        Ok(info) => info,
        Err(e) => {
            return ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: invoc_span, close: invoc_span }),
                e,
            )
        }
    };
    let trait_body = make_trait_body(&info);
    let mut where_block: Vec<_> =
        info.where_clause.into_iter().map(|w| quote! {invoc_span => #w , }).collect();
    let (params, args): (Vec<_>, Vec<_>) = info
        .param_types
        .into_iter()
        .map(|(ident, param_ty, bound)| {
            let ident_ = ident.clone();
            if let Some(b) = bound {
                let ident = ident.clone();
                where_block.push(quote! {invoc_span => #ident : #b , });
            }
            if let Some(ty) = param_ty {
                (quote! {invoc_span => const #ident : #ty , }, quote! {invoc_span => #ident_ , })
            } else {
                let bound = trait_path.clone();
                (quote! {invoc_span => #ident : #bound , }, quote! {invoc_span => #ident_ , })
            }
        })
        .unzip();

    where_block.extend(info.associated_types.iter().map(|it| {
        let it = it.clone();
        let bound = trait_path.clone();
        quote! {invoc_span => #it : #bound , }
    }));

    let name = info.name;
    let expanded = quote! {invoc_span =>
        impl < ##params > #trait_path for #name < ##args > where ##where_block { #trait_body }
    };
    ExpandResult::ok(expanded)
}

fn copy_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::marker::Copy }, |_| quote! {span =>})
}

fn clone_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::clone::Clone }, |adt| {
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
                    ##arms
                }
            }
        }
    })
}

/// This function exists since `quote! {span => => }` doesn't work.
fn fat_arrow(span: Span) -> tt::Subtree {
    let eq = tt::Punct { char: '=', spacing: ::tt::Spacing::Joint, span };
    quote! {span => #eq> }
}

/// This function exists since `quote! {span => && }` doesn't work.
fn and_and(span: Span) -> tt::Subtree {
    let and = tt::Punct { char: '&', spacing: ::tt::Spacing::Joint, span };
    quote! {span => #and& }
}

fn default_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::default::Default }, |adt| {
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

fn debug_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::fmt::Debug }, |adt| {
        let for_variant = |name: String, v: &VariantShape| match v {
            VariantShape::Struct(fields) => {
                let for_fields = fields.iter().map(|it| {
                    let x_string = it.to_string();
                    quote! {span =>
                        .field(#x_string, & #it)
                    }
                });
                quote! {span =>
                    f.debug_struct(#name) ##for_fields .finish()
                }
            }
            VariantShape::Tuple(n) => {
                let for_fields = tuple_field_iterator(span, *n).map(|it| {
                    quote! {span =>
                        .field( & #it)
                    }
                });
                quote! {span =>
                    f.debug_tuple(#name) ##for_fields .finish()
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
                    ##arms
                }
            }
        }
    })
}

fn hash_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::hash::Hash }, |adt| {
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
                            ##it
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
                    ##arms
                }
            }
        }
    })
}

fn eq_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::cmp::Eq }, |_| quote! {span =>})
}

fn partial_eq_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::cmp::PartialEq }, |adt| {
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
                            let t1 = tt::Ident::new(format!("{}_self", it.text), it.span);
                            let t2 = tt::Ident::new(format!("{}_other", it.text), it.span);
                            let and_and = and_and(span);
                            quote!(span =>#and_and #t1 .eq( #t2 ))
                        });
                        let first = {
                            let t1 = tt::Ident::new(format!("{}_self", first.text), first.span);
                            let t2 = tt::Ident::new(format!("{}_other", first.text), first.span);
                            quote!(span =>#t1 .eq( #t2 ))
                        };
                        quote!(span =>#first ##rest)
                    }
                };
                quote! {span => ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );

        let fat_arrow = fat_arrow(span);
        quote! {span =>
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
    span: Span,
) -> (Vec<tt::Subtree>, Vec<tt::Subtree>) {
    let self_patterns = adt.shape.as_pattern_map(
        name,
        |it| {
            let t = tt::Ident::new(format!("{}_self", it.text), it.span);
            quote!(span =>#t)
        },
        span,
    );
    let other_patterns = adt.shape.as_pattern_map(
        name,
        |it| {
            let t = tt::Ident::new(format!("{}_other", it.text), it.span);
            quote!(span =>#t)
        },
        span,
    );
    (self_patterns, other_patterns)
}

fn ord_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::cmp::Ord }, |adt| {
        fn compare(
            krate: &tt::Ident,
            left: tt::Subtree,
            right: tt::Subtree,
            rest: tt::Subtree,
            span: Span,
        ) -> tt::Subtree {
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
                    let t1 = tt::Ident::new(format!("{}_self", f.text), f.span);
                    let t2 = tt::Ident::new(format!("{}_other", f.text), f.span);
                    body = compare(krate, quote!(span =>#t1), quote!(span =>#t2), body, span);
                }
                let fat_arrow = fat_arrow(span);
                quote! {span => ( #pat1 , #pat2 ) #fat_arrow #body , }
            },
        );
        let fat_arrow = fat_arrow(span);
        let mut body = quote! {span =>
            match (self, other) {
                ##arms
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

fn partial_ord_expand(span: Span, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
    let krate = &dollar_crate(span);
    expand_simple_derive(span, tt, quote! {span => #krate::cmp::PartialOrd }, |adt| {
        fn compare(
            krate: &tt::Ident,
            left: tt::Subtree,
            right: tt::Subtree,
            rest: tt::Subtree,
            span: Span,
        ) -> tt::Subtree {
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
                    let t1 = tt::Ident::new(format!("{}_self", f.text), f.span);
                    let t2 = tt::Ident::new(format!("{}_other", f.text), f.span);
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
                    ##arms
                    _unused #fat_arrow #krate::option::Option::Some(#krate::cmp::Ordering::Equal)
                }
            },
            span,
        );
        quote! {span =>
            fn partial_cmp(&self, other: &Self) -> #krate::option::Option::Option<#krate::cmp::Ordering> {
                #body
            }
        }
    })
}
