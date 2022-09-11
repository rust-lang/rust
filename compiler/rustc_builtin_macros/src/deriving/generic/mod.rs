//! Some code that abstracts away much of the boilerplate of writing
//! `derive` instances for traits. Among other things it manages getting
//! access to the fields of the 4 different sorts of structs and enum
//! variants, as well as creating the method and impl ast instances.
//!
//! Supported features (fairly exhaustive):
//!
//! - Methods taking any number of parameters of any type, and returning
//!   any type, other than vectors, bottom and closures.
//! - Generating `impl`s for types with type parameters and lifetimes
//!   (e.g., `Option<T>`), the parameters are automatically given the
//!   current trait as a bound. (This includes separate type parameters
//!   and lifetimes for methods.)
//! - Additional bounds on the type parameters (`TraitDef.additional_bounds`)
//!
//! The most important thing for implementors is the `Substructure` and
//! `SubstructureFields` objects. The latter groups 5 possibilities of the
//! arguments:
//!
//! - `Struct`, when `Self` is a struct (including tuple structs, e.g
//!   `struct T(i32, char)`).
//! - `EnumMatching`, when `Self` is an enum and all the arguments are the
//!   same variant of the enum (e.g., `Some(1)`, `Some(3)` and `Some(4)`)
//! - `EnumTag` when `Self` is an enum, for comparing the enum tags.
//! - `StaticEnum` and `StaticStruct` for static methods, where the type
//!   being derived upon is either an enum or struct respectively. (Any
//!   argument with type Self is just grouped among the non-self
//!   arguments.)
//!
//! In the first two cases, the values from the corresponding fields in
//! all the arguments are grouped together.
//!
//! The non-static cases have `Option<ident>` in several places associated
//! with field `expr`s. This represents the name of the field it is
//! associated with. It is only not `None` when the associated field has
//! an identifier in the source code. For example, the `x`s in the
//! following snippet
//!
//! ```rust
//! # #![allow(dead_code)]
//! struct A { x : i32 }
//!
//! struct B(i32);
//!
//! enum C {
//!     C0(i32),
//!     C1 { x: i32 }
//! }
//! ```
//!
//! The `i32`s in `B` and `C0` don't have an identifier, so the
//! `Option<ident>`s would be `None` for them.
//!
//! In the static cases, the structure is summarized, either into the just
//! spans of the fields or a list of spans and the field idents (for tuple
//! structs and record structs, respectively), or a list of these, for
//! enums (one for each variant). For empty struct and empty enum
//! variants, it is represented as a count of 0.
//!
//! # "`cs`" functions
//!
//! The `cs_...` functions ("combine substructure") are designed to
//! make life easier by providing some pre-made recipes for common
//! threads; mostly calling the function being derived on all the
//! arguments and then combining them back together in some way (or
//! letting the user chose that). They are not meant to be the only
//! way to handle the structures that this code creates.
//!
//! # Examples
//!
//! The following simplified `PartialEq` is used for in-code examples:
//!
//! ```rust
//! trait PartialEq {
//!     fn eq(&self, other: &Self) -> bool;
//! }
//! impl PartialEq for i32 {
//!     fn eq(&self, other: &i32) -> bool {
//!         *self == *other
//!     }
//! }
//! ```
//!
//! Some examples of the values of `SubstructureFields` follow, using the
//! above `PartialEq`, `A`, `B` and `C`.
//!
//! ## Structs
//!
//! When generating the `expr` for the `A` impl, the `SubstructureFields` is
//!
//! ```{.text}
//! Struct(vec![FieldInfo {
//!            span: <span of x>
//!            name: Some(<ident of x>),
//!            self_: <expr for &self.x>,
//!            other: vec![<expr for &other.x]
//!          }])
//! ```
//!
//! For the `B` impl, called with `B(a)` and `B(b)`,
//!
//! ```{.text}
//! Struct(vec![FieldInfo {
//!           span: <span of `i32`>,
//!           name: None,
//!           self_: <expr for &a>
//!           other: vec![<expr for &b>]
//!          }])
//! ```
//!
//! ## Enums
//!
//! When generating the `expr` for a call with `self == C0(a)` and `other
//! == C0(b)`, the SubstructureFields is
//!
//! ```{.text}
//! EnumMatching(0, <ast::Variant for C0>,
//!              vec![FieldInfo {
//!                 span: <span of i32>
//!                 name: None,
//!                 self_: <expr for &a>,
//!                 other: vec![<expr for &b>]
//!               }])
//! ```
//!
//! For `C1 {x}` and `C1 {x}`,
//!
//! ```{.text}
//! EnumMatching(1, <ast::Variant for C1>,
//!              vec![FieldInfo {
//!                 span: <span of x>
//!                 name: Some(<ident of x>),
//!                 self_: <expr for &self.x>,
//!                 other: vec![<expr for &other.x>]
//!                }])
//! ```
//!
//! For the tags,
//!
//! ```{.text}
//! EnumTag(
//!     &[<ident of self tag>, <ident of other tag>], <expr to combine with>)
//! ```
//! Note that this setup doesn't allow for the brute-force "match every variant
//! against every other variant" approach, which is bad because it produces a
//! quadratic amount of code (see #15375).
//!
//! ## Static
//!
//! A static method on the types above would result in,
//!
//! ```{.text}
//! StaticStruct(<ast::VariantData of A>, Named(vec![(<ident of x>, <span of x>)]))
//!
//! StaticStruct(<ast::VariantData of B>, Unnamed(vec![<span of x>]))
//!
//! StaticEnum(<ast::EnumDef of C>,
//!            vec![(<ident of C0>, <span of C0>, Unnamed(vec![<span of i32>])),
//!                 (<ident of C1>, <span of C1>, Named(vec![(<ident of x>, <span of x>)]))])
//! ```

pub use StaticFields::*;
pub use SubstructureFields::*;

use crate::deriving;
use rustc_ast::ptr::P;
use rustc_ast::{
    self as ast, BindingAnnotation, ByRef, EnumDef, Expr, Generics, Mutability, PatKind,
};
use rustc_ast::{GenericArg, GenericParamKind, VariantData};
use rustc_attr as attr;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use std::cell::RefCell;
use std::iter;
use std::vec;
use thin_vec::thin_vec;
use ty::{Bounds, Path, Ref, Self_, Ty};

pub mod ty;

pub struct TraitDef<'a> {
    /// The span for the current #[derive(Foo)] header.
    pub span: Span,

    /// Path of the trait, including any type parameters
    pub path: Path,

    /// Additional bounds required of any type parameters of the type,
    /// other than the current trait
    pub additional_bounds: Vec<Ty>,

    /// Any extra lifetimes and/or bounds, e.g., `D: serialize::Decoder`
    pub generics: Bounds,

    /// Can this trait be derived for unions?
    pub supports_unions: bool,

    pub methods: Vec<MethodDef<'a>>,

    pub associated_types: Vec<(Ident, Ty)>,
}

pub struct MethodDef<'a> {
    /// name of the method
    pub name: Symbol,
    /// List of generics, e.g., `R: rand::Rng`
    pub generics: Bounds,

    /// Is there is a `&self` argument? If not, it is a static function.
    pub explicit_self: bool,

    /// Arguments other than the self argument.
    pub nonself_args: Vec<(Ty, Symbol)>,

    /// Returns type
    pub ret_ty: Ty,

    pub attributes: ast::AttrVec,

    /// Can we combine fieldless variants for enums into a single match arm?
    /// If true, indicates that the trait operation uses the enum tag in some
    /// way.
    pub unify_fieldless_variants: bool,

    pub combine_substructure: RefCell<CombineSubstructureFunc<'a>>,
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'a> {
    /// ident of self
    pub type_ident: Ident,
    /// Verbatim access to any non-selflike arguments, i.e. arguments that
    /// don't have type `&Self`.
    pub nonselflike_args: &'a [P<Expr>],
    pub fields: &'a SubstructureFields<'a>,
}

/// Summary of the relevant parts of a struct/enum field.
pub struct FieldInfo {
    pub span: Span,
    /// None for tuple structs/normal enum variants, Some for normal
    /// structs/struct enum variants.
    pub name: Option<Ident>,
    /// The expression corresponding to this field of `self`
    /// (specifically, a reference to it).
    pub self_expr: P<Expr>,
    /// The expressions corresponding to references to this field in
    /// the other selflike arguments.
    pub other_selflike_exprs: Vec<P<Expr>>,
}

/// Fields for a static method
pub enum StaticFields {
    /// Tuple and unit structs/enum variants like this.
    Unnamed(Vec<Span>, bool /*is tuple*/),
    /// Normal structs/struct variants.
    Named(Vec<(Ident, Span)>),
}

/// A summary of the possible sets of fields.
pub enum SubstructureFields<'a> {
    /// A non-static method with `Self` is a struct.
    Struct(&'a ast::VariantData, Vec<FieldInfo>),

    /// Matching variants of the enum: variant index, variant count, ast::Variant,
    /// fields: the field name is only non-`None` in the case of a struct
    /// variant.
    EnumMatching(usize, usize, &'a ast::Variant, Vec<FieldInfo>),

    /// The tag of an enum. The first field is a `FieldInfo` for the tags, as
    /// if they were fields. The second field is the expression to combine the
    /// tag expression with; it will be `None` if no match is necessary.
    EnumTag(FieldInfo, Option<P<Expr>>),

    /// A static method where `Self` is a struct.
    StaticStruct(&'a ast::VariantData, StaticFields),

    /// A static method where `Self` is an enum.
    StaticEnum(&'a ast::EnumDef, Vec<(Ident, Span, StaticFields)>),
}

/// Combine the values of all the fields together. The last argument is
/// all the fields of all the structures.
pub type CombineSubstructureFunc<'a> =
    Box<dyn FnMut(&mut ExtCtxt<'_>, Span, &Substructure<'_>) -> BlockOrExpr + 'a>;

pub fn combine_substructure(
    f: CombineSubstructureFunc<'_>,
) -> RefCell<CombineSubstructureFunc<'_>> {
    RefCell::new(f)
}

struct TypeParameter {
    bound_generic_params: Vec<ast::GenericParam>,
    ty: P<ast::Ty>,
}

// The code snippets built up for derived code are sometimes used as blocks
// (e.g. in a function body) and sometimes used as expressions (e.g. in a match
// arm). This structure avoids committing to either form until necessary,
// avoiding the insertion of any unnecessary blocks.
//
// The statements come before the expression.
pub struct BlockOrExpr(Vec<ast::Stmt>, Option<P<Expr>>);

impl BlockOrExpr {
    pub fn new_stmts(stmts: Vec<ast::Stmt>) -> BlockOrExpr {
        BlockOrExpr(stmts, None)
    }

    pub fn new_expr(expr: P<Expr>) -> BlockOrExpr {
        BlockOrExpr(vec![], Some(expr))
    }

    pub fn new_mixed(stmts: Vec<ast::Stmt>, expr: Option<P<Expr>>) -> BlockOrExpr {
        BlockOrExpr(stmts, expr)
    }

    // Converts it into a block.
    fn into_block(mut self, cx: &ExtCtxt<'_>, span: Span) -> P<ast::Block> {
        if let Some(expr) = self.1 {
            self.0.push(cx.stmt_expr(expr));
        }
        cx.block(span, self.0)
    }

    // Converts it into an expression.
    fn into_expr(self, cx: &ExtCtxt<'_>, span: Span) -> P<Expr> {
        if self.0.is_empty() {
            match self.1 {
                None => cx.expr_block(cx.block(span, vec![])),
                Some(expr) => expr,
            }
        } else if self.0.len() == 1
            && let ast::StmtKind::Expr(expr) = &self.0[0].kind
            && self.1.is_none()
        {
            // There's only a single statement expression. Pull it out.
            expr.clone()
        } else {
            // Multiple statements and/or expressions.
            cx.expr_block(self.into_block(cx, span))
        }
    }
}

/// This method helps to extract all the type parameters referenced from a
/// type. For a type parameter `<T>`, it looks for either a `TyPath` that
/// is not global and starts with `T`, or a `TyQPath`.
/// Also include bound generic params from the input type.
fn find_type_parameters(
    ty: &ast::Ty,
    ty_param_names: &[Symbol],
    cx: &ExtCtxt<'_>,
) -> Vec<TypeParameter> {
    use rustc_ast::visit;

    struct Visitor<'a, 'b> {
        cx: &'a ExtCtxt<'b>,
        ty_param_names: &'a [Symbol],
        bound_generic_params_stack: Vec<ast::GenericParam>,
        type_params: Vec<TypeParameter>,
    }

    impl<'a, 'b> visit::Visitor<'a> for Visitor<'a, 'b> {
        fn visit_ty(&mut self, ty: &'a ast::Ty) {
            if let ast::TyKind::Path(_, ref path) = ty.kind {
                if let Some(segment) = path.segments.first() {
                    if self.ty_param_names.contains(&segment.ident.name) {
                        self.type_params.push(TypeParameter {
                            bound_generic_params: self.bound_generic_params_stack.clone(),
                            ty: P(ty.clone()),
                        });
                    }
                }
            }

            visit::walk_ty(self, ty)
        }

        // Place bound generic params on a stack, to extract them when a type is encountered.
        fn visit_poly_trait_ref(&mut self, trait_ref: &'a ast::PolyTraitRef) {
            let stack_len = self.bound_generic_params_stack.len();
            self.bound_generic_params_stack.extend(trait_ref.bound_generic_params.iter().cloned());

            visit::walk_poly_trait_ref(self, trait_ref);

            self.bound_generic_params_stack.truncate(stack_len);
        }

        fn visit_mac_call(&mut self, mac: &ast::MacCall) {
            self.cx.span_err(mac.span(), "`derive` cannot be used on items with type macros");
        }
    }

    let mut visitor = Visitor {
        cx,
        ty_param_names,
        bound_generic_params_stack: Vec::new(),
        type_params: Vec::new(),
    };
    visit::Visitor::visit_ty(&mut visitor, ty);

    visitor.type_params
}

impl<'a> TraitDef<'a> {
    pub fn expand(
        self,
        cx: &mut ExtCtxt<'_>,
        mitem: &ast::MetaItem,
        item: &'a Annotatable,
        push: &mut dyn FnMut(Annotatable),
    ) {
        self.expand_ext(cx, mitem, item, push, false);
    }

    pub fn expand_ext(
        self,
        cx: &mut ExtCtxt<'_>,
        mitem: &ast::MetaItem,
        item: &'a Annotatable,
        push: &mut dyn FnMut(Annotatable),
        from_scratch: bool,
    ) {
        match *item {
            Annotatable::Item(ref item) => {
                let is_packed = item.attrs.iter().any(|attr| {
                    for r in attr::find_repr_attrs(&cx.sess, attr) {
                        if let attr::ReprPacked(_) = r {
                            return true;
                        }
                    }
                    false
                });
                let has_no_type_params = match item.kind {
                    ast::ItemKind::Struct(_, ref generics)
                    | ast::ItemKind::Enum(_, ref generics)
                    | ast::ItemKind::Union(_, ref generics) => !generics
                        .params
                        .iter()
                        .any(|param| matches!(param.kind, ast::GenericParamKind::Type { .. })),
                    _ => unreachable!(),
                };
                let container_id = cx.current_expansion.id.expn_data().parent.expect_local();
                let always_copy = has_no_type_params && cx.resolver.has_derive_copy(container_id);

                let newitem = match item.kind {
                    ast::ItemKind::Struct(ref struct_def, ref generics) => self.expand_struct_def(
                        cx,
                        &struct_def,
                        item.ident,
                        generics,
                        from_scratch,
                        is_packed,
                        always_copy,
                    ),
                    ast::ItemKind::Enum(ref enum_def, ref generics) => {
                        // We ignore `is_packed`/`always_copy` here, because
                        // `repr(packed)` enums cause an error later on.
                        //
                        // This can only cause further compilation errors
                        // downstream in blatantly illegal code, so it
                        // is fine.
                        self.expand_enum_def(cx, enum_def, item.ident, generics, from_scratch)
                    }
                    ast::ItemKind::Union(ref struct_def, ref generics) => {
                        if self.supports_unions {
                            self.expand_struct_def(
                                cx,
                                &struct_def,
                                item.ident,
                                generics,
                                from_scratch,
                                is_packed,
                                always_copy,
                            )
                        } else {
                            cx.span_err(mitem.span, "this trait cannot be derived for unions");
                            return;
                        }
                    }
                    _ => unreachable!(),
                };
                // Keep the lint attributes of the previous item to control how the
                // generated implementations are linted
                let mut attrs = newitem.attrs.clone();
                attrs.extend(
                    item.attrs
                        .iter()
                        .filter(|a| {
                            [
                                sym::allow,
                                sym::warn,
                                sym::deny,
                                sym::forbid,
                                sym::stable,
                                sym::unstable,
                            ]
                            .contains(&a.name_or_empty())
                        })
                        .cloned(),
                );
                push(Annotatable::Item(P(ast::Item { attrs, ..(*newitem).clone() })))
            }
            _ => unreachable!(),
        }
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
    fn create_derived_impl(
        &self,
        cx: &mut ExtCtxt<'_>,
        type_ident: Ident,
        generics: &Generics,
        field_tys: Vec<P<ast::Ty>>,
        methods: Vec<P<ast::AssocItem>>,
    ) -> P<ast::Item> {
        let trait_path = self.path.to_path(cx, self.span, type_ident, generics);

        // Transform associated types from `deriving::ty::Ty` into `ast::AssocItem`
        let associated_types = self.associated_types.iter().map(|&(ident, ref type_def)| {
            P(ast::AssocItem {
                id: ast::DUMMY_NODE_ID,
                span: self.span,
                ident,
                vis: ast::Visibility {
                    span: self.span.shrink_to_lo(),
                    kind: ast::VisibilityKind::Inherited,
                    tokens: None,
                },
                attrs: ast::AttrVec::new(),
                kind: ast::AssocItemKind::TyAlias(Box::new(ast::TyAlias {
                    defaultness: ast::Defaultness::Final,
                    generics: Generics::default(),
                    where_clauses: (
                        ast::TyAliasWhereClause::default(),
                        ast::TyAliasWhereClause::default(),
                    ),
                    where_predicates_split: 0,
                    bounds: Vec::new(),
                    ty: Some(type_def.to_ty(cx, self.span, type_ident, generics)),
                })),
                tokens: None,
            })
        });

        let Generics { mut params, mut where_clause, .. } =
            self.generics.to_generics(cx, self.span, type_ident, generics);
        where_clause.span = generics.where_clause.span;
        let ctxt = self.span.ctxt();
        let span = generics.span.with_ctxt(ctxt);

        // Create the generic parameters
        params.extend(generics.params.iter().map(|param| match &param.kind {
            GenericParamKind::Lifetime { .. } => param.clone(),
            GenericParamKind::Type { .. } => {
                // I don't think this can be moved out of the loop, since
                // a GenericBound requires an ast id
                let bounds: Vec<_> =
                    // extra restrictions on the generics parameters to the
                    // type being derived upon
                    self.additional_bounds.iter().map(|p| {
                        cx.trait_bound(p.to_path(cx, self.span, type_ident, generics))
                    }).chain(
                        // require the current trait
                        iter::once(cx.trait_bound(trait_path.clone()))
                    ).chain(
                        // also add in any bounds from the declaration
                        param.bounds.iter().cloned()
                    ).collect();

                cx.typaram(param.ident.span.with_ctxt(ctxt), param.ident, bounds, None)
            }
            GenericParamKind::Const { ty, kw_span, .. } => {
                let const_nodefault_kind = GenericParamKind::Const {
                    ty: ty.clone(),
                    kw_span: kw_span.with_ctxt(ctxt),

                    // We can't have default values inside impl block
                    default: None,
                };
                let mut param_clone = param.clone();
                param_clone.kind = const_nodefault_kind;
                param_clone
            }
        }));

        // and similarly for where clauses
        where_clause.predicates.extend(generics.where_clause.predicates.iter().map(|clause| {
            match clause {
                ast::WherePredicate::BoundPredicate(wb) => {
                    let span = wb.span.with_ctxt(ctxt);
                    ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                        span,
                        ..wb.clone()
                    })
                }
                ast::WherePredicate::RegionPredicate(wr) => {
                    let span = wr.span.with_ctxt(ctxt);
                    ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                        span,
                        ..wr.clone()
                    })
                }
                ast::WherePredicate::EqPredicate(we) => {
                    let span = we.span.with_ctxt(ctxt);
                    ast::WherePredicate::EqPredicate(ast::WhereEqPredicate { span, ..we.clone() })
                }
            }
        }));

        {
            // Extra scope required here so ty_params goes out of scope before params is moved

            let mut ty_params = params
                .iter()
                .filter(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }))
                .peekable();

            if ty_params.peek().is_some() {
                let ty_param_names: Vec<Symbol> =
                    ty_params.map(|ty_param| ty_param.ident.name).collect();

                for field_ty in field_tys {
                    let field_ty_params = find_type_parameters(&field_ty, &ty_param_names, cx);

                    for field_ty_param in field_ty_params {
                        // if we have already handled this type, skip it
                        if let ast::TyKind::Path(_, ref p) = field_ty_param.ty.kind {
                            if p.segments.len() == 1
                                && ty_param_names.contains(&p.segments[0].ident.name)
                            {
                                continue;
                            };
                        }
                        let mut bounds: Vec<_> = self
                            .additional_bounds
                            .iter()
                            .map(|p| cx.trait_bound(p.to_path(cx, self.span, type_ident, generics)))
                            .collect();

                        // require the current trait
                        bounds.push(cx.trait_bound(trait_path.clone()));

                        let predicate = ast::WhereBoundPredicate {
                            span: self.span,
                            bound_generic_params: field_ty_param.bound_generic_params,
                            bounded_ty: field_ty_param.ty,
                            bounds,
                        };

                        let predicate = ast::WherePredicate::BoundPredicate(predicate);
                        where_clause.predicates.push(predicate);
                    }
                }
            }
        }

        let trait_generics = Generics { params, where_clause, span };

        // Create the reference to the trait.
        let trait_ref = cx.trait_ref(trait_path);

        let self_params: Vec<_> = generics
            .params
            .iter()
            .map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    GenericArg::Lifetime(cx.lifetime(param.ident.span.with_ctxt(ctxt), param.ident))
                }
                GenericParamKind::Type { .. } => {
                    GenericArg::Type(cx.ty_ident(param.ident.span.with_ctxt(ctxt), param.ident))
                }
                GenericParamKind::Const { .. } => {
                    GenericArg::Const(cx.const_ident(param.ident.span.with_ctxt(ctxt), param.ident))
                }
            })
            .collect();

        // Create the type of `self`.
        let path = cx.path_all(self.span, false, vec![type_ident], self_params);
        let self_type = cx.ty_path(path);

        let attr = cx.attribute(cx.meta_word(self.span, sym::automatically_derived));
        let attrs = thin_vec![attr];
        let opt_trait_ref = Some(trait_ref);

        cx.item(
            self.span,
            Ident::empty(),
            attrs,
            ast::ItemKind::Impl(Box::new(ast::Impl {
                unsafety: ast::Unsafe::No,
                polarity: ast::ImplPolarity::Positive,
                defaultness: ast::Defaultness::Final,
                constness: ast::Const::No,
                generics: trait_generics,
                of_trait: opt_trait_ref,
                self_ty: self_type,
                items: methods.into_iter().chain(associated_types).collect(),
            })),
        )
    }

    fn expand_struct_def(
        &self,
        cx: &mut ExtCtxt<'_>,
        struct_def: &'a VariantData,
        type_ident: Ident,
        generics: &Generics,
        from_scratch: bool,
        is_packed: bool,
        always_copy: bool,
    ) -> P<ast::Item> {
        let field_tys: Vec<P<ast::Ty>> =
            struct_def.fields().iter().map(|field| field.ty.clone()).collect();

        let methods = self
            .methods
            .iter()
            .map(|method_def| {
                let (explicit_self, selflike_args, nonselflike_args, nonself_arg_tys) =
                    method_def.extract_arg_details(cx, self, type_ident, generics);

                let body = if from_scratch || method_def.is_static() {
                    method_def.expand_static_struct_method_body(
                        cx,
                        self,
                        struct_def,
                        type_ident,
                        &nonselflike_args,
                    )
                } else {
                    method_def.expand_struct_method_body(
                        cx,
                        self,
                        struct_def,
                        type_ident,
                        &selflike_args,
                        &nonselflike_args,
                        is_packed,
                        always_copy,
                    )
                };

                method_def.create_method(
                    cx,
                    self,
                    type_ident,
                    generics,
                    explicit_self,
                    nonself_arg_tys,
                    body,
                )
            })
            .collect();

        self.create_derived_impl(cx, type_ident, generics, field_tys, methods)
    }

    fn expand_enum_def(
        &self,
        cx: &mut ExtCtxt<'_>,
        enum_def: &'a EnumDef,
        type_ident: Ident,
        generics: &Generics,
        from_scratch: bool,
    ) -> P<ast::Item> {
        let mut field_tys = Vec::new();

        for variant in &enum_def.variants {
            field_tys.extend(variant.data.fields().iter().map(|field| field.ty.clone()));
        }

        let methods = self
            .methods
            .iter()
            .map(|method_def| {
                let (explicit_self, selflike_args, nonselflike_args, nonself_arg_tys) =
                    method_def.extract_arg_details(cx, self, type_ident, generics);

                let body = if from_scratch || method_def.is_static() {
                    method_def.expand_static_enum_method_body(
                        cx,
                        self,
                        enum_def,
                        type_ident,
                        &nonselflike_args,
                    )
                } else {
                    method_def.expand_enum_method_body(
                        cx,
                        self,
                        enum_def,
                        type_ident,
                        selflike_args,
                        &nonselflike_args,
                    )
                };

                method_def.create_method(
                    cx,
                    self,
                    type_ident,
                    generics,
                    explicit_self,
                    nonself_arg_tys,
                    body,
                )
            })
            .collect();

        self.create_derived_impl(cx, type_ident, generics, field_tys, methods)
    }
}

impl<'a> MethodDef<'a> {
    fn call_substructure_method(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        type_ident: Ident,
        nonselflike_args: &[P<Expr>],
        fields: &SubstructureFields<'_>,
    ) -> BlockOrExpr {
        let span = trait_.span;
        let substructure = Substructure { type_ident, nonselflike_args, fields };
        let mut f = self.combine_substructure.borrow_mut();
        let f: &mut CombineSubstructureFunc<'_> = &mut *f;
        f(cx, span, &substructure)
    }

    fn get_ret_ty(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        generics: &Generics,
        type_ident: Ident,
    ) -> P<ast::Ty> {
        self.ret_ty.to_ty(cx, trait_.span, type_ident, generics)
    }

    fn is_static(&self) -> bool {
        !self.explicit_self
    }

    // The return value includes:
    // - explicit_self: The `&self` arg, if present.
    // - selflike_args: Expressions for `&self` (if present) and also any other
    //   args with the same type (e.g. the `other` arg in `PartialEq::eq`).
    // - nonselflike_args: Expressions for all the remaining args.
    // - nonself_arg_tys: Additional information about all the args other than
    //   `&self`.
    fn extract_arg_details(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        type_ident: Ident,
        generics: &Generics,
    ) -> (Option<ast::ExplicitSelf>, Vec<P<Expr>>, Vec<P<Expr>>, Vec<(Ident, P<ast::Ty>)>) {
        let mut selflike_args = Vec::new();
        let mut nonselflike_args = Vec::new();
        let mut nonself_arg_tys = Vec::new();
        let span = trait_.span;

        let explicit_self = if self.explicit_self {
            let (self_expr, explicit_self) = ty::get_explicit_self(cx, span);
            selflike_args.push(self_expr);
            Some(explicit_self)
        } else {
            None
        };

        for (ty, name) in self.nonself_args.iter() {
            let ast_ty = ty.to_ty(cx, span, type_ident, generics);
            let ident = Ident::new(*name, span);
            nonself_arg_tys.push((ident, ast_ty));

            let arg_expr = cx.expr_ident(span, ident);

            match ty {
                // Selflike (`&Self`) arguments only occur in non-static methods.
                Ref(box Self_, _) if !self.is_static() => selflike_args.push(arg_expr),
                Self_ => cx.span_bug(span, "`Self` in non-return position"),
                _ => nonselflike_args.push(arg_expr),
            }
        }

        (explicit_self, selflike_args, nonselflike_args, nonself_arg_tys)
    }

    fn create_method(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        type_ident: Ident,
        generics: &Generics,
        explicit_self: Option<ast::ExplicitSelf>,
        nonself_arg_tys: Vec<(Ident, P<ast::Ty>)>,
        body: BlockOrExpr,
    ) -> P<ast::AssocItem> {
        let span = trait_.span;
        // Create the generics that aren't for `Self`.
        let fn_generics = self.generics.to_generics(cx, span, type_ident, generics);

        let args = {
            let self_arg = explicit_self.map(|explicit_self| {
                let ident = Ident::with_dummy_span(kw::SelfLower).with_span_pos(span);
                ast::Param::from_self(ast::AttrVec::default(), explicit_self, ident)
            });
            let nonself_args =
                nonself_arg_tys.into_iter().map(|(name, ty)| cx.param(span, name, ty));
            self_arg.into_iter().chain(nonself_args).collect()
        };

        let ret_type = self.get_ret_ty(cx, trait_, generics, type_ident);

        let method_ident = Ident::new(self.name, span);
        let fn_decl = cx.fn_decl(args, ast::FnRetTy::Ty(ret_type));
        let body_block = body.into_block(cx, span);

        let trait_lo_sp = span.shrink_to_lo();

        let sig = ast::FnSig { header: ast::FnHeader::default(), decl: fn_decl, span };
        let defaultness = ast::Defaultness::Final;

        // Create the method.
        P(ast::AssocItem {
            id: ast::DUMMY_NODE_ID,
            attrs: self.attributes.clone(),
            span,
            vis: ast::Visibility {
                span: trait_lo_sp,
                kind: ast::VisibilityKind::Inherited,
                tokens: None,
            },
            ident: method_ident,
            kind: ast::AssocItemKind::Fn(Box::new(ast::Fn {
                defaultness,
                sig,
                generics: fn_generics,
                body: Some(body_block),
            })),
            tokens: None,
        })
    }

    /// The normal case uses field access.
    /// ```
    /// #[derive(PartialEq)]
    /// # struct Dummy;
    /// struct A { x: u8, y: u8 }
    ///
    /// // equivalent to:
    /// impl PartialEq for A {
    ///     fn eq(&self, other: &A) -> bool {
    ///         self.x == other.x && self.y == other.y
    ///     }
    /// }
    /// ```
    /// But if the struct is `repr(packed)`, we can't use something like
    /// `&self.x` because that might cause an unaligned ref. So for any trait
    /// method that takes a reference, if the struct impls `Copy` then we use a
    /// local block to force a copy:
    /// ```
    /// # struct A { x: u8, y: u8 }
    /// impl PartialEq for A {
    ///     fn eq(&self, other: &A) -> bool {
    ///         // Desugars to `{ self.x }.eq(&{ other.y }) && ...`
    ///         { self.x } == { other.y } && { self.y } == { other.y }
    ///     }
    /// }
    /// impl Hash for A {
    ///     fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
    ///         ::core::hash::Hash::hash(&{ self.x }, state);
    ///         ::core::hash::Hash::hash(&{ self.y }, state)
    ///     }
    /// }
    /// ```
    /// If the struct doesn't impl `Copy`, we use let-destructuring with `ref`:
    /// ```
    /// # struct A { x: u8, y: u8 }
    /// impl PartialEq for A {
    ///     fn eq(&self, other: &A) -> bool {
    ///         let Self { x: ref __self_0_0, y: ref __self_0_1 } = *self;
    ///         let Self { x: ref __self_1_0, y: ref __self_1_1 } = *other;
    ///         *__self_0_0 == *__self_1_0 && *__self_0_1 == *__self_1_1
    ///     }
    /// }
    /// ```
    /// This latter case only works if the fields match the alignment required
    /// by the `packed(N)` attribute. (We'll get errors later on if not.)
    fn expand_struct_method_body<'b>(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'b>,
        struct_def: &'b VariantData,
        type_ident: Ident,
        selflike_args: &[P<Expr>],
        nonselflike_args: &[P<Expr>],
        is_packed: bool,
        always_copy: bool,
    ) -> BlockOrExpr {
        let span = trait_.span;
        assert!(selflike_args.len() == 1 || selflike_args.len() == 2);

        let mk_body = |cx, selflike_fields| {
            self.call_substructure_method(
                cx,
                trait_,
                type_ident,
                nonselflike_args,
                &Struct(struct_def, selflike_fields),
            )
        };

        if !is_packed {
            let selflike_fields =
                trait_.create_struct_field_access_fields(cx, selflike_args, struct_def, false);
            mk_body(cx, selflike_fields)
        } else if always_copy {
            let selflike_fields =
                trait_.create_struct_field_access_fields(cx, selflike_args, struct_def, true);
            mk_body(cx, selflike_fields)
        } else {
            // Neither packed nor copy. Need to use ref patterns.
            let prefixes: Vec<_> =
                (0..selflike_args.len()).map(|i| format!("__self_{}", i)).collect();
            let addr_of = always_copy;
            let selflike_fields =
                trait_.create_struct_pattern_fields(cx, struct_def, &prefixes, addr_of);
            let mut body = mk_body(cx, selflike_fields);

            let struct_path = cx.path(span, vec![Ident::new(kw::SelfUpper, type_ident.span)]);
            let by_ref = ByRef::from(is_packed && !always_copy);
            let patterns =
                trait_.create_struct_patterns(cx, struct_path, struct_def, &prefixes, by_ref);

            // Do the let-destructuring.
            let mut stmts: Vec<_> = iter::zip(selflike_args, patterns)
                .map(|(selflike_arg_expr, pat)| {
                    let selflike_arg_expr = cx.expr_deref(span, selflike_arg_expr.clone());
                    cx.stmt_let_pat(span, pat, selflike_arg_expr)
                })
                .collect();
            stmts.extend(std::mem::take(&mut body.0));
            BlockOrExpr(stmts, body.1)
        }
    }

    fn expand_static_struct_method_body(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        struct_def: &VariantData,
        type_ident: Ident,
        nonselflike_args: &[P<Expr>],
    ) -> BlockOrExpr {
        let summary = trait_.summarise_struct(cx, struct_def);

        self.call_substructure_method(
            cx,
            trait_,
            type_ident,
            nonselflike_args,
            &StaticStruct(struct_def, summary),
        )
    }

    /// ```
    /// #[derive(PartialEq)]
    /// # struct Dummy;
    /// enum A {
    ///     A1,
    ///     A2(i32)
    /// }
    /// ```
    /// is equivalent to:
    /// ```
    /// impl ::core::cmp::PartialEq for A {
    ///     #[inline]
    ///     fn eq(&self, other: &A) -> bool {
    ///         let __self_tag = ::core::intrinsics::discriminant_value(self);
    ///         let __arg1_tag = ::core::intrinsics::discriminant_value(other);
    ///         __self_tag == __arg1_tag &&
    ///             match (self, other) {
    ///                 (A::A2(__self_0), A::A2(__arg1_0)) =>
    ///                     *__self_0 == *__arg1_0,
    ///                 _ => true,
    ///             }
    ///     }
    /// }
    /// ```
    /// Creates a tag check combined with a match for a tuple of all
    /// `selflike_args`, with an arm for each variant with fields, possibly an
    /// arm for each fieldless variant (if `!unify_fieldless_variants` is not
    /// true), and possibly a default arm.
    fn expand_enum_method_body<'b>(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'b>,
        enum_def: &'b EnumDef,
        type_ident: Ident,
        selflike_args: Vec<P<Expr>>,
        nonselflike_args: &[P<Expr>],
    ) -> BlockOrExpr {
        let span = trait_.span;
        let variants = &enum_def.variants;

        // Traits that unify fieldless variants always use the tag(s).
        let uses_tags = self.unify_fieldless_variants;

        // There is no sensible code to be generated for *any* deriving on a
        // zero-variant enum. So we just generate a failing expression.
        if variants.is_empty() {
            return BlockOrExpr(vec![], Some(deriving::call_unreachable(cx, span)));
        }

        let prefixes = iter::once("__self".to_string())
            .chain(
                selflike_args
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(arg_count, _selflike_arg)| format!("__arg{}", arg_count)),
            )
            .collect::<Vec<String>>();

        // Build a series of let statements mapping each selflike_arg
        // to its discriminant value.
        //
        // e.g. for `PartialEq::eq` builds two statements:
        // ```
        // let __self_tag = ::core::intrinsics::discriminant_value(self);
        // let __arg1_tag = ::core::intrinsics::discriminant_value(other);
        // ```
        let get_tag_pieces = |cx: &ExtCtxt<'_>| {
            let tag_idents: Vec<_> = prefixes
                .iter()
                .map(|name| Ident::from_str_and_span(&format!("{}_tag", name), span))
                .collect();

            let mut tag_exprs: Vec<_> = tag_idents
                .iter()
                .map(|&ident| cx.expr_addr_of(span, cx.expr_ident(span, ident)))
                .collect();

            let self_expr = tag_exprs.remove(0);
            let other_selflike_exprs = tag_exprs;
            let tag_field = FieldInfo { span, name: None, self_expr, other_selflike_exprs };

            let tag_let_stmts: Vec<_> = iter::zip(&tag_idents, &selflike_args)
                .map(|(&ident, selflike_arg)| {
                    let variant_value = deriving::call_intrinsic(
                        cx,
                        span,
                        sym::discriminant_value,
                        vec![selflike_arg.clone()],
                    );
                    cx.stmt_let(span, false, ident, variant_value)
                })
                .collect();

            (tag_field, tag_let_stmts)
        };

        // There are some special cases involving fieldless enums where no
        // match is necessary.
        let all_fieldless = variants.iter().all(|v| v.data.fields().is_empty());
        if all_fieldless {
            if uses_tags && variants.len() > 1 {
                // If the type is fieldless and the trait uses the tag and
                // there are multiple variants, we need just an operation on
                // the tag(s).
                let (tag_field, mut tag_let_stmts) = get_tag_pieces(cx);
                let mut tag_check = self.call_substructure_method(
                    cx,
                    trait_,
                    type_ident,
                    nonselflike_args,
                    &EnumTag(tag_field, None),
                );
                tag_let_stmts.append(&mut tag_check.0);
                return BlockOrExpr(tag_let_stmts, tag_check.1);
            }

            if variants.len() == 1 {
                // If there is a single variant, we don't need an operation on
                // the tag(s). Just use the most degenerate result.
                return self.call_substructure_method(
                    cx,
                    trait_,
                    type_ident,
                    nonselflike_args,
                    &EnumMatching(0, 1, &variants[0], Vec::new()),
                );
            };
        }

        // These arms are of the form:
        // (Variant1, Variant1, ...) => Body1
        // (Variant2, Variant2, ...) => Body2
        // ...
        // where each tuple has length = selflike_args.len()
        let mut match_arms: Vec<ast::Arm> = variants
            .iter()
            .enumerate()
            .filter(|&(_, v)| !(self.unify_fieldless_variants && v.data.fields().is_empty()))
            .map(|(index, variant)| {
                // A single arm has form (&VariantK, &VariantK, ...) => BodyK
                // (see "Final wrinkle" note below for why.)

                let addr_of = false; // because enums can't be repr(packed)
                let fields =
                    trait_.create_struct_pattern_fields(cx, &variant.data, &prefixes, addr_of);

                let sp = variant.span.with_ctxt(trait_.span.ctxt());
                let variant_path = cx.path(sp, vec![type_ident, variant.ident]);
                let by_ref = ByRef::No; // because enums can't be repr(packed)
                let mut subpats: Vec<_> = trait_.create_struct_patterns(
                    cx,
                    variant_path,
                    &variant.data,
                    &prefixes,
                    by_ref,
                );

                // `(VariantK, VariantK, ...)` or just `VariantK`.
                let single_pat = if subpats.len() == 1 {
                    subpats.pop().unwrap()
                } else {
                    cx.pat_tuple(span, subpats)
                };

                // For the BodyK, we need to delegate to our caller,
                // passing it an EnumMatching to indicate which case
                // we are in.
                //
                // Now, for some given VariantK, we have built up
                // expressions for referencing every field of every
                // Self arg, assuming all are instances of VariantK.
                // Build up code associated with such a case.
                let substructure = EnumMatching(index, variants.len(), variant, fields);
                let arm_expr = self
                    .call_substructure_method(
                        cx,
                        trait_,
                        type_ident,
                        nonselflike_args,
                        &substructure,
                    )
                    .into_expr(cx, span);

                cx.arm(span, single_pat, arm_expr)
            })
            .collect();

        // Add a default arm to the match, if necessary.
        let first_fieldless = variants.iter().find(|v| v.data.fields().is_empty());
        let default = match first_fieldless {
            Some(v) if self.unify_fieldless_variants => {
                // We need a default case that handles all the fieldless
                // variants. The index and actual variant aren't meaningful in
                // this case, so just use dummy values.
                Some(
                    self.call_substructure_method(
                        cx,
                        trait_,
                        type_ident,
                        nonselflike_args,
                        &EnumMatching(0, variants.len(), v, Vec::new()),
                    )
                    .into_expr(cx, span),
                )
            }
            _ if variants.len() > 1 && selflike_args.len() > 1 => {
                // Because we know that all the arguments will match if we reach
                // the match expression we add the unreachable intrinsics as the
                // result of the default which should help llvm in optimizing it.
                Some(deriving::call_unreachable(cx, span))
            }
            _ => None,
        };
        if let Some(arm) = default {
            match_arms.push(cx.arm(span, cx.pat_wild(span), arm));
        }

        // Create a match expression with one arm per discriminant plus
        // possibly a default arm, e.g.:
        //      match (self, other) {
        //          (Variant1, Variant1, ...) => Body1
        //          (Variant2, Variant2, ...) => Body2,
        //          ...
        //          _ => ::core::intrinsics::unreachable()
        //      }
        let get_match_expr = |mut selflike_args: Vec<P<Expr>>| {
            let match_arg = if selflike_args.len() == 1 {
                selflike_args.pop().unwrap()
            } else {
                cx.expr(span, ast::ExprKind::Tup(selflike_args))
            };
            cx.expr_match(span, match_arg, match_arms)
        };

        // If the trait uses the tag and there are multiple variants, we need
        // to add a tag check operation before the match. Otherwise, the match
        // is enough.
        if uses_tags && variants.len() > 1 {
            let (tag_field, mut tag_let_stmts) = get_tag_pieces(cx);

            // Combine a tag check with the match.
            let mut tag_check_plus_match = self.call_substructure_method(
                cx,
                trait_,
                type_ident,
                nonselflike_args,
                &EnumTag(tag_field, Some(get_match_expr(selflike_args))),
            );
            tag_let_stmts.append(&mut tag_check_plus_match.0);
            BlockOrExpr(tag_let_stmts, tag_check_plus_match.1)
        } else {
            BlockOrExpr(vec![], Some(get_match_expr(selflike_args)))
        }
    }

    fn expand_static_enum_method_body(
        &self,
        cx: &mut ExtCtxt<'_>,
        trait_: &TraitDef<'_>,
        enum_def: &EnumDef,
        type_ident: Ident,
        nonselflike_args: &[P<Expr>],
    ) -> BlockOrExpr {
        let summary = enum_def
            .variants
            .iter()
            .map(|v| {
                let sp = v.span.with_ctxt(trait_.span.ctxt());
                let summary = trait_.summarise_struct(cx, &v.data);
                (v.ident, sp, summary)
            })
            .collect();
        self.call_substructure_method(
            cx,
            trait_,
            type_ident,
            nonselflike_args,
            &StaticEnum(enum_def, summary),
        )
    }
}

// general helper methods.
impl<'a> TraitDef<'a> {
    fn summarise_struct(&self, cx: &mut ExtCtxt<'_>, struct_def: &VariantData) -> StaticFields {
        let mut named_idents = Vec::new();
        let mut just_spans = Vec::new();
        for field in struct_def.fields() {
            let sp = field.span.with_ctxt(self.span.ctxt());
            match field.ident {
                Some(ident) => named_idents.push((ident, sp)),
                _ => just_spans.push(sp),
            }
        }

        let is_tuple = matches!(struct_def, ast::VariantData::Tuple(..));
        match (just_spans.is_empty(), named_idents.is_empty()) {
            (false, false) => {
                cx.span_bug(self.span, "a struct with named and unnamed fields in generic `derive`")
            }
            // named fields
            (_, false) => Named(named_idents),
            // unnamed fields
            (false, _) => Unnamed(just_spans, is_tuple),
            // empty
            _ => Named(Vec::new()),
        }
    }

    fn create_struct_patterns(
        &self,
        cx: &mut ExtCtxt<'_>,
        struct_path: ast::Path,
        struct_def: &'a VariantData,
        prefixes: &[String],
        by_ref: ByRef,
    ) -> Vec<P<ast::Pat>> {
        prefixes
            .iter()
            .map(|prefix| {
                let pieces_iter =
                    struct_def.fields().iter().enumerate().map(|(i, struct_field)| {
                        let sp = struct_field.span.with_ctxt(self.span.ctxt());
                        let ident = self.mk_pattern_ident(prefix, i);
                        let path = ident.with_span_pos(sp);
                        (
                            sp,
                            struct_field.ident,
                            cx.pat(
                                path.span,
                                PatKind::Ident(
                                    BindingAnnotation(by_ref, Mutability::Not),
                                    path,
                                    None,
                                ),
                            ),
                        )
                    });

                let struct_path = struct_path.clone();
                match *struct_def {
                    VariantData::Struct(..) => {
                        let field_pats = pieces_iter
                            .map(|(sp, ident, pat)| {
                                if ident.is_none() {
                                    cx.span_bug(
                                        sp,
                                        "a braced struct with unnamed fields in `derive`",
                                    );
                                }
                                ast::PatField {
                                    ident: ident.unwrap(),
                                    is_shorthand: false,
                                    attrs: ast::AttrVec::new(),
                                    id: ast::DUMMY_NODE_ID,
                                    span: pat.span.with_ctxt(self.span.ctxt()),
                                    pat,
                                    is_placeholder: false,
                                }
                            })
                            .collect();
                        cx.pat_struct(self.span, struct_path, field_pats)
                    }
                    VariantData::Tuple(..) => {
                        let subpats = pieces_iter.map(|(_, _, subpat)| subpat).collect();
                        cx.pat_tuple_struct(self.span, struct_path, subpats)
                    }
                    VariantData::Unit(..) => cx.pat_path(self.span, struct_path),
                }
            })
            .collect()
    }

    fn create_fields<F>(&self, struct_def: &'a VariantData, mk_exprs: F) -> Vec<FieldInfo>
    where
        F: Fn(usize, &ast::FieldDef, Span) -> Vec<P<ast::Expr>>,
    {
        struct_def
            .fields()
            .iter()
            .enumerate()
            .map(|(i, struct_field)| {
                // For this field, get an expr for each selflike_arg. E.g. for
                // `PartialEq::eq`, one for each of `&self` and `other`.
                let sp = struct_field.span.with_ctxt(self.span.ctxt());
                let mut exprs: Vec<_> = mk_exprs(i, struct_field, sp);
                let self_expr = exprs.remove(0);
                let other_selflike_exprs = exprs;
                FieldInfo {
                    span: sp.with_ctxt(self.span.ctxt()),
                    name: struct_field.ident,
                    self_expr,
                    other_selflike_exprs,
                }
            })
            .collect()
    }

    fn mk_pattern_ident(&self, prefix: &str, i: usize) -> Ident {
        Ident::from_str_and_span(&format!("{}_{}", prefix, i), self.span)
    }

    fn create_struct_pattern_fields(
        &self,
        cx: &mut ExtCtxt<'_>,
        struct_def: &'a VariantData,
        prefixes: &[String],
        addr_of: bool,
    ) -> Vec<FieldInfo> {
        self.create_fields(struct_def, |i, _struct_field, sp| {
            prefixes
                .iter()
                .map(|prefix| {
                    let ident = self.mk_pattern_ident(prefix, i);
                    let expr = cx.expr_path(cx.path_ident(sp, ident));
                    if addr_of { cx.expr_addr_of(sp, expr) } else { expr }
                })
                .collect()
        })
    }

    fn create_struct_field_access_fields(
        &self,
        cx: &mut ExtCtxt<'_>,
        selflike_args: &[P<Expr>],
        struct_def: &'a VariantData,
        copy: bool,
    ) -> Vec<FieldInfo> {
        self.create_fields(struct_def, |i, struct_field, sp| {
            selflike_args
                .iter()
                .map(|selflike_arg| {
                    // Note: we must use `struct_field.span` rather than `sp` in the
                    // `unwrap_or_else` case otherwise the hygiene is wrong and we get
                    // "field `0` of struct `Point` is private" errors on tuple
                    // structs.
                    let mut field_expr = cx.expr(
                        sp,
                        ast::ExprKind::Field(
                            selflike_arg.clone(),
                            struct_field.ident.unwrap_or_else(|| {
                                Ident::from_str_and_span(&i.to_string(), struct_field.span)
                            }),
                        ),
                    );
                    if copy {
                        field_expr = cx.expr_block(
                            cx.block(struct_field.span, vec![cx.stmt_expr(field_expr)]),
                        );
                    }
                    cx.expr_addr_of(sp, field_expr)
                })
                .collect()
        })
    }
}

/// The function passed to `cs_fold` is called repeatedly with a value of this
/// type. It describes one part of the code generation. The result is always an
/// expression.
pub enum CsFold<'a> {
    /// The basic case: a field expression for one or more selflike args. E.g.
    /// for `PartialEq::eq` this is something like `self.x == other.x`.
    Single(&'a FieldInfo),

    /// The combination of two field expressions. E.g. for `PartialEq::eq` this
    /// is something like `<field1 equality> && <field2 equality>`.
    Combine(Span, P<Expr>, P<Expr>),

    // The fallback case for a struct or enum variant with no fields.
    Fieldless,
}

/// Folds over fields, combining the expressions for each field in a sequence.
/// Statics may not be folded over.
pub fn cs_fold<F>(
    use_foldl: bool,
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substructure: &Substructure<'_>,
    mut f: F,
) -> P<Expr>
where
    F: FnMut(&mut ExtCtxt<'_>, CsFold<'_>) -> P<Expr>,
{
    match substructure.fields {
        EnumMatching(.., all_fields) | Struct(_, all_fields) => {
            if all_fields.is_empty() {
                return f(cx, CsFold::Fieldless);
            }

            let (base_field, rest) = if use_foldl {
                all_fields.split_first().unwrap()
            } else {
                all_fields.split_last().unwrap()
            };

            let base_expr = f(cx, CsFold::Single(base_field));

            let op = |old, field: &FieldInfo| {
                let new = f(cx, CsFold::Single(field));
                f(cx, CsFold::Combine(field.span, old, new))
            };

            if use_foldl {
                rest.iter().fold(base_expr, op)
            } else {
                rest.iter().rfold(base_expr, op)
            }
        }
        EnumTag(tag_field, match_expr) => {
            let tag_check_expr = f(cx, CsFold::Single(tag_field));
            if let Some(match_expr) = match_expr {
                if use_foldl {
                    f(cx, CsFold::Combine(trait_span, tag_check_expr, match_expr.clone()))
                } else {
                    f(cx, CsFold::Combine(trait_span, match_expr.clone(), tag_check_expr))
                }
            } else {
                tag_check_expr
            }
        }
        StaticEnum(..) | StaticStruct(..) => cx.span_bug(trait_span, "static function in `derive`"),
    }
}
