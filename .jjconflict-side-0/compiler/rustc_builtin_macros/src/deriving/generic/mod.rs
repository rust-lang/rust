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
//! - `EnumDiscr` when `Self` is an enum, for comparing the enum discriminants.
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
//! struct A {
//!     x: i32,
//! }
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
//!
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
//! ```text
//! Struct(vec![FieldInfo {
//!     span: <span of x>,
//!     name: Some(<ident of x>),
//!     self_: <expr for &self.x>,
//!     other: vec![<expr for &other.x>],
//! }])
//! ```
//!
//! For the `B` impl, called with `B(a)` and `B(b)`,
//!
//! ```text
//! Struct(vec![FieldInfo {
//!     span: <span of i32>,
//!     name: None,
//!     self_: <expr for &a>,
//!     other: vec![<expr for &b>],
//! }])
//! ```
//!
//! ## Enums
//!
//! When generating the `expr` for a call with `self == C0(a)` and `other
//! == C0(b)`, the SubstructureFields is
//!
//! ```text
//! EnumMatching(
//!     0,
//!     <ast::Variant for C0>,
//!     vec![FieldInfo {
//!         span: <span of i32>,
//!         name: None,
//!         self_: <expr for &a>,
//!         other: vec![<expr for &b>],
//!     }],
//! )
//! ```
//!
//! For `C1 {x}` and `C1 {x}`,
//!
//! ```text
//! EnumMatching(
//!     1,
//!     <ast::Variant for C1>,
//!     vec![FieldInfo {
//!         span: <span of x>,
//!         name: Some(<ident of x>),
//!         self_: <expr for &self.x>,
//!         other: vec![<expr for &other.x>],
//!     }],
//! )
//! ```
//!
//! For the discriminants,
//!
//! ```text
//! EnumDiscr(
//!     &[<ident of self discriminant>, <ident of other discriminant>],
//!     <expr to combine with>,
//! )
//! ```
//!
//! Note that this setup doesn't allow for the brute-force "match every variant
//! against every other variant" approach, which is bad because it produces a
//! quadratic amount of code (see #15375).
//!
//! ## Static
//!
//! A static method on the types above would result in,
//!
//! ```text
//! StaticStruct(<ast::VariantData of A>, Named(vec![(<ident of x>, <span of x>)]))
//!
//! StaticStruct(<ast::VariantData of B>, Unnamed(vec![<span of x>]))
//!
//! StaticEnum(
//!     <ast::EnumDef of C>,
//!     vec![
//!         (<ident of C0>, <span of C0>, Unnamed(vec![<span of i32>])),
//!         (<ident of C1>, <span of C1>, Named(vec![(<ident of x>, <span of x>)])),
//!     ],
//! )
//! ```

use std::ops::Not;
use std::vec;

pub(crate) use Substructure::*;
pub(crate) use rustc_ast as ast;
use rustc_ast::token::{IdentKind, LitKind, Token, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, Spacing, TokenTree};
use rustc_ast::{
    AttrArgs, DelimArgs, EnumDef, Expr, GenericArg, GenericParam, GenericParamKind, Generics,
    Safety, SelfKind, VariantData,
};
use rustc_attr_ir::{Attribute, AttributeKind, ReprPacked};
use rustc_attr_parsing::AttributeParser;
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, Symbol, kw, respan, sym};
pub(crate) use smallvec::{SmallVec, smallvec};
use thin_vec::{ThinVec, thin_vec};

use crate::{deriving, diagnostics};

pub(crate) struct TraitDef<'a> {
    /// The span for the current #[derive(Foo)] header.
    pub span: Span,

    /// Path of the trait, including any type parameters
    pub path: ast::Path,

    /// Whether to skip adding the current trait as a bound to the type parameters of the type.
    pub skip_path_as_bound: bool,

    /// Whether `Copy` is needed as an additional bound on type parameters in a packed struct.
    pub needs_copy_as_bound_if_packed: bool,

    /// Additional bounds required of any type parameters of the type,
    /// other than the current trait
    pub additional_bounds: SmallVec<[ast::Path; 1]>,

    /// Can this trait be derived for unions?
    pub supports_unions: bool,

    pub methods: SmallVec<[MethodDef<'a>; 1]>,

    pub is_const: bool,

    /// The safety of the `impl`.
    pub safety: Safety,

    /// Whether the added `impl` should appear in rustdoc output.
    pub document: bool,
}

pub(crate) struct MethodDef<'a> {
    /// name of the method
    pub name: Symbol,
    /// List of generics, e.g., `R: rand::Rng`
    pub generics: Generics,

    /// Is there is a `&self` argument? If not, it is a static function.
    pub explicit_self: bool,

    /// Arguments other than the self argument.
    pub nonself_args: SmallVec<[(Box<ast::Ty>, Symbol); 1]>,

    /// Whether this method has another arg of type `&self`.
    pub has_other_selflike_arg: bool,

    /// Returns type
    pub ret_ty: Box<ast::Ty>,

    pub attributes: ast::AttrVec,

    pub fieldless_variants_strategy: FieldlessVariantsStrategy,

    pub combine_substructure: CombineSubstructureFunc<'a>,
}

/// How to handle fieldless enum variants.
#[derive(PartialEq)]
pub(crate) enum FieldlessVariantsStrategy {
    /// Combine fieldless variants into a single match arm.
    /// This assumes that relevant information has been handled
    /// by looking at the enum's discriminant.
    Unify,
    /// Don't do anything special about fieldless variants. They are
    /// handled like any other variant.
    Default,
    /// If all variants of the enum are fieldless, expand the special
    /// `AllFieldLessEnum` substructure, so that the entire enum can be handled
    /// at once.
    SpecializeIfAllVariantsFieldless,
}

/// Summary of the relevant parts of a struct/enum field.
pub(crate) struct FieldInfo {
    pub span: Span,
    /// None for tuple structs/normal enum variants, Some for normal
    /// structs/struct enum variants.
    pub name: Option<Ident>,
    /// The expression corresponding to this field of `self`
    /// (specifically, a reference to it).
    pub self_expr: Box<Expr>,
    /// The expression corresponding to a reference to this field in
    /// the other selflike argument.
    pub other_selflike_expr: Option<Box<Expr>>,
    pub maybe_scalar: bool,
}

/// A summary of the possible sets of fields.
pub(crate) enum Substructure<'a> {
    /// A non-static method where `Self` is a struct.
    Struct(&'a ast::VariantData, Vec<FieldInfo>),

    /// A non-static method handling the entire enum at once
    /// (after it has been determined that none of the enum
    /// variants has any fields).
    AllFieldlessEnum(&'a ast::EnumDef),

    /// Matching variants of the enum: ast::Variant,
    /// fields: the field name is only non-`None` in the case of a struct
    /// variant.
    EnumMatching(&'a ast::Variant, Vec<FieldInfo>),

    /// The discriminant of an enum. The field is the expression to combine the
    /// discriminant expression with; it will be `None` if no match is necessary.
    EnumDiscr(Option<Box<Expr>>),

    /// A static method where `Self` is a struct.
    StaticStruct(&'a ast::VariantData),

    /// A static method where `Self` is an enum.
    StaticEnum(&'a ast::EnumDef),
}

/// Combine the values of all the fields together. The last argument is
/// all the fields of all the structures.
pub(crate) type CombineSubstructureFunc<'a> =
    Box<dyn Fn(&ExtCtxt<'_>, Span, Substructure<'_>) -> BlockOrExpr + 'a>;

pub(crate) fn combine_substructure<'a>(
    f: impl Fn(&ExtCtxt<'_>, Span, Substructure<'_>) -> BlockOrExpr + 'a,
) -> CombineSubstructureFunc<'a> {
    Box::new(f)
}

struct TypeParameter {
    bound_generic_params: ThinVec<ast::GenericParam>,
    ty: Box<ast::Ty>,
}

/// The code snippets built up for derived code are sometimes used as blocks
/// (e.g. in a function body) and sometimes used as expressions (e.g. in a match
/// arm). This structure avoids committing to either form until necessary,
/// avoiding the insertion of any unnecessary blocks.
///
/// The statements come before the expression.
pub(crate) struct BlockOrExpr(ThinVec<ast::Stmt>, Option<Box<Expr>>);

impl BlockOrExpr {
    pub(crate) fn new_stmts(stmts: ThinVec<ast::Stmt>) -> BlockOrExpr {
        BlockOrExpr(stmts, None)
    }

    pub(crate) fn new_expr(expr: Box<Expr>) -> BlockOrExpr {
        BlockOrExpr(ThinVec::new(), Some(expr))
    }

    pub(crate) fn new_mixed(stmts: ThinVec<ast::Stmt>, expr: Option<Box<Expr>>) -> BlockOrExpr {
        BlockOrExpr(stmts, expr)
    }

    // Converts it into a block.
    fn into_block(mut self, cx: &ExtCtxt<'_>, span: Span) -> Box<ast::Block> {
        if let Some(expr) = self.1 {
            self.0.push(cx.stmt_expr(expr));
        }
        cx.block(span, self.0)
    }

    // Converts it into an expression.
    fn into_expr(self, cx: &ExtCtxt<'_>, span: Span) -> Box<Expr> {
        if self.0.is_empty() {
            match self.1 {
                None => cx.expr_block(cx.block(span, ThinVec::new())),
                Some(expr) => expr,
            }
        } else if let [stmt] = self.0.as_slice()
            && let ast::StmtKind::Expr(expr) = &stmt.kind
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
        bound_generic_params_stack: ThinVec<ast::GenericParam>,
        type_params: Vec<TypeParameter>,
    }

    impl<'a, 'b> visit::Visitor<'a> for Visitor<'a, 'b> {
        fn visit_ty(&mut self, ty: &'a ast::Ty) {
            let stack_len = self.bound_generic_params_stack.len();
            if let ast::TyKind::FnPtr(fn_ptr) = &ty.kind
                && !fn_ptr.generic_params.is_empty()
            {
                // Given a field `x: for<'a> fn(T::SomeType<'a>)`, we wan't to account for `'a` so
                // that we generate `where for<'a> T::SomeType<'a>: ::core::clone::Clone`. #122622
                self.bound_generic_params_stack.extend(fn_ptr.generic_params.iter().cloned());
            }

            if let ast::TyKind::Path(_, path) = &ty.kind
                && let Some(segment) = path.segments.first()
                && self.ty_param_names.contains(&segment.ident.name)
            {
                self.type_params.push(TypeParameter {
                    bound_generic_params: self.bound_generic_params_stack.clone(),
                    ty: Box::new(ty.clone()),
                });
            }

            visit::walk_ty(self, ty);
            self.bound_generic_params_stack.truncate(stack_len);
        }

        // Place bound generic params on a stack, to extract them when a type is encountered.
        fn visit_poly_trait_ref(&mut self, trait_ref: &'a ast::PolyTraitRef) {
            let stack_len = self.bound_generic_params_stack.len();
            self.bound_generic_params_stack.extend(trait_ref.bound_generic_params.iter().cloned());

            visit::walk_poly_trait_ref(self, trait_ref);

            self.bound_generic_params_stack.truncate(stack_len);
        }

        fn visit_mac_call(&mut self, mac: &ast::MacCall) {
            self.cx.dcx().emit_err(diagnostics::DeriveMacroCall { span: mac.span() });
        }
    }

    let mut visitor = Visitor {
        cx,
        ty_param_names,
        bound_generic_params_stack: ThinVec::new(),
        type_params: Vec::new(),
    };
    visit::Visitor::visit_ty(&mut visitor, ty);

    visitor.type_params
}

impl<'a> TraitDef<'a> {
    pub(crate) fn expand(
        self,
        cx: &ExtCtxt<'_>,
        item: &'a ast::Item,
        push: &mut dyn FnMut(Box<ast::Item>),
    ) {
        self.expand_ext(cx, item, push, false);
    }

    pub(crate) fn expand_ext(
        self,
        cx: &ExtCtxt<'_>,
        item: &'a ast::Item,
        push: &mut dyn FnMut(Box<ast::Item>),
        from_scratch: bool,
    ) {
        let span = self.span;
        let is_packed = matches!(
            AttributeParser::parse_limited_sym(cx.sess, &item.attrs, &[sym::repr]),
            Some(Attribute::Parsed(AttributeKind::Repr { reprs, .. })) if reprs.iter().any(|(x, _)| matches!(x, ReprPacked(..)))
        );

        let mut newitem = match &item.kind {
            ast::ItemKind::Union(..) if !self.supports_unions => {
                cx.dcx().emit_err(diagnostics::DeriveUnion { span });
                return;
            }
            ast::ItemKind::Struct(ident, generics, struct_def)
            | ast::ItemKind::Union(ident, generics, struct_def) => {
                let fields = struct_def.fields().iter();
                let methods = self.methods.iter().filter_map(|method_def| {
                    let body = if from_scratch || method_def.is_static() {
                        method_def.call_substructure_method(cx, span, StaticStruct(struct_def))
                    } else {
                        method_def.expand_struct_method_body(cx, span, struct_def, is_packed)
                    };

                    method_def.create_method(cx, span, body)
                });
                self.create_derived_impl(cx, *ident, generics, fields, methods, is_packed)
            }
            ast::ItemKind::Enum(ident, generics, enum_def) => {
                // We can skip generating the impl here, because `repr(packed)`
                // enums cause an error later on and to prevent ICEs like #133025.
                // This can only cause further compilation errors
                // downstream in blatantly illegal code, so it is fine.
                if is_packed {
                    return;
                }
                let fields = enum_def.variants.iter().flat_map(|variant| variant.data.fields());
                let methods = self.methods.iter().filter_map(|method_def| {
                    let body = if from_scratch || method_def.is_static() {
                        method_def.call_substructure_method(cx, span, StaticEnum(enum_def))
                    } else {
                        method_def.expand_enum_method_body(cx, span, enum_def)
                    };

                    method_def.create_method(cx, span, body)
                });
                self.create_derived_impl(cx, *ident, generics, fields, methods, is_packed)
            }
            _ => unreachable!(),
        };
        // Keep the lint attributes of the previous item to control how the
        // generated implementations are linted
        newitem.attrs.extend(
            item.attrs
                .iter()
                .filter(|a| {
                    a.has_any_name(&[
                        sym::allow,
                        sym::warn,
                        sym::deny,
                        sym::forbid,
                        sym::stable,
                        sym::unstable,
                    ])
                })
                .cloned(),
        );
        push(newitem);
    }

    /// Given that we are deriving a trait `DerivedTrait` for a type like:
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// struct Struct<'a, ..., 'z, A, B: DeclaredTrait, C, ..., Z>
    /// where
    ///     C: WhereTrait,
    /// {
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
    /// impl<'a, ..., 'z, A, B: DeclaredTrait, C, ..., Z>
    /// where
    ///     C: WhereTrait,
    ///     A: DerivedTrait + B1 + ... + BN,
    ///     B: DerivedTrait + B1 + ... + BN,
    ///     C: DerivedTrait + B1 + ... + BN,
    ///     B::Item: DerivedTrait + B1 + ... + BN,
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
        cx: &ExtCtxt<'_>,
        type_ident: Ident,
        generics: &Generics,
        fields: impl Iterator<Item = &'a ast::FieldDef>,
        methods: impl Iterator<Item = Box<ast::AssocItem>>,
        is_packed: bool,
    ) -> Box<ast::Item> {
        let mut where_clause = ast::WhereClause::default();
        where_clause.span = generics.where_clause.span;
        let ctxt = self.span.ctxt();

        // Create the generic parameters
        let params: ThinVec<_> = generics
            .params
            .iter()
            .map(|param| match &param.kind {
                GenericParamKind::Lifetime => param.clone(),
                GenericParamKind::Type { .. } => {
                    // Extra restrictions on the generics parameters to the
                    // type being derived upon.
                    let span = param.ident.span.with_ctxt(ctxt);
                    let bounds: ThinVec<_> = self
                        .additional_bounds
                        .iter()
                        .map(|p| cx.trait_bound(ast::Path { span, ..p.clone() }, self.is_const))
                        .chain(
                            // Add a bound for the current trait.
                            self.skip_path_as_bound.not().then(|| {
                                let mut trait_path = self.path.clone();
                                trait_path.span = span;
                                cx.trait_bound(trait_path, self.is_const)
                            }),
                        )
                        .chain({
                            // Add a `Copy` bound if required.
                            if is_packed && self.needs_copy_as_bound_if_packed {
                                let p = deriving::path_std!(cx, span, marker::Copy);
                                Some(cx.trait_bound(p, self.is_const))
                            } else {
                                None
                            }
                        })
                        .chain(
                            // Also add in any bounds from the declaration.
                            param.bounds.iter().cloned(),
                        )
                        .collect();

                    cx.typaram(param.ident, bounds, None)
                }
                GenericParamKind::Const { ty, span, .. } => {
                    let const_nodefault_kind = GenericParamKind::Const {
                        ty: ty.clone(),
                        span: span.with_ctxt(ctxt),

                        // We can't have default values inside impl block
                        default: None,
                    };
                    let mut param_clone = param.clone();
                    param_clone.kind = const_nodefault_kind;
                    param_clone
                }
            })
            .map(|mut param| {
                // Remove all attributes, because there might be helper attributes
                // from other macros that will not be valid in the expanded implementation.
                param.attrs.clear();
                param
            })
            .collect();

        // and similarly for where clauses
        where_clause.predicates.extend(generics.where_clause.predicates.iter().map(|clause| {
            ast::WherePredicate {
                attrs: clause.attrs.clone(),
                kind: clause.kind.clone(),
                id: ast::DUMMY_NODE_ID,
                span: clause.span.with_ctxt(ctxt),
                is_placeholder: false,
            }
        }));

        let ty_param_names: Vec<Symbol> = params
            .iter()
            .filter(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }))
            .map(|ty_param| ty_param.ident.name)
            .collect();

        if !ty_param_names.is_empty() {
            for field in fields {
                let field_ty_params = find_type_parameters(&field.ty, &ty_param_names, cx);

                for field_ty_param in field_ty_params {
                    // if we have already handled this type, skip it
                    if let ast::TyKind::Path(_, p) = &field_ty_param.ty.kind
                        && let [sole_segment] = &*p.segments
                        && ty_param_names.contains(&sole_segment.ident.name)
                    {
                        continue;
                    }
                    let mut bounds: ThinVec<_> = self
                        .additional_bounds
                        .iter()
                        .map(|p| cx.trait_bound(p.clone(), self.is_const))
                        .collect();

                    // Require the current trait.
                    if !self.skip_path_as_bound {
                        bounds.push(cx.trait_bound(self.path.clone(), self.is_const));
                    }

                    // Add a `Copy` bound if required.
                    if is_packed && self.needs_copy_as_bound_if_packed {
                        let p = deriving::path_std!(cx, self.span, marker::Copy);
                        bounds.push(cx.trait_bound(p, self.is_const));
                    }

                    if !bounds.is_empty() {
                        let predicate = ast::WhereBoundPredicate {
                            bound_generic_params: field_ty_param.bound_generic_params,
                            bounded_ty: field_ty_param.ty,
                            bounds,
                        };

                        let kind = ast::WherePredicateKind::BoundPredicate(predicate);
                        let predicate = ast::WherePredicate {
                            attrs: ThinVec::new(),
                            kind,
                            id: ast::DUMMY_NODE_ID,
                            span: self.span,
                            is_placeholder: false,
                        };
                        where_clause.predicates.push(predicate);
                    }
                }
            }
        }

        let trait_generics = Generics { params, where_clause, span: generics.span.with_ctxt(ctxt) };

        // Create the reference to the trait.
        let trait_ref = cx.trait_ref(self.path.clone());

        let self_params: Vec<_> = generics
            .params
            .iter()
            .map(|param| generic_param_to_arg(cx, param, param.ident.span.with_ctxt(ctxt)))
            .collect();

        // Create the type of `self`.
        let path =
            cx.path_all(type_ident.span.with_ctxt(ctxt), false, vec![type_ident], self_params);
        let self_type = cx.ty_path(path);

        let mut attrs = thin_vec![cx.attr_word(sym::automatically_derived, self.span),];

        // Only add `rustc_const_unstable` attributes if `derive_const` is used within libcore/libstd,
        // Other crates don't need stability attributes, so adding them is not useful, but libcore needs them
        // on all const trait impls.
        if self.is_const && cx.ecfg.features.staged_api() {
            let rustc_const_unstable =
                cx.path_ident(self.span, Ident::new(sym::rustc_const_unstable, self.span));

            // #[rustc_const_unstable(feature = "derive_const", issue = "118304")]
            attrs.push(
                cx.attr_nested(
                    rustc_ast::AttrItem {
                        unsafety: Safety::Default,
                        path: rustc_const_unstable,
                        args: AttrArgs::Delimited(DelimArgs {
                            dspan: DelimSpan::from_single(self.span),
                            delim: rustc_ast::token::Delimiter::Parenthesis,
                            tokens: [
                                TokenKind::Ident(sym::feature, IdentKind::Normal),
                                TokenKind::Eq,
                                TokenKind::lit(LitKind::Str, sym::derive_const, None),
                                TokenKind::Comma,
                                TokenKind::Ident(sym::issue, IdentKind::Normal),
                                TokenKind::Eq,
                                TokenKind::lit(LitKind::Str, sym::derive_const_issue, None),
                            ]
                            .into_iter()
                            .map(|kind| {
                                TokenTree::Token(Token { kind, span: self.span }, Spacing::Alone)
                            })
                            .collect(),
                        }),
                        span: self.span,
                    },
                    self.span,
                ),
            )
        }

        if !self.document {
            attrs.push(cx.attr_nested_word(sym::doc, sym::hidden, self.span));
        }

        cx.item_trait_impl(
            self.span,
            attrs,
            trait_generics,
            self.safety,
            self.is_const,
            trait_ref,
            self_type,
            methods.collect(),
        )
    }
}

impl<'a> MethodDef<'a> {
    fn call_substructure_method(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        substructure: Substructure<'_>,
    ) -> BlockOrExpr {
        (self.combine_substructure)(cx, span, substructure)
    }

    fn is_static(&self) -> bool {
        !self.explicit_self
    }

    /// Expressions for `&self` and also any other
    /// args with the same type (e.g. the `other` arg in `PartialEq::eq`).
    fn get_selflike_args(&self, cx: &ExtCtxt<'_>, span: Span) -> ThinVec<Box<Expr>> {
        assert!(self.explicit_self);

        let self_expr = cx.expr_self(span);
        if self.has_other_selflike_arg {
            thin_vec![self_expr, cx.expr_ident_sym(span, self.nonself_args[0].1)]
        } else {
            thin_vec![self_expr]
        }
    }

    fn create_method(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        body: BlockOrExpr,
    ) -> Option<Box<ast::AssocItem>> {
        // `assert_fields_are_eq` has an empty default implementation
        if body.0.is_empty() && body.1.is_none() && self.name == sym::assert_fields_are_eq {
            return None;
        }
        // Create the generics that aren't for `Self`.
        let fn_generics = self.generics.clone();

        let self_arg = self.explicit_self.then(|| {
            let ident = Ident::new(kw::SelfLower, span);
            ast::Param::from_self(
                ast::AttrVec::default(),
                respan(span, SelfKind::Region(None, ast::Mutability::Not)),
                ident,
            )
        });
        let args = self_arg
            .into_iter()
            .chain(self.nonself_args.iter().map(|(ty, name)| {
                let ast_ty = ty.clone();
                let ident = Ident::new(*name, span);
                cx.param(span, ident, ast_ty)
            }))
            .collect();

        let ret_type = if self.ret_ty.kind.is_unit() {
            ast::FnRetTy::Default(span)
        } else {
            ast::FnRetTy::Ty(self.ret_ty.clone())
        };

        let method_ident = Ident::new(self.name, span);
        let fn_decl = cx.fn_decl(args, ret_type);
        let body_block = body.into_block(cx, span);

        let trait_lo_sp = span.shrink_to_lo();

        let sig = ast::FnSig { header: ast::FnHeader::default(), decl: fn_decl, span };
        let defaultness = ast::Defaultness::Implicit;

        // Create the method.
        Some(Box::new(ast::AssocItem {
            id: ast::DUMMY_NODE_ID,
            attrs: self.attributes.clone(),
            span,
            vis: ast::Visibility { span: trait_lo_sp, kind: ast::VisibilityKind::Inherited },
            kind: ast::AssocItemKind::Fn(Box::new(ast::Fn {
                defaultness,
                sig,
                ident: method_ident,
                generics: fn_generics,
                contract: None,
                body: Some(body_block),
                define_opaque: None,
                eii_impl: None,
            })),
            tokens: None,
        }))
    }

    /// The normal case uses field access.
    ///
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
    ///
    /// But if the struct is `repr(packed)`, we can't use something like
    /// `&self.x` because that might cause an unaligned ref. So for any trait
    /// method that takes a reference, we use a local block to force a copy.
    /// This requires that the field impl `Copy`.
    ///
    /// ```rust,ignore (example)
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
    ///         ::core::hash::Hash::hash(&{ self.y }, state);
    ///     }
    /// }
    /// ```
    fn expand_struct_method_body<'b>(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        struct_def: &'b VariantData,
        is_packed: bool,
    ) -> BlockOrExpr {
        let selflike_args = self.get_selflike_args(cx, span);

        let selflike_fields =
            create_struct_field_access_fields(span, cx, &selflike_args, struct_def, is_packed);
        self.call_substructure_method(cx, span, Struct(struct_def, selflike_fields))
    }

    /// ```
    /// #[derive(PartialEq)]
    /// # struct Dummy;
    /// enum A {
    ///     A1,
    ///     A2(i32)
    /// }
    /// ```
    ///
    /// is equivalent to:
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    /// enum A {
    ///     A1,
    ///     A2(i32)
    /// }
    /// impl ::core::cmp::PartialEq for A {
    ///     #[inline]
    ///     fn eq(&self, other: &A) -> bool {
    ///         let __self_discr = ::core::intrinsics::discriminant_value(self);
    ///         let __arg1_discr = ::core::intrinsics::discriminant_value(other);
    ///         __self_discr == __arg1_discr
    ///             && match (self, other) {
    ///                 (A::A2(__self_0), A::A2(__arg1_0)) => *__self_0 == *__arg1_0,
    ///                 _ => true,
    ///             }
    ///     }
    /// }
    /// ```
    ///
    /// Creates a discriminant check combined with a match for a tuple of all
    /// `selflike_args`, with an arm for each variant with fields, possibly an
    /// arm for each fieldless variant (if `unify_fieldless_variants` is not
    /// `Unify`), and possibly a default arm.
    fn expand_enum_method_body<'b>(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        enum_def: &'b EnumDef,
    ) -> BlockOrExpr {
        let variants = &enum_def.variants;

        // Traits that unify fieldless variants always use the discriminant(s).
        let unify_fieldless_variants =
            self.fieldless_variants_strategy == FieldlessVariantsStrategy::Unify;

        // For zero-variant enum, this function body is unreachable. Generate
        // `match *self {}`. This produces machine code identical to `unsafe {
        // core::intrinsics::unreachable() }` while being safe and stable.
        if variants.is_empty() {
            let match_arg = cx.expr_deref(span, cx.expr_self(span));
            let match_arms = ThinVec::new();
            let expr = cx.expr_match(span, match_arg, match_arms);
            return BlockOrExpr(ThinVec::new(), Some(expr));
        }

        let selflike_args = self.get_selflike_args(cx, span);

        let prefixes: &[&str] = match selflike_args.len() {
            1 => &["__self"],
            2 => &["__self", "__arg1"],
            _ => unreachable!(),
        };

        // There are some special cases involving fieldless enums where no
        // match is necessary.
        let all_fieldless = variants.iter().all(|v| v.data.fields().is_empty());
        if all_fieldless {
            if variants.len() > 1 {
                match self.fieldless_variants_strategy {
                    FieldlessVariantsStrategy::Unify => {
                        // If the type is fieldless and the trait uses the discriminant and
                        // there are multiple variants, we need just an operation on
                        // the discriminant(s).
                        return self.call_substructure_method(cx, span, EnumDiscr(None));
                    }
                    FieldlessVariantsStrategy::SpecializeIfAllVariantsFieldless => {
                        return self.call_substructure_method(cx, span, AllFieldlessEnum(enum_def));
                    }
                    FieldlessVariantsStrategy::Default => (),
                }
            } else if let [variant] = variants.as_slice() {
                // If there is a single variant, we don't need an operation on
                // the discriminant(s). Just use the most degenerate result.
                return self.call_substructure_method(cx, span, EnumMatching(variant, Vec::new()));
            }
        }

        // These arms are of the form:
        // (Variant1, Variant1, ...) => Body1
        // (Variant2, Variant2, ...) => Body2
        // ...
        // where each tuple has length = selflike_args.len()
        let mut match_arms: ThinVec<ast::Arm> = variants
            .iter()
            .filter(|&v| !(unify_fieldless_variants && v.data.fields().is_empty()))
            .map(|variant| {
                // A single arm has form (&VariantK, &VariantK, ...) => BodyK
                // (see "Final wrinkle" note below for why.)

                let fields = create_struct_pattern_fields(span, cx, &variant.data, &prefixes);

                let sp = variant.span.with_ctxt(span.ctxt());
                let variant_path =
                    cx.path(sp, vec![Ident::new(kw::SelfUpper, span), variant.ident]);
                let mut subpats =
                    create_struct_patterns(span, cx, variant_path, &variant.data, &prefixes);

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
                let substructure = EnumMatching(variant, fields);
                let arm_expr =
                    self.call_substructure_method(cx, span, substructure).into_expr(cx, span);

                cx.arm(span, single_pat, arm_expr)
            })
            .collect();

        // Add a default arm to the match, if necessary.
        let first_fieldless = variants.iter().find(|v| v.data.fields().is_empty());
        let default = match first_fieldless {
            Some(v) if unify_fieldless_variants => {
                // We need a default case that handles all the fieldless variants.
                Some(
                    self.call_substructure_method(cx, span, EnumMatching(v, Vec::new()))
                        .into_expr(cx, span),
                )
            }
            _ if variants.len() > 1 && selflike_args.len() > 1 => {
                // Because we know that all the arguments will match if we reach
                // the match expression we add the unreachable intrinsic as the
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
        //          _ => ::core::intrinsics::unreachable(),
        //      }
        let get_match_expr = |mut selflike_args: ThinVec<Box<Expr>>| {
            let match_arg = if selflike_args.len() == 1 {
                selflike_args.pop().unwrap()
            } else {
                cx.expr_tuple(span, selflike_args)
            };
            cx.expr_match(span, match_arg, match_arms)
        };

        // If the trait uses the discriminant and there are multiple variants, we need
        // to add a discriminant check operation before the match. Otherwise, the match
        // is enough.
        if unify_fieldless_variants && variants.len() > 1 {
            // Combine a discriminant check with the match.
            self.call_substructure_method(cx, span, EnumDiscr(Some(get_match_expr(selflike_args))))
        } else {
            BlockOrExpr(ThinVec::new(), Some(get_match_expr(selflike_args)))
        }
    }
}

// general helper methods.
fn create_struct_patterns(
    span: Span,
    cx: &ExtCtxt<'_>,
    struct_path: ast::Path,
    struct_def: &VariantData,
    prefixes: &[&str],
) -> ThinVec<ast::Pat> {
    prefixes
        .iter()
        .map(|prefix| {
            let pieces_iter = struct_def.fields().iter().enumerate().map(|(i, struct_field)| {
                let ident = mk_pattern_ident(span, prefix, i);
                let path = ident.with_span_pos(struct_field.span);
                (struct_field.ident, cx.pat_ident(path.span, path))
            });

            let struct_path = struct_path.clone();
            match *struct_def {
                VariantData::Struct { .. } => {
                    let field_pats = pieces_iter
                        .map(|(ident, pat)| ast::PatField {
                            ident: ident.expect("a braced struct with unnamed fields in `derive`"),
                            is_shorthand: false,
                            attrs: ast::AttrVec::new(),
                            id: ast::DUMMY_NODE_ID,
                            span: pat.span.with_ctxt(span.ctxt()),
                            pat: Box::new(pat),
                            is_placeholder: false,
                        })
                        .collect();
                    cx.pat_struct(span, struct_path, field_pats)
                }
                VariantData::Tuple(..) => {
                    let subpats = pieces_iter.map(|(_, subpat)| subpat).collect();
                    cx.pat_tuple_struct(span, struct_path, subpats)
                }
                VariantData::Unit(..) => cx.pat_path(span, struct_path),
            }
        })
        .collect()
}

fn create_fields<F>(span: Span, struct_def: &VariantData, mk_exprs: F) -> Vec<FieldInfo>
where
    F: Fn(usize, &ast::FieldDef, Span) -> Vec<Box<Expr>>,
{
    struct_def
        .fields()
        .iter()
        .enumerate()
        .map(|(i, struct_field)| {
            // For this field, get an expr for each selflike_arg. E.g. for
            // `PartialEq::eq`, one for each of `&self` and `other`.
            let span = struct_field.span.with_ctxt(span.ctxt());
            let mut exprs: Vec<_> = mk_exprs(i, struct_field, span);
            let self_expr = exprs.remove(0);
            debug_assert!(exprs.len() <= 1);
            FieldInfo {
                span,
                name: struct_field.ident,
                self_expr,
                other_selflike_expr: exprs.pop(),
                maybe_scalar: struct_field.ty.peel_refs().kind.maybe_scalar(),
            }
        })
        .collect()
}

fn mk_pattern_ident(span: Span, prefix: &str, i: usize) -> Ident {
    Ident::from_str_and_span(&format!("{prefix}_{i}"), span)
}

fn create_struct_pattern_fields(
    span: Span,
    cx: &ExtCtxt<'_>,
    struct_def: &VariantData,
    prefixes: &[&str],
) -> Vec<FieldInfo> {
    create_fields(span, struct_def, |i, _struct_field, sp| {
        prefixes
            .iter()
            .map(|prefix| {
                let ident = mk_pattern_ident(span, prefix, i);
                cx.expr_path(cx.path_ident(sp, ident))
            })
            .collect()
    })
}

fn create_struct_field_access_fields(
    span: Span,
    cx: &ExtCtxt<'_>,
    selflike_args: &[Box<Expr>],
    struct_def: &VariantData,
    is_packed: bool,
) -> Vec<FieldInfo> {
    create_fields(span, struct_def, |i, struct_field, sp| {
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
                if is_packed {
                    // Fields in packed structs are wrapped in a block, e.g. `&{self.0}`,
                    // causing a copy instead of a (potentially misaligned) reference.
                    field_expr = cx.expr_block(
                        cx.block(struct_field.span, thin_vec![cx.stmt_expr(field_expr)]),
                    );
                }
                cx.expr_addr_of(sp, field_expr)
            })
            .collect()
    })
}

pub(crate) fn generic_param_to_arg(cx: &ExtCtxt<'_>, p: &GenericParam, span: Span) -> GenericArg {
    match p.kind {
        GenericParamKind::Lifetime => GenericArg::Lifetime(cx.lifetime(span, p.ident)),
        GenericParamKind::Type { .. } => GenericArg::Type(cx.ty_ident(span, p.ident)),
        GenericParamKind::Const { .. } => GenericArg::Const(cx.const_ident(span, p.ident)),
    }
}

pub(crate) fn generics_without_defaults(g: &Generics) -> Generics {
    Generics {
        params: g
            .params
            .iter()
            .map(|p| {
                let mut p = p.clone();
                match &mut p.kind {
                    ast::GenericParamKind::Const { default, .. } => *default = None,
                    ast::GenericParamKind::Type { default } => *default = None,
                    ast::GenericParamKind::Lifetime => {}
                };
                p.attrs.clear();
                p
            })
            .collect(),
        where_clause: g.where_clause.clone(),
        span: g.span,
    }
}
