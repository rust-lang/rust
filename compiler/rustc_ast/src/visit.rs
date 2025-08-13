//! AST walker. Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `visit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! Note: it is an important invariant that the default visitor walks the body
//! of a function in "execution order" (more concretely, reverse post-order
//! with respect to the CFG implied by the AST), meaning that if AST node A may
//! execute before AST node B, then A is visited first. The borrow checker in
//! particular relies on this property.
//!
//! Note: walking an AST before macro expansion is probably a bad idea. For
//! instance, a walker looking for item names in a module will miss all of
//! those that are created by the expansion of a macro.

pub use rustc_ast_ir::visit::VisitorResult;
pub use rustc_ast_ir::{try_visit, visit_opt, walk_list, walk_visitable_list};
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::tokenstream::DelimSpan;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssocCtxt {
    Trait,
    Impl { of_trait: bool },
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FnCtxt {
    Free,
    Foreign,
    Assoc(AssocCtxt),
}

#[derive(Copy, Clone, Debug)]
pub enum BoundKind {
    /// Trait bounds in generics bounds and type/trait alias.
    /// E.g., `<T: Bound>`, `type A: Bound`, or `where T: Bound`.
    Bound,

    /// Trait bounds in `impl` type.
    /// E.g., `type Foo = impl Bound1 + Bound2 + Bound3`.
    Impl,

    /// Trait bounds in trait object type.
    /// E.g., `dyn Bound1 + Bound2 + Bound3`.
    TraitObject,

    /// Super traits of a trait.
    /// E.g., `trait A: B`
    SuperTraits,
}
impl BoundKind {
    pub fn descr(self) -> &'static str {
        match self {
            BoundKind::Bound => "bounds",
            BoundKind::Impl => "`impl Trait`",
            BoundKind::TraitObject => "`dyn` trait object bounds",
            BoundKind::SuperTraits => "supertrait bounds",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LifetimeCtxt {
    /// Appears in a reference type.
    Ref,
    /// Appears as a bound on a type or another lifetime.
    Bound,
    /// Appears as a generic argument.
    GenericArg,
}

pub(crate) trait Visitable<'a, V: Visitor<'a>> {
    type Extra: Copy;

    #[must_use]
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result;
}

impl<'a, V: Visitor<'a>, T: ?Sized> Visitable<'a, V> for Box<T>
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        (**self).visit(visitor, extra)
    }
}

impl<'a, V: Visitor<'a>, T> Visitable<'a, V> for Option<T>
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        if let Some(this) = self {
            try_visit!(this.visit(visitor, extra));
        }
        V::Result::output()
    }
}

impl<'a, V: Visitor<'a>, T> Visitable<'a, V> for Spanned<T>
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        let Spanned { span: _, node } = self;
        node.visit(visitor, extra)
    }
}

impl<'a, V: Visitor<'a>, T> Visitable<'a, V> for [T]
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        for item in self {
            try_visit!(item.visit(visitor, extra));
        }
        V::Result::output()
    }
}

impl<'a, V: Visitor<'a>, T> Visitable<'a, V> for Vec<T>
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        for item in self {
            try_visit!(item.visit(visitor, extra));
        }
        V::Result::output()
    }
}

impl<'a, V: Visitor<'a>, T> Visitable<'a, V> for (T,)
where
    T: Visitable<'a, V>,
{
    type Extra = T::Extra;
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        self.0.visit(visitor, extra)
    }
}

impl<'a, V: Visitor<'a>, T1, T2> Visitable<'a, V> for (T1, T2)
where
    T1: Visitable<'a, V, Extra = ()>,
    T2: Visitable<'a, V, Extra = ()>,
{
    type Extra = ();
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        try_visit!(self.0.visit(visitor, extra));
        try_visit!(self.1.visit(visitor, extra));
        V::Result::output()
    }
}

impl<'a, V: Visitor<'a>, T1, T2, T3> Visitable<'a, V> for (T1, T2, T3)
where
    T1: Visitable<'a, V, Extra = ()>,
    T2: Visitable<'a, V, Extra = ()>,
    T3: Visitable<'a, V, Extra = ()>,
{
    type Extra = ();
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        try_visit!(self.0.visit(visitor, extra));
        try_visit!(self.1.visit(visitor, extra));
        try_visit!(self.2.visit(visitor, extra));
        V::Result::output()
    }
}

impl<'a, V: Visitor<'a>, T1, T2, T3, T4> Visitable<'a, V> for (T1, T2, T3, T4)
where
    T1: Visitable<'a, V, Extra = ()>,
    T2: Visitable<'a, V, Extra = ()>,
    T3: Visitable<'a, V, Extra = ()>,
    T4: Visitable<'a, V, Extra = ()>,
{
    type Extra = ();
    fn visit(&'a self, visitor: &mut V, extra: Self::Extra) -> V::Result {
        try_visit!(self.0.visit(visitor, extra));
        try_visit!(self.1.visit(visitor, extra));
        try_visit!(self.2.visit(visitor, extra));
        try_visit!(self.3.visit(visitor, extra));
        V::Result::output()
    }
}

pub(crate) trait Walkable<'a, V: Visitor<'a>> {
    #[must_use]
    fn walk_ref(&'a self, visitor: &mut V) -> V::Result;
}

macro_rules! visit_visitable {
    ($visitor:expr, $($expr:expr),* $(,)?) => {{
        $(try_visit!(Visitable::visit($expr, $visitor, ()));)*
    }};
}

macro_rules! visit_visitable_with {
    ($visitor:expr, $expr:expr, $extra:expr $(,)?) => {
        try_visit!(Visitable::visit($expr, $visitor, $extra))
    };
}

macro_rules! walk_walkable {
    ($visitor:expr, $expr:expr, ) => {
        Walkable::walk_ref($expr, $visitor)
    };
}

macro_rules! impl_visitable {
    (|&$lt:lifetime $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident,
      $extra:ident: $extra_ty:ty| $block:block) => {
        #[allow(unused_parens, non_local_definitions)]
        impl<$lt, $vis_ty: Visitor<$lt>> Visitable<$lt, $vis_ty> for $self_ty {
            type Extra = $extra_ty;
            fn visit(&$lt $self, $vis: &mut $vis_ty, $extra: Self::Extra) -> V::Result {
                $block
            }
        }
    };
}

macro_rules! impl_walkable {
    ($(<$K:ident: $Kb:ident>)? |&$lt:lifetime $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident| $block:block) => {
        #[allow(unused_parens, non_local_definitions)]
        impl<$($K: $Kb,)? $lt, $vis_ty: Visitor<$lt>> Walkable<$lt, $vis_ty> for $self_ty {
            fn walk_ref(&$lt $self, $vis: &mut $vis_ty) -> V::Result {
                $block
            }
        }
    };
}

macro_rules! impl_visitable_noop {
    (<$lt:lifetime> $($ty:ty,)*) => {
        $(
            impl_visitable!(|&$lt self: $ty, _vis: &mut V, _extra: ()| {
                V::Result::output()
            });
        )*
    };
}

macro_rules! impl_visitable_list {
    (<$lt:lifetime> $($ty:ty,)*) => {
        $(impl<$lt, V: Visitor<$lt>, T> Visitable<$lt, V> for $ty
        where
            &$lt $ty: IntoIterator<Item = &$lt T>,
            T: $lt + Visitable<$lt, V>,
        {
            type Extra = <T as Visitable<$lt, V>>::Extra;

            #[inline]
            fn visit(&$lt self, visitor: &mut V, extra: Self::Extra) -> V::Result {
                for i in self {
                    try_visit!(i.visit(visitor, extra));
                }
                V::Result::output()
            }
        })*
    };
}

macro_rules! impl_visitable_direct {
    (<$lt:lifetime> $($ty:ty,)*) => {
        $(impl_visitable!(
            |&$lt self: $ty, visitor: &mut V, _extra: ()| {
                Walkable::walk_ref(self, visitor)
            }
        );)*
    };
}

macro_rules! impl_visitable_calling_walkable {
    (<$lt:lifetime>
        $( fn $method:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?); )*
    ) => {
        $(fn $method(&mut self, node: &$lt $ty $(, $extra_name:$extra_ty)?) -> Self::Result {
            impl_visitable!(|&$lt self: $ty, visitor: &mut V, extra: ($($extra_ty)?)| {
                let ($($extra_name)?) = extra;
                visitor.$method(self $(, $extra_name)?)
            });
            walk_walkable!(self, node, )
        })*
    };
}

macro_rules! define_named_walk {
    ($Visitor:ident<$lt:lifetime>
        $( pub fn $method:ident($ty:ty); )*
    ) => {
        $(pub fn $method<$lt, V: $Visitor<$lt>>(visitor: &mut V, node: &$lt $ty) -> V::Result {
            walk_walkable!(visitor, node,)
        })*
    };
}

#[macro_export]
macro_rules! common_visitor_and_walkers {
    ($(($mut: ident))? $Visitor:ident$(<$lt:lifetime>)?) => {
        $(${ignore($lt)}
            #[derive(Copy, Clone)]
        )?
        #[derive(Debug)]
        pub enum FnKind<'a> {
            /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
            Fn(FnCtxt, &'a $($mut)? Visibility, &'a $($mut)? Fn),

            /// E.g., `|x, y| body`.
            Closure(&'a $($mut)? ClosureBinder, &'a $($mut)? Option<CoroutineKind>, &'a $($mut)? Box<FnDecl>, &'a $($mut)? Box<Expr>),
        }

        impl<'a> FnKind<'a> {
            pub fn header(&'a $($mut)? self) -> Option<&'a $($mut)? FnHeader> {
                match *self {
                    FnKind::Fn(_, _, Fn { sig, .. }) => Some(&$($mut)? sig.header),
                    FnKind::Closure(..) => None,
                }
            }

            pub fn ident(&'a $($mut)? self) -> Option<&'a $($mut)? Ident> {
                match self {
                    FnKind::Fn(_, _, Fn { ident, .. }) => Some(ident),
                    _ => None,
                }
            }

            pub fn decl(&'a $($mut)? self) -> &'a $($mut)? FnDecl {
                match self {
                    FnKind::Fn(_, _, Fn { sig, .. }) => &$($mut)? sig.decl,
                    FnKind::Closure(_, _, decl, _) => decl,
                }
            }

            pub fn ctxt(&self) -> Option<FnCtxt> {
                match self {
                    FnKind::Fn(ctxt, ..) => Some(*ctxt),
                    FnKind::Closure(..) => None,
                }
            }
        }

        // This macro generates `impl Visitable` and `impl MutVisitable` that do nothing.
        impl_visitable_noop!(<$($lt)? $($mut)?>
            AttrId,
            bool,
            rustc_span::ByteSymbol,
            char,
            crate::token::CommentKind,
            crate::token::Delimiter,
            crate::token::Lit,
            crate::token::LitKind,
            crate::tokenstream::LazyAttrTokenStream,
            crate::tokenstream::TokenStream,
            Movability,
            Mutability,
            Result<(), rustc_span::ErrorGuaranteed>,
            rustc_data_structures::fx::FxHashMap<Symbol, usize>,
            rustc_span::ErrorGuaranteed,
            std::borrow::Cow<'_, str>,
            Symbol,
            u8,
            usize,
        );
        // `Span` is only a no-op for the non-mutable visitor.
        $(impl_visitable_noop!(<$lt> Span,);)?

        // This macro generates `impl Visitable` and `impl MutVisitable` that simply iterate over
        // their contents. We do not use a generic impl for `ThinVec` because we want to allow
        // custom visits for the `MutVisitor`.
        impl_visitable_list!(<$($lt)? $($mut)?>
            ThinVec<AngleBracketedArg>,
            ThinVec<Attribute>,
            ThinVec<(Ident, Option<Ident>)>,
            ThinVec<(NodeId, Path)>,
            ThinVec<PathSegment>,
            ThinVec<PreciseCapturingArg>,
            ThinVec<Box<Pat>>,
            ThinVec<Box<Ty>>,
            ThinVec<Box<TyPat>>,
        );

        // This macro generates `impl Visitable` and `impl MutVisitable` that forward to `Walkable`
        // or `MutWalkable`. By default, all types that do not have a custom visit method in the
        // visitor should appear here.
        impl_visitable_direct!(<$($lt)? $($mut)?>
            AngleBracketedArg,
            AngleBracketedArgs,
            AsmMacro,
            AssignOpKind,
            AssocItemConstraintKind,
            AttrArgs,
            AttrItem,
            AttrKind,
            AttrStyle,
            FnPtrTy,
            BindingMode,
            GenBlockKind,
            RangeLimits,
            UnsafeBinderCastKind,
            BinOpKind,
            BlockCheckMode,
            BorrowKind,
            BoundAsyncness,
            BoundConstness,
            BoundPolarity,
            ByRef,
            Closure,
            Const,
            ConstItem,
            Defaultness,
            Delegation,
            DelegationMac,
            DelimArgs,
            DelimSpan,
            EnumDef,
            Extern,
            ForLoopKind,
            FormatArgPosition,
            FormatArgsPiece,
            FormatArgument,
            FormatArgumentKind,
            FormatArguments,
            FormatPlaceholder,
            GenericParamKind,
            Impl,
            ImplPolarity,
            Inline,
            InlineAsmOperand,
            InlineAsmRegOrRegClass,
            InlineAsmTemplatePiece,
            IsAuto,
            LocalKind,
            MacCallStmt,
            MacStmtStyle,
            MatchKind,
            MethodCall,
            ModKind,
            ModSpans,
            MutTy,
            NormalAttr,
            Parens,
            ParenthesizedArgs,
            PatFieldsRest,
            PatKind,
            RangeEnd,
            RangeSyntax,
            Recovered,
            Safety,
            StaticItem,
            StrLit,
            StrStyle,
            StructExpr,
            StructRest,
            Term,
            Trait,
            TraitBoundModifiers,
            TraitObjectSyntax,
            TyAlias,
            TyAliasWhereClause,
            TyAliasWhereClauses,
            TyKind,
            TyPatKind,
            UnOp,
            UnsafeBinderTy,
            UnsafeSource,
            UseTreeKind,
            VisibilityKind,
            WhereBoundPredicate,
            WhereClause,
            WhereEqPredicate,
            WhereRegionPredicate,
            YieldKind,
        );

        /// Each method of this trait is a hook to be potentially
        /// overridden. Each method's default implementation recursively visits
        /// the substructure of the input via the corresponding `walk` method;
        #[doc = concat!(" e.g., the `visit_item` method by default calls `visit"$(, "_", stringify!($mut))?, "::walk_item`.")]
        ///
        /// If you want to ensure that your code handles every variant
        /// explicitly, you need to override each method. (And you also need
        /// to monitor future changes to this trait in case a new method with a
        /// new default implementation gets introduced.)
        ///
        /// Every `walk_*` method uses deconstruction to access fields of structs and
        /// enums. This will result in a compile error if a field is added, which makes
        /// it more likely the appropriate visit call will be added for it.
        pub trait $Visitor<$($lt)?> : Sized $(${ignore($mut)} + MutVisitorResult<Result = ()>)? {
            $(
                ${ignore($lt)}
                /// The result type of the `visit_*` methods. Can be either `()`,
                /// or `ControlFlow<T>`.
                type Result: VisitorResult = ();
            )?

            // Methods in this trait have one of three forms, with the last two forms
            // only occurring on `MutVisitor`:
            //
            //   fn visit_t(&mut self, t: &mut T);                      // common
            //   fn flat_map_t(&mut self, t: T) -> SmallVec<[T; 1]>;    // rare
            //   fn filter_map_t(&mut self, t: T) -> Option<T>;         // rarest
            //
            // When writing these methods, it is better to use destructuring like this:
            //
            //   fn visit_abc(&mut self, ABC { a, b, c: _ }: &mut ABC) {
            //       visit_a(a);
            //       visit_b(b);
            //   }
            //
            // than to use field access like this:
            //
            //   fn visit_abc(&mut self, abc: &mut ABC) {
            //       visit_a(&mut abc.a);
            //       visit_b(&mut abc.b);
            //       // ignore abc.c
            //   }
            //
            // As well as being more concise, the former is explicit about which fields
            // are skipped. Furthermore, if a new field is added, the destructuring
            // version will cause a compile error, which is good. In comparison, the
            // field access version will continue working and it would be easy to
            // forget to add handling for it.
            fn visit_ident(&mut self, Ident { name: _, span }: &$($lt)? $($mut)? Ident) -> Self::Result {
                impl_visitable!(|&$($lt)? $($mut)? self: Ident, visitor: &mut V, _extra: ()| {
                    visitor.visit_ident(self)
                });
                visit_span(self, span)
            }

            // This macro defines a custom visit method for each listed type.
            // It implements `impl Visitable` and `impl MutVisitable` to call those methods on the
            // visitor.
            impl_visitable_calling_walkable!(<$($lt)? $($mut)?>
                fn visit_anon_const(AnonConst);
                fn visit_arm(Arm);
                //fn visit_assoc_item(AssocItem, _ctxt: AssocCtxt);
                fn visit_assoc_item_constraint(AssocItemConstraint);
                fn visit_attribute(Attribute);
                fn visit_block(Block);
                //fn visit_nested_use_tree((UseTree, NodeId));
                fn visit_capture_by(CaptureBy);
                fn visit_closure_binder(ClosureBinder);
                fn visit_contract(FnContract);
                fn visit_coroutine_kind(CoroutineKind);
                fn visit_crate(Crate);
                fn visit_expr(Expr);
                fn visit_expr_field(ExprField);
                fn visit_field_def(FieldDef);
                fn visit_fn_decl(FnDecl);
                fn visit_fn_header(FnHeader);
                fn visit_fn_ret_ty(FnRetTy);
                //fn visit_foreign_item(ForeignItem);
                fn visit_foreign_mod(ForeignMod);
                fn visit_format_args(FormatArgs);
                fn visit_generic_arg(GenericArg);
                fn visit_generic_args(GenericArgs);
                fn visit_generic_param(GenericParam);
                fn visit_generics(Generics);
                fn visit_inline_asm(InlineAsm);
                fn visit_inline_asm_sym(InlineAsmSym);
                //fn visit_item(Item);
                fn visit_label(Label);
                fn visit_lifetime(Lifetime, _ctxt: LifetimeCtxt);
                fn visit_local(Local);
                fn visit_mac_call(MacCall);
                fn visit_macro_def(MacroDef);
                fn visit_param_bound(GenericBound, _ctxt: BoundKind);
                fn visit_param(Param);
                fn visit_pat_field(PatField);
                fn visit_path(Path);
                fn visit_path_segment(PathSegment);
                fn visit_pat(Pat);
                fn visit_poly_trait_ref(PolyTraitRef);
                fn visit_precise_capturing_arg(PreciseCapturingArg);
                fn visit_qself(QSelf);
                fn visit_trait_ref(TraitRef);
                fn visit_ty_pat(TyPat);
                fn visit_ty(Ty);
                fn visit_use_tree(UseTree);
                fn visit_variant_data(VariantData);
                fn visit_variant(Variant);
                fn visit_vis(Visibility);
                fn visit_where_predicate_kind(WherePredicateKind);
                fn visit_where_predicate(WherePredicate);
            );

            // We want `Visitor` to take the `NodeId` by value.
            fn visit_id(&mut self, _id: $(&$mut)? NodeId) -> Self::Result {
                $(impl_visitable!(
                    |&$lt self: NodeId, visitor: &mut V, _extra: ()| {
                        visitor.visit_id(*self)
                    }
                );)?
                $(impl_visitable!(
                    |&$mut self: NodeId, visitor: &mut V, _extra: ()| {
                        visitor.visit_id(self)
                    }
                );)?
                Self::Result::output()
            }

            /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
            /// It can be removed once that feature is stabilized.
            fn visit_method_receiver_expr(&mut self, ex: &$($lt)? $($mut)? Expr) -> Self::Result {
                self.visit_expr(ex)
            }

            fn visit_item(&mut self, item: &$($lt)? $($mut)? Item) -> Self::Result {
                impl_visitable!(|&$($lt)? $($mut)? self: Item, vis: &mut V, _extra: ()| {
                    vis.visit_item(self)
                });
                walk_item(self, item)
            }

            fn visit_foreign_item(&mut self, item: &$($lt)? $($mut)? ForeignItem) -> Self::Result {
                impl_visitable!(|&$($lt)? $($mut)? self: ForeignItem, vis: &mut V, _extra: ()| {
                    vis.visit_foreign_item(self)
                });
                walk_item(self, item)
            }

            fn visit_assoc_item(&mut self, item: &$($lt)? $($mut)? AssocItem, ctxt: AssocCtxt) -> Self::Result {
                impl_visitable!(|&$($lt)? $($mut)? self: AssocItem, vis: &mut V, ctxt: AssocCtxt| {
                    vis.visit_assoc_item(self, ctxt)
                });
                walk_assoc_item(self, item, ctxt)
            }

            // for `MutVisitor`: `Span` and `NodeId` are mutated at the caller site.
            fn visit_fn(
                &mut self,
                fk: FnKind<$($lt)? $(${ignore($mut)} '_)?>,
                _: Span,
                _: NodeId
            ) -> Self::Result {
                walk_fn(self, fk)
            }

            // (non-mut) `Visitor`-only methods
            $(
                fn visit_stmt(&mut self, s: &$lt Stmt) -> Self::Result {
                    walk_stmt(self, s)
                }

                fn visit_nested_use_tree(&mut self, use_tree: &$lt UseTree, id: NodeId) -> Self::Result {
                    try_visit!(self.visit_id(id));
                    self.visit_use_tree(use_tree)
                }
            )?

            // `MutVisitor`-only methods
            $(
                // Span visiting is no longer used, but we keep it for now,
                // in case it's needed for something like #127241.
                #[inline]
                fn visit_span(&mut self, _sp: &$mut Span) {
                    impl_visitable!(|&mut self: Span, visitor: &mut V, _extra: ()| {
                        visitor.visit_span(self)
                    });
                    // Do nothing.
                }

                fn flat_map_foreign_item(&mut self, ni: Box<ForeignItem>) -> SmallVec<[Box<ForeignItem>; 1]> {
                    walk_flat_map_foreign_item(self, ni)
                }

                fn flat_map_item(&mut self, i: Box<Item>) -> SmallVec<[Box<Item>; 1]> {
                    walk_flat_map_item(self, i)
                }

                fn flat_map_field_def(&mut self, fd: FieldDef) -> SmallVec<[FieldDef; 1]> {
                    walk_flat_map_field_def(self, fd)
                }

                fn flat_map_assoc_item(
                    &mut self,
                    i: Box<AssocItem>,
                    ctxt: AssocCtxt,
                ) -> SmallVec<[Box<AssocItem>; 1]> {
                    walk_flat_map_assoc_item(self, i, ctxt)
                }

                fn flat_map_stmt(&mut self, s: Stmt) -> SmallVec<[Stmt; 1]> {
                    walk_flat_map_stmt(self, s)
                }

                fn flat_map_arm(&mut self, arm: Arm) -> SmallVec<[Arm; 1]> {
                    walk_flat_map_arm(self, arm)
                }

                fn filter_map_expr(&mut self, e: Box<Expr>) -> Option<Box<Expr>> {
                    walk_filter_map_expr(self, e)
                }

                fn flat_map_variant(&mut self, v: Variant) -> SmallVec<[Variant; 1]> {
                    walk_flat_map_variant(self, v)
                }

                fn flat_map_param(&mut self, param: Param) -> SmallVec<[Param; 1]> {
                    walk_flat_map_param(self, param)
                }

                fn flat_map_generic_param(&mut self, param: GenericParam) -> SmallVec<[GenericParam; 1]> {
                    walk_flat_map_generic_param(self, param)
                }

                fn flat_map_expr_field(&mut self, f: ExprField) -> SmallVec<[ExprField; 1]> {
                    walk_flat_map_expr_field(self, f)
                }

                fn flat_map_where_predicate(
                    &mut self,
                    where_predicate: WherePredicate,
                ) -> SmallVec<[WherePredicate; 1]> {
                    walk_flat_map_where_predicate(self, where_predicate)
                }

                fn flat_map_pat_field(&mut self, fp: PatField) -> SmallVec<[PatField; 1]> {
                    walk_flat_map_pat_field(self, fp)
                }
            )?
        }

        pub trait WalkItemKind {
            type Ctxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result;
        }

        // this is only used by the MutVisitor. We include this symmetry here to make writing other functions easier
        $(${ignore($lt)}
            #[expect(unused, rustc::pass_by_value)]
            #[inline]
        )?
        fn visit_span<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, span: &$($lt)? $($mut)? Span) -> V::Result {
            $(${ignore($mut)} vis.visit_span(span))?;
            V::Result::output()
        }

        $(impl_visitable!(|&$lt self: ThinVec<(UseTree, NodeId)>, vis: &mut V, _extra: ()| {
            for (nested_tree, nested_id) in self {
                try_visit!(vis.visit_nested_use_tree(nested_tree, *nested_id));
            }
            V::Result::output()
        });)?
        $(impl_visitable_list!(<$mut> ThinVec<(UseTree, NodeId)>,);)?

        fn walk_item_inner<$($lt,)? K: WalkItemKind, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? Item<K>,
            ctxt: K::Ctxt,
        ) -> V::Result {
            let Item { attrs, id, kind, vis, span, tokens: _ } = item;
            visit_visitable!($($mut)? visitor, id, attrs, vis);
            try_visit!(kind.walk(*span, *id, vis, ctxt, visitor));
            visit_visitable!($($mut)? visitor, span);
            V::Result::output()
        }

        // Do not implement `Walkable`/`MutWalkable` for *Item to avoid confusion.
        pub fn walk_item<$($lt,)? K: WalkItemKind<Ctxt = ()>, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? Item<K>,
        ) -> V::Result {
            walk_item_inner(visitor, item, ())
        }

        // Do not implement `Walkable`/`MutWalkable` for *Item to avoid confusion.
        pub fn walk_assoc_item<$($lt,)? K: WalkItemKind<Ctxt = AssocCtxt>, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? Item<K>,
            ctxt: AssocCtxt,
        ) -> V::Result {
            walk_item_inner(visitor, item, ctxt)
        }

        impl WalkItemKind for ItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Free, visibility, &$($mut)? *func);
                        try_visit!(vis.visit_fn(kind, span, id));
                    }
                    ItemKind::ExternCrate(orig_name, ident) =>
                        visit_visitable!($($mut)? vis, orig_name, ident),
                    ItemKind::Use(use_tree) =>
                        visit_visitable!($($mut)? vis, use_tree),
                    ItemKind::Static(item) =>
                        visit_visitable!($($mut)? vis, item),
                    ItemKind::Const(item) =>
                        visit_visitable!($($mut)? vis, item),
                    ItemKind::Mod(safety, ident, mod_kind) =>
                        visit_visitable!($($mut)? vis, safety, ident, mod_kind),
                    ItemKind::ForeignMod(nm) =>
                        visit_visitable!($($mut)? vis, nm),
                    ItemKind::GlobalAsm(asm) =>
                        visit_visitable!($($mut)? vis, asm),
                    ItemKind::TyAlias(ty_alias) =>
                        visit_visitable!($($mut)? vis, ty_alias),
                    ItemKind::Enum(ident, generics, enum_definition) =>
                        visit_visitable!($($mut)? vis, ident, generics, enum_definition),
                    ItemKind::Struct(ident, generics, variant_data)
                    | ItemKind::Union(ident, generics, variant_data) =>
                        visit_visitable!($($mut)? vis, ident, generics, variant_data),
                    ItemKind::Impl(impl_) =>
                        visit_visitable!($($mut)? vis, impl_),
                    ItemKind::Trait(trait_) =>
                        visit_visitable!($($mut)? vis, trait_),
                    ItemKind::TraitAlias(ident, generics, bounds) => {
                        visit_visitable!($($mut)? vis, ident, generics);
                        visit_visitable_with!($($mut)? vis, bounds, BoundKind::Bound)
                    }
                    ItemKind::MacCall(m) =>
                        visit_visitable!($($mut)? vis, m),
                    ItemKind::MacroDef(ident, def) =>
                        visit_visitable!($($mut)? vis, ident, def),
                    ItemKind::Delegation(delegation) =>
                        visit_visitable!($($mut)? vis, delegation),
                    ItemKind::DelegationMac(dm) =>
                        visit_visitable!($($mut)? vis, dm),
                }
                V::Result::output()
            }
        }

        impl WalkItemKind for AssocItemKind {
            type Ctxt = AssocCtxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    AssocItemKind::Const(item) =>
                        visit_visitable!($($mut)? vis, item),
                    AssocItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), visibility, &$($mut)? *func);
                        try_visit!(vis.visit_fn(kind, span, id))
                    }
                    AssocItemKind::Type(alias) =>
                        visit_visitable!($($mut)? vis, alias),
                    AssocItemKind::MacCall(mac) =>
                        visit_visitable!($($mut)? vis, mac),
                    AssocItemKind::Delegation(delegation) =>
                        visit_visitable!($($mut)? vis, delegation),
                    AssocItemKind::DelegationMac(dm) =>
                        visit_visitable!($($mut)? vis, dm),
                }
                V::Result::output()
            }
        }

        impl WalkItemKind for ForeignItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ForeignItemKind::Static(item) =>
                        visit_visitable!($($mut)? vis, item),
                    ForeignItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Foreign, visibility, &$($mut)?*func);
                        try_visit!(vis.visit_fn(kind, span, id))
                    }
                    ForeignItemKind::TyAlias(alias) =>
                        visit_visitable!($($mut)? vis, alias),
                    ForeignItemKind::MacCall(mac) =>
                        visit_visitable!($($mut)? vis, mac),
                }
                V::Result::output()
            }
        }

        pub fn walk_fn<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, kind: FnKind<$($lt)? $(${ignore($mut)} '_)?>) -> V::Result {
            match kind {
                FnKind::Fn(
                    _ctxt,
                    // Visibility is visited as a part of the item.
                    _vis,
                    Fn { defaultness, ident, sig, generics, contract, body, define_opaque },
                ) => {
                    let FnSig { header, decl, span } = sig;
                    visit_visitable!($($mut)? vis,
                        defaultness, ident, header, generics, decl,
                        contract, body, span, define_opaque
                    )
                }
                FnKind::Closure(binder, coroutine_kind, decl, body) =>
                    visit_visitable!($($mut)? vis, binder, coroutine_kind, decl, body),
            }
            V::Result::output()
        }

        impl_walkable!(|&$($mut)? $($lt)? self: Impl, vis: &mut V| {
            let Impl { generics, of_trait, self_ty, items } = self;
            try_visit!(vis.visit_generics(generics));
            if let Some(box of_trait) = of_trait {
                let TraitImplHeader { defaultness, safety, constness, polarity, trait_ref } = of_trait;
                visit_visitable!($($mut)? vis, defaultness, safety, constness, polarity, trait_ref);
            }
            try_visit!(vis.visit_ty(self_ty));
            visit_visitable_with!($($mut)? vis, items, AssocCtxt::Impl { of_trait: of_trait.is_some() });
            V::Result::output()
        });

        // Special case to call `visit_method_receiver_expr`.
        impl_walkable!(|&$($mut)? $($lt)? self: MethodCall, vis: &mut V| {
            let MethodCall { seg, receiver, args, span } = self;
            try_visit!(vis.visit_method_receiver_expr(receiver));
            visit_visitable!($($mut)? vis, seg, args, span);
            V::Result::output()
        });

        impl_walkable!(|&$($mut)? $($lt)? self: Expr, vis: &mut V| {
            let Expr { id, kind, span, attrs, tokens: _ } = self;
            visit_visitable!($($mut)? vis, id, attrs);
            match kind {
                ExprKind::Array(exprs) =>
                    visit_visitable!($($mut)? vis, exprs),
                ExprKind::ConstBlock(anon_const) =>
                    visit_visitable!($($mut)? vis, anon_const),
                ExprKind::Repeat(element, count) =>
                    visit_visitable!($($mut)? vis, element, count),
                ExprKind::Struct(se) =>
                    visit_visitable!($($mut)? vis, se),
                ExprKind::Tup(exprs) =>
                    visit_visitable!($($mut)? vis, exprs),
                ExprKind::Call(callee_expression, arguments) =>
                    visit_visitable!($($mut)? vis, callee_expression, arguments),
                ExprKind::MethodCall(mc) =>
                    visit_visitable!($($mut)? vis, mc),
                ExprKind::Binary(op, lhs, rhs) =>
                    visit_visitable!($($mut)? vis, op, lhs, rhs),
                ExprKind::AddrOf(kind, mutbl, subexpression) =>
                    visit_visitable!($($mut)? vis, kind, mutbl, subexpression),
                ExprKind::Unary(op, subexpression) =>
                    visit_visitable!($($mut)? vis, op, subexpression),
                ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) =>
                    visit_visitable!($($mut)? vis, subexpression, typ),
                ExprKind::Let(pat, expr, span, _recovered) =>
                    visit_visitable!($($mut)? vis, pat, expr, span),
                ExprKind::If(head_expression, if_block, optional_else) =>
                    visit_visitable!($($mut)? vis, head_expression, if_block, optional_else),
                ExprKind::While(subexpression, block, opt_label) =>
                    visit_visitable!($($mut)? vis, subexpression, block, opt_label),
                ExprKind::ForLoop { pat, iter, body, label, kind } =>
                    visit_visitable!($($mut)? vis, pat, iter, body, label, kind),
                ExprKind::Loop(block, opt_label, span) =>
                    visit_visitable!($($mut)? vis, block, opt_label, span),
                ExprKind::Match(subexpression, arms, kind) =>
                    visit_visitable!($($mut)? vis, subexpression, arms, kind),
                ExprKind::Closure(box Closure {
                    binder,
                    capture_clause,
                    coroutine_kind,
                    constness,
                    movability,
                    fn_decl,
                    body,
                    fn_decl_span,
                    fn_arg_span,
                }) => {
                    visit_visitable!($($mut)? vis, constness, movability, capture_clause);
                    let kind = FnKind::Closure(binder, coroutine_kind, fn_decl, body);
                    try_visit!(vis.visit_fn(kind, *span, *id));
                    visit_visitable!($($mut)? vis, fn_decl_span, fn_arg_span);
                }
                ExprKind::Block(block, opt_label) =>
                    visit_visitable!($($mut)? vis, block, opt_label),
                ExprKind::Gen(capt, body, kind, decl_span) =>
                    visit_visitable!($($mut)? vis, capt, body, kind, decl_span),
                ExprKind::Await(expr, span) | ExprKind::Use(expr, span) =>
                    visit_visitable!($($mut)? vis, expr, span),
                ExprKind::Assign(lhs, rhs, span) =>
                    visit_visitable!($($mut)? vis, lhs, rhs, span),
                ExprKind::AssignOp(op, lhs, rhs) =>
                    visit_visitable!($($mut)? vis, op, lhs, rhs),
                ExprKind::Field(subexpression, ident) =>
                    visit_visitable!($($mut)? vis, subexpression, ident),
                ExprKind::Index(main_expression, index_expression, span) =>
                    visit_visitable!($($mut)? vis, main_expression, index_expression, span),
                ExprKind::Range(start, end, limit) =>
                    visit_visitable!($($mut)? vis, start, end, limit),
                ExprKind::Underscore => {}
                ExprKind::Path(maybe_qself, path) =>
                    visit_visitable!($($mut)? vis, maybe_qself, path),
                ExprKind::Break(opt_label, opt_expr) =>
                    visit_visitable!($($mut)? vis, opt_label, opt_expr),
                ExprKind::Continue(opt_label) =>
                    visit_visitable!($($mut)? vis, opt_label),
                ExprKind::Ret(optional_expression) | ExprKind::Yeet(optional_expression) =>
                    visit_visitable!($($mut)? vis, optional_expression),
                ExprKind::Become(expr) =>
                    visit_visitable!($($mut)? vis, expr),
                ExprKind::MacCall(mac) =>
                    visit_visitable!($($mut)? vis, mac),
                ExprKind::Paren(subexpression) =>
                    visit_visitable!($($mut)? vis, subexpression),
                ExprKind::InlineAsm(asm) =>
                    visit_visitable!($($mut)? vis, asm),
                ExprKind::FormatArgs(f) =>
                    visit_visitable!($($mut)? vis, f),
                ExprKind::OffsetOf(container, fields) =>
                    visit_visitable!($($mut)? vis, container, fields),
                ExprKind::Yield(kind) =>
                    visit_visitable!($($mut)? vis, kind),
                ExprKind::Try(subexpression) =>
                    visit_visitable!($($mut)? vis, subexpression),
                ExprKind::TryBlock(body) =>
                    visit_visitable!($($mut)? vis, body),
                ExprKind::Lit(token) =>
                    visit_visitable!($($mut)? vis, token),
                ExprKind::IncludedBytes(bytes) =>
                    visit_visitable!($($mut)? vis, bytes),
                ExprKind::UnsafeBinderCast(kind, expr, ty) =>
                    visit_visitable!($($mut)? vis, kind, expr, ty),
                ExprKind::Err(_guar) => {}
                ExprKind::Dummy => {}
            }

            visit_span(vis, span)
        });

        define_named_walk!($(($mut))? $Visitor$(<$lt>)?
            pub fn walk_anon_const(AnonConst);
            pub fn walk_arm(Arm);
            //pub fn walk_assoc_item(AssocItem, _ctxt: AssocCtxt);
            pub fn walk_assoc_item_constraint(AssocItemConstraint);
            pub fn walk_attribute(Attribute);
            pub fn walk_block(Block);
            //pub fn walk_nested_use_tree((UseTree, NodeId));
            pub fn walk_capture_by(CaptureBy);
            pub fn walk_closure_binder(ClosureBinder);
            pub fn walk_contract(FnContract);
            pub fn walk_coroutine_kind(CoroutineKind);
            pub fn walk_crate(Crate);
            pub fn walk_expr(Expr);
            pub fn walk_expr_field(ExprField);
            pub fn walk_field_def(FieldDef);
            pub fn walk_fn_decl(FnDecl);
            pub fn walk_fn_header(FnHeader);
            pub fn walk_fn_ret_ty(FnRetTy);
            //pub fn walk_foreign_item(ForeignItem);
            pub fn walk_foreign_mod(ForeignMod);
            pub fn walk_format_args(FormatArgs);
            pub fn walk_generic_arg(GenericArg);
            pub fn walk_generic_args(GenericArgs);
            pub fn walk_generic_param(GenericParam);
            pub fn walk_generics(Generics);
            pub fn walk_inline_asm(InlineAsm);
            pub fn walk_inline_asm_sym(InlineAsmSym);
            //pub fn walk_item(Item);
            pub fn walk_label(Label);
            pub fn walk_lifetime(Lifetime);
            pub fn walk_local(Local);
            pub fn walk_mac(MacCall);
            pub fn walk_macro_def(MacroDef);
            pub fn walk_param_bound(GenericBound);
            pub fn walk_param(Param);
            pub fn walk_pat_field(PatField);
            pub fn walk_path(Path);
            pub fn walk_path_segment(PathSegment);
            pub fn walk_pat(Pat);
            pub fn walk_poly_trait_ref(PolyTraitRef);
            pub fn walk_precise_capturing_arg(PreciseCapturingArg);
            pub fn walk_qself(QSelf);
            pub fn walk_trait_ref(TraitRef);
            pub fn walk_ty_pat(TyPat);
            pub fn walk_ty(Ty);
            pub fn walk_use_tree(UseTree);
            pub fn walk_variant_data(VariantData);
            pub fn walk_variant(Variant);
            pub fn walk_vis(Visibility);
            pub fn walk_where_predicate_kind(WherePredicateKind);
            pub fn walk_where_predicate(WherePredicate);
        );
    };
}

common_visitor_and_walkers!(Visitor<'a>);

macro_rules! generate_list_visit_fns {
    ($($name:ident, $Ty:ty, $visit_fn:ident$(, $param:ident: $ParamTy:ty)*;)+) => {
        $(
            #[allow(unused_parens)]
            impl<'a, V: Visitor<'a>> Visitable<'a, V> for ThinVec<$Ty> {
                type Extra = ($($ParamTy),*);

                #[inline]
                fn visit(
                    &'a self,
                    visitor: &mut V,
                    ($($param),*): Self::Extra,
                ) -> V::Result {
                    $name(visitor, self $(, $param)*)
                }
            }

            fn $name<'a, V: Visitor<'a>>(
                vis: &mut V,
                values: &'a ThinVec<$Ty>,
                $(
                    $param: $ParamTy,
                )*
            ) -> V::Result {
                walk_list!(vis, $visit_fn, values$(,$param)*);
                V::Result::output()
            }
        )+
    }
}

generate_list_visit_fns! {
    visit_items, Box<Item>, visit_item;
    visit_foreign_items, Box<ForeignItem>, visit_foreign_item;
    visit_generic_params, GenericParam, visit_generic_param;
    visit_stmts, Stmt, visit_stmt;
    visit_exprs, Box<Expr>, visit_expr;
    visit_expr_fields, ExprField, visit_expr_field;
    visit_pat_fields, PatField, visit_pat_field;
    visit_variants, Variant, visit_variant;
    visit_assoc_items, Box<AssocItem>, visit_assoc_item, ctxt: AssocCtxt;
    visit_where_predicates, WherePredicate, visit_where_predicate;
    visit_params, Param, visit_param;
    visit_field_defs, FieldDef, visit_field_def;
    visit_arms, Arm, visit_arm;
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) -> V::Result {
    let Stmt { id, kind, span: _ } = statement;
    try_visit!(visitor.visit_id(*id));
    match kind {
        StmtKind::Let(local) => try_visit!(visitor.visit_local(local)),
        StmtKind::Item(item) => try_visit!(visitor.visit_item(item)),
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => try_visit!(visitor.visit_expr(expr)),
        StmtKind::Empty => {}
        StmtKind::MacCall(mac) => {
            let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
            walk_list!(visitor, visit_attribute, attrs);
            try_visit!(visitor.visit_mac_call(mac));
        }
    }
    V::Result::output()
}
