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
use rustc_macros::StableHash;
use rustc_span::{Ident, Span, Spanned, Symbol};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::tokenstream::DelimSpan;

#[derive(Copy, Clone, Debug, PartialEq, Eq, StableHash)]
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

macro_rules! impl_visitable {
    (|&$lt:lifetime $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident,
      $extra:ident: $extra_ty:ty| $block:block) => {
        #[allow(unused_parens)]
        impl<$lt, $vis_ty: Visitor<$lt>> Visitable<$lt, $vis_ty> for $self_ty {
            type Extra = $extra_ty;
            fn visit(&$lt $self, $vis: &mut $vis_ty, $extra: Self::Extra) -> V::Result {
                $block
            }
        }
    };
}

macro_rules! impl_walkable {
    (|&$lt:lifetime $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident| $block:block) => {
        impl<$lt, $vis_ty: Visitor<$lt>> Walkable<$lt, $vis_ty> for $self_ty {
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

macro_rules! fn_visit {
    ($Visitor:ident<$lt:lifetime>
        $( $visit:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?) => $walk:ident; )*
    ) => {
        $(fn $visit(&mut self, node: &$lt $ty $(, $extra_name: $extra_ty)?) -> Self::Result {
            Walkable::walk_ref(node, self)
        })*
    };
}

macro_rules! impl_visitable_visit {
    ($Visitor:ident<$lt:lifetime>
        $( $visit:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?) => $walk:ident; )*
    ) => {
        $(impl_visitable!(|&$lt self: $ty, visitor: &mut V, extra: ($($extra_ty)?)| {
            let ($($extra_name)?) = extra;
            visitor.$visit(self $(, $extra_name)?)
        });)*
    };
}

macro_rules! fn_walk {
    ($Visitor:ident<$lt:lifetime>
        $( $visit:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?) => $walk:ident; )*
    ) => {
        $(pub fn $walk<$lt, V: $Visitor<$lt>>(visitor: &mut V, node: &$lt $ty) -> V::Result {
            Walkable::walk_ref(node, visitor)
        })*
    }
}

/// Higher-order macro that puts all the visit/walk hook information in a single place. The
/// passed-in macro should have a left hand side like this:
/// ```ignore (partial)
/// ($Visitor:ident
///     $( $visit:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?) => $walk:ident; )*
/// ) => ...
/// ```
#[macro_export]
macro_rules! for_each_ast_visit_hook {
    ($Visitor:ident$(<$lt:lifetime>)?
        $macro:ident!
    ) => {
        $macro!($Visitor$(<$lt>)?
            visit_anon_const(AnonConst) => walk_anon_const;
            visit_arm(Arm) => walk_arm;
            //visit_assoc_item(AssocItem, _ctxt: AssocCtxt) => walk_assoc_item;
            visit_assoc_item_constraint(AssocItemConstraint) => walk_assoc_item_constraint;
            visit_attribute(Attribute) => walk_attribute;
            visit_block(Block) => walk_block;
            //visit_nested_use_tree((UseTree, NodeId)) => walk_nested_use_tree;
            visit_capture_by(CaptureBy) => walk_capture_by;
            visit_closure_binder(ClosureBinder) => walk_closure_binder;
            visit_contract(FnContract) => walk_contract;
            visit_coroutine_marker(CoroutineMarker) => walk_coroutine_marker;
            visit_crate(Crate) => walk_crate;
            visit_expr(Expr) => walk_expr;
            visit_expr_field(ExprField) => walk_expr_field;
            visit_field_def(FieldDef) => walk_field_def;
            visit_field_def_extras(FieldDefExtras) => walk_field_def_extras;
            visit_fn_decl(FnDecl) => walk_fn_decl;
            visit_fn_header(FnHeader) => walk_fn_header;
            visit_fn_ret_ty(FnRetTy) => walk_fn_ret_ty;
            //visit_foreign_item(ForeignItem) => walk_foreign_item;
            visit_foreign_mod(ForeignMod) => walk_foreign_mod;
            visit_format_args(FormatArgs) => walk_format_args;
            visit_generic_arg(GenericArg) => walk_generic_arg;
            visit_generic_args(GenericArgs) => walk_generic_args;
            visit_generic_param(GenericParam) => walk_generic_param;
            visit_generics(Generics) => walk_generics;
            visit_inline_asm(InlineAsm) => walk_inline_asm;
            visit_inline_asm_sym(InlineAsmSym) => walk_inline_asm_sym;
            visit_impl_restriction(ImplRestriction) => walk_impl_restriction;
            //visit_item(Item) => walk_item;
            visit_label(Label) => walk_label;
            visit_lifetime(Lifetime, _ctxt: LifetimeCtxt) => walk_lifetime;
            visit_local(Local) => walk_local;
            visit_mac_call(MacCall) => walk_mac;
            visit_macro_def(MacroDef) => walk_macro_def;
            visit_mut_restriction(MutRestriction) => walk_mut_restriction;
            visit_param_bound(GenericBound, _ctxt: BoundKind) => walk_param_bound;
            visit_param(Param) => walk_param;
            visit_pat_field(PatField) => walk_pat_field;
            visit_path(Path) => walk_path;
            visit_path_segment(PathSegment) => walk_path_segment;
            visit_pat(Pat) => walk_pat;
            visit_poly_trait_ref(PolyTraitRef) => walk_poly_trait_ref;
            visit_precise_capturing_arg(PreciseCapturingArg) => walk_precise_capturing_arg;
            visit_qself(QSelf) => walk_qself;
            visit_test_binder_body(TestBinderBody) => walk_test_binder_body;
            visit_test_binder_constraint(TestBinderConstraint) => walk_test_binder_constraint;
            visit_test_binder_constraints(TestBinderConstraints) => walk_test_binder_constraints;
            visit_test_binder_exists(TestBinderExists) => walk_test_binder_exists;
            visit_test_binder_forall(TestBinderForall) => walk_test_binder_forall;
            visit_trait_ref(TraitRef) => walk_trait_ref;
            visit_ty_pat(TyPat) => walk_ty_pat;
            visit_ty(Ty) => walk_ty;
            visit_use_tree(UseTree) => walk_use_tree;
            visit_variant_data(VariantData) => walk_variant_data;
            visit_variant(Variant) => walk_variant;
            visit_vis(Visibility) => walk_vis;
            visit_where_predicate_kind(WherePredicateKind) => walk_where_predicate_kind;
            visit_where_predicate(WherePredicate) => walk_where_predicate;
        );
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
            Closure(
                &'a $($mut)? ClosureBinder,
                &'a $($mut)? Option<CoroutineMarker>,
                &'a $($mut)? Box<FnDecl>,
                &'a $($mut)? Box<Expr>,
            ),
        }

        impl<'a> FnKind<'_> {
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
        impl_visitable_noop!($(<$lt>)?
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
            Pinnedness,
            Result<(), rustc_span::ErrorGuaranteed>,
            rustc_data_structures::fx::FxHashMap<Symbol, usize>,
            rustc_span::ErrorGuaranteed,
            std::borrow::Cow<'_, str>,
            Symbol,
            SyntheticAttr,
            u8,
            usize,
        );
        // `Span` is only a no-op for the non-mutable visitor.
        $(impl_visitable_noop!(<$lt> Span,);)?

        // This macro generates `impl Visitable` and `impl MutVisitable` that simply iterate over
        // their contents. We do not use a generic impl for `ThinVec` because we want to allow
        // custom visits for the `MutVisitor`.
        impl_visitable_list!($(<$lt>)?
            ThinVec<AngleBracketedArg>,
            ThinVec<Attribute>,
            ThinVec<GenericBound>,
            ThinVec<Ident>,
            ThinVec<(Ident, Option<Ident>)>,
            ThinVec<(NodeId, Path)>,
            ThinVec<PathSegment>,
            ThinVec<PreciseCapturingArg>,
            ThinVec<Pat>,
            ThinVec<TestBinderConstraint>,
            ThinVec<TestBinderExists>,
            ThinVec<TestBinderForall>,
            ThinVec<Box<Ty>>,
            ThinVec<TyPat>,
            ThinVec<EiiImpl>,
        );

        // This macro generates `impl Visitable` and `impl MutVisitable` that forward to `Walkable`
        // or `MutWalkable`. By default, all types that do not have a custom visit method in the
        // visitor should appear here.
        impl_visitable_direct!($(<$lt>)?
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
            CoroutineKind,
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
            ConstBlockItem,
            ConstItem,
            Defaultness,
            Delegation,
            DelegationMac,
            DelegationSuffixes,
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
            Guard,
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
            RestrictionKind,
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
            EiiDecl,
            EiiImpl,
        );

        /// Each method of this trait is a hook to be potentially
        /// overridden. Each method's default implementation recursively visits
        /// the substructure of the input via the corresponding `walk` method;
        #[doc = concat!(
            " e.g., the `visit_item` method by default calls `visit"
            $(, "_", stringify!($mut))?,
            "::walk_item`."
        )]
        ///
        /// If you want to ensure that your code handles every variant
        /// explicitly, you need to override each method. (And you also need
        /// to monitor future changes to this trait in case a new method with a
        /// new default implementation gets introduced.)
        ///
        /// Every `walk_*` method uses deconstruction to access fields of structs and
        /// enums. This will result in a compile error if a field is added, which makes
        /// it more likely the appropriate visit call will be added for it.
        pub trait $Visitor<$($lt)?>: Sized $(${ignore($mut)} + MutVisitorResult<Result = ()>)? {
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
            fn visit_ident(&mut self, Ident { name: _, span }: &$($lt)? $($mut)? Ident)
                -> Self::Result
            {
                visit_visitable!(self, span);
                Self::Result::output()
            }

            crate::for_each_ast_visit_hook!($Visitor$(<$lt>)? fn_visit!);

            // We want `Visitor` to take the `NodeId` by value.
            fn visit_id(&mut self, _id: $(&$mut)? NodeId) -> Self::Result {
                Self::Result::output()
            }

            /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
            /// It can be removed once that feature is stabilized.
            fn visit_method_receiver_expr(&mut self, ex: &$($lt)? $($mut)? Expr) -> Self::Result {
                self.visit_expr(ex)
            }

            fn visit_item(&mut self, item: &$($lt)? $($mut)? Item) -> Self::Result {
                walk_item(self, item)
            }

            fn visit_foreign_item(&mut self, item: &$($lt)? $($mut)? ForeignItem) -> Self::Result {
                walk_item(self, item)
            }

            fn visit_assoc_item(&mut self, item: &$($lt)? $($mut)? AssocItem, ctxt: AssocCtxt)
                -> Self::Result
            {
                walk_assoc_item(self, item, ctxt)
            }

            // for `MutVisitor`: `Span` and `NodeId` are mutated at the caller site.
            fn visit_fn(
                &mut self,
                fk: FnKind<$($lt)? $(${ignore($mut)} '_)?>,
                _: &AttrVec,
                _: Span,
                _: NodeId,
            ) -> Self::Result {
                walk_fn(self, fk)
            }

            // (non-mut) `Visitor`-only methods
            $(
                fn visit_stmt(&mut self, s: &$lt Stmt) -> Self::Result {
                    walk_stmt(self, s)
                }

                fn visit_nested_use_tree(&mut self, use_tree: &$lt UseTree, id: NodeId)
                    -> Self::Result
                {
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
                    // Do nothing.
                }

                fn flat_map_foreign_item(&mut self, ni: Box<ForeignItem>)
                    -> SmallVec<[Box<ForeignItem>; 1]>
                {
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

                fn flat_map_generic_param(&mut self, param: GenericParam)
                    -> SmallVec<[GenericParam; 1]>
                {
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

        crate::for_each_ast_visit_hook!($Visitor$(<$lt>)? impl_visitable_visit!);

        impl_visitable!(|&$($lt)? $($mut)? self: Ident, visitor: &mut V, _extra: ()| {
            visitor.visit_ident(self)
        });

        $(
            impl_visitable!(
                |&$lt self: NodeId, visitor: &mut V, _extra: ()| {
                    visitor.visit_id(*self)
                }
            );
        )?
        $(
            impl_visitable!(
                |&$mut self: NodeId, visitor: &mut V, _extra: ()| {
                    visitor.visit_id(self)
                }
            );

            impl_visitable!(|&mut self: Span, visitor: &mut V, _extra: ()| {
                visitor.visit_span(self)
            });
        )?

        impl_visitable!(|&$($lt)? $($mut)? self: Item, vis: &mut V, _extra: ()| {
            vis.visit_item(self)
        });
        impl_visitable!(|&$($lt)? $($mut)? self: ForeignItem, vis: &mut V, _extra: ()| {
            vis.visit_foreign_item(self)
        });
        impl_visitable!(|&$($lt)? $($mut)? self: AssocItem, vis: &mut V, ctxt: AssocCtxt| {
            vis.visit_assoc_item(self, ctxt)
        });

        pub trait WalkItemKind {
            type Ctxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                attrs: &AttrVec,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result;
        }

        $(impl_visitable!(|&$lt self: ThinVec<(UseTree, NodeId)>, vis: &mut V, _extra: ()| {
            for (nested_tree, nested_id) in self {
                try_visit!(vis.visit_nested_use_tree(nested_tree, *nested_id));
            }
            V::Result::output()
        });)?
        $(${ignore($mut)} impl_visitable_list!(ThinVec<(UseTree, NodeId)>,);)?

        fn walk_item_inner<$($lt,)? K: WalkItemKind, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($lt)? $($mut)? Item<K>,
            ctxt: K::Ctxt,
        ) -> V::Result {
            let Item { attrs, id, kind, vis, span, tokens: _ } = item;
            visit_visitable!(visitor, id, attrs, vis);
            try_visit!(kind.walk(attrs, *span, *id, vis, ctxt, visitor));
            visit_visitable!(visitor, span);
            V::Result::output()
        }

        // Do not implement `Walkable`/`MutWalkable` for *Item to avoid confusion.
        pub fn walk_item<$($lt,)? K: WalkItemKind<Ctxt = ()>, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($lt)? $($mut)? Item<K>,
        ) -> V::Result {
            walk_item_inner(visitor, item, ())
        }

        // Do not implement `Walkable`/`MutWalkable` for *Item to avoid confusion.
        pub fn walk_assoc_item<$($lt,)? K: WalkItemKind<Ctxt = AssocCtxt>, V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($lt)? $($mut)? Item<K>,
            ctxt: AssocCtxt,
        ) -> V::Result {
            walk_item_inner(visitor, item, ctxt)
        }

        impl WalkItemKind for ItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                attrs: &AttrVec,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Free, visibility, &$($mut)? *func);
                        try_visit!(vis.visit_fn(kind, attrs, span, id));
                    }
                    ItemKind::ExternCrate(orig_name, ident) =>
                        visit_visitable!(vis, orig_name, ident),
                    ItemKind::Use(use_tree) =>
                        visit_visitable!(vis, use_tree),
                    ItemKind::Static(item) =>
                        visit_visitable!(vis, item),
                    ItemKind::ConstBlock(item) =>
                        visit_visitable!(vis, item),
                    ItemKind::Const(item) =>
                        visit_visitable!(vis, item),
                    ItemKind::Mod(safety, ident, mod_kind) =>
                        visit_visitable!(vis, safety, ident, mod_kind),
                    ItemKind::ForeignMod(nm) =>
                        visit_visitable!(vis, nm),
                    ItemKind::GlobalAsm(asm) =>
                        visit_visitable!(vis, asm),
                    ItemKind::TyAlias(ty_alias) =>
                        visit_visitable!(vis, ty_alias),
                    ItemKind::Enum(ident, generics, enum_definition) =>
                        visit_visitable!(vis, ident, generics, enum_definition),
                    ItemKind::Struct(ident, generics, variant_data)
                    | ItemKind::Union(ident, generics, variant_data) =>
                        visit_visitable!(vis, ident, generics, variant_data),
                    ItemKind::Impl(impl_) =>
                        visit_visitable!(vis, impl_),
                    ItemKind::Trait(trait_) =>
                        visit_visitable!(vis, trait_),
                    ItemKind::TraitAlias(TraitAlias { constness, ident, generics, bounds }) => {
                        visit_visitable!(vis, constness, ident, generics);
                        visit_visitable_with!(vis, bounds, BoundKind::Bound)
                    }
                    ItemKind::MacCall(m) =>
                        visit_visitable!(vis, m),
                    ItemKind::MacroDef(ident, def) =>
                        visit_visitable!(vis, ident, def),
                    ItemKind::Delegation(delegation) =>
                        visit_visitable!(vis, delegation),
                    ItemKind::DelegationMac(dm) =>
                        visit_visitable!(vis, dm),
                    ItemKind::TestBinderConstraints(item) =>
                        visit_visitable!(vis, item),
                }
                V::Result::output()
            }
        }

        impl WalkItemKind for AssocItemKind {
            type Ctxt = AssocCtxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                attrs: &AttrVec,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    AssocItemKind::Const(item) =>
                        visit_visitable!(vis, item),
                    AssocItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), visibility, &$($mut)? *func);
                        try_visit!(vis.visit_fn(kind, attrs, span, id))
                    }
                    AssocItemKind::Type(alias) =>
                        visit_visitable!(vis, alias),
                    AssocItemKind::MacCall(mac) =>
                        visit_visitable!(vis, mac),
                    AssocItemKind::Delegation(delegation) =>
                        visit_visitable!(vis, delegation),
                    AssocItemKind::DelegationMac(dm) =>
                        visit_visitable!(vis, dm),
                }
                V::Result::output()
            }
        }

        impl WalkItemKind for ForeignItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                attrs: &AttrVec,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ForeignItemKind::Static(item) =>
                        visit_visitable!(vis, item),
                    ForeignItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Foreign, visibility, &$($mut)? *func);
                        try_visit!(vis.visit_fn(kind, attrs, span, id))
                    }
                    ForeignItemKind::TyAlias(alias) =>
                        visit_visitable!(vis, alias),
                    ForeignItemKind::MacCall(mac) =>
                        visit_visitable!(vis, mac),
                }
                V::Result::output()
            }
        }

        pub fn walk_fn<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            kind: FnKind<$($lt)? $(${ignore($mut)} '_)?>,
        ) -> V::Result {
            match kind {
                FnKind::Fn(
                    _ctxt,
                    // Visibility is visited as a part of the item.
                    _vis,
                    Fn {
                        defaultness,
                        ident,
                        sig,
                        generics,
                        contract,
                        body,
                        define_opaque,
                        eii_impl,
                    },
                ) => {
                    let FnSig { header, decl, span } = sig;
                    visit_visitable!(vis,
                        defaultness, ident, header, generics, decl,
                        contract, body, span, define_opaque, eii_impl
                    );
                }
                FnKind::Closure(binder, coroutine_marker, decl, body) =>
                    visit_visitable!(vis, binder, coroutine_marker, decl, body),
            }
            V::Result::output()
        }

        impl_walkable!(|&$($lt)? $($mut)? self: Impl, vis: &mut V| {
            let Impl { generics, of_trait, self_ty, items, constness: _ } = self;
            try_visit!(vis.visit_generics(generics));
            if let Some(of_trait) = of_trait {
                let TraitImplHeader { defaultness, safety, polarity, trait_ref } = of_trait;
                visit_visitable!(vis, defaultness, safety, polarity, trait_ref);
            }
            try_visit!(vis.visit_ty(self_ty));
            visit_visitable_with!(vis, items, AssocCtxt::Impl { of_trait: of_trait.is_some() });
            V::Result::output()
        });

        // Special case to call `visit_method_receiver_expr`.
        impl_walkable!(|&$($lt)? $($mut)? self: MethodCall, vis: &mut V| {
            let MethodCall { seg, receiver, args, span } = self;
            try_visit!(vis.visit_method_receiver_expr(receiver));
            visit_visitable!(vis, seg, args, span);
            V::Result::output()
        });

        impl_walkable!(|&$($lt)? $($mut)? self: Expr, vis: &mut V| {
            let Expr { id, kind, span, attrs, tokens: _ } = self;
            visit_visitable!(vis, id, attrs);
            match kind {
                ExprKind::Array(exprs) =>
                    visit_visitable!(vis, exprs),
                ExprKind::ConstBlock(anon_const) =>
                    visit_visitable!(vis, anon_const),
                ExprKind::Repeat(element, count) =>
                    visit_visitable!(vis, element, count),
                ExprKind::Struct(se) =>
                    visit_visitable!(vis, se),
                ExprKind::Tup(exprs) =>
                    visit_visitable!(vis, exprs),
                ExprKind::Call(callee_expression, arguments) =>
                    visit_visitable!(vis, callee_expression, arguments),
                ExprKind::MethodCall(mc) =>
                    visit_visitable!(vis, mc),
                ExprKind::Binary(op, lhs, rhs) =>
                    visit_visitable!(vis, op, lhs, rhs),
                ExprKind::AddrOf(kind, mutbl, subexpression) =>
                    visit_visitable!(vis, kind, mutbl, subexpression),
                ExprKind::Unary(op, subexpression) =>
                    visit_visitable!(vis, op, subexpression),
                ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) =>
                    visit_visitable!(vis, subexpression, typ),
                ExprKind::Let(pat, expr, span, _recovered) =>
                    visit_visitable!(vis, pat, expr, span),
                ExprKind::If(head_expression, if_block, optional_else) =>
                    visit_visitable!(vis, head_expression, if_block, optional_else),
                ExprKind::While(subexpression, block, opt_label) =>
                    visit_visitable!(vis, subexpression, block, opt_label),
                ExprKind::ForLoop(ForLoop { pat, iter, body, label, kind }) =>
                    visit_visitable!(vis, pat, iter, body, label, kind),
                ExprKind::Loop(block, opt_label, span) =>
                    visit_visitable!(vis, block, opt_label, span),
                ExprKind::Match(subexpression, arms, kind) =>
                    visit_visitable!(vis, subexpression, arms, kind),
                ExprKind::Closure(Closure {
                    binder,
                    capture_clause,
                    coroutine_marker,
                    constness,
                    movability,
                    fn_decl,
                    body,
                    fn_decl_span,
                    fn_arg_span,
                }) => {
                    visit_visitable!(vis, constness, movability, capture_clause);
                    let kind = FnKind::Closure(binder, coroutine_marker, fn_decl, body);
                    try_visit!(vis.visit_fn(kind, attrs, *span, *id));
                    visit_visitable!(vis, fn_decl_span, fn_arg_span);
                }
                ExprKind::Block(block, opt_label) =>
                    visit_visitable!(vis, block, opt_label),
                ExprKind::Gen(capt, body, kind, decl_span) =>
                    visit_visitable!(vis, capt, body, kind, decl_span),
                ExprKind::Await(expr, span)
                | ExprKind::Move(expr, span)
                | ExprKind::Use(expr, span) =>
                    visit_visitable!(vis, expr, span),
                ExprKind::Assign(lhs, rhs, span) =>
                    visit_visitable!(vis, lhs, rhs, span),
                ExprKind::AssignOp(op, lhs, rhs) =>
                    visit_visitable!(vis, op, lhs, rhs),
                ExprKind::Field(subexpression, ident) =>
                    visit_visitable!(vis, subexpression, ident),
                ExprKind::Index(main_expression, index_expression, span) =>
                    visit_visitable!(vis, main_expression, index_expression, span),
                ExprKind::Range(start, end, limit) =>
                    visit_visitable!(vis, start, end, limit),
                ExprKind::Underscore => {}
                ExprKind::Path(maybe_qself, path) =>
                    visit_visitable!(vis, maybe_qself, path),
                ExprKind::Break(opt_label, opt_expr) =>
                    visit_visitable!(vis, opt_label, opt_expr),
                ExprKind::Continue(opt_label) =>
                    visit_visitable!(vis, opt_label),
                ExprKind::Ret(optional_expression) | ExprKind::Yeet(optional_expression) =>
                    visit_visitable!(vis, optional_expression),
                ExprKind::Become(expr) =>
                    visit_visitable!(vis, expr),
                ExprKind::MacCall(mac) =>
                    visit_visitable!(vis, mac),
                ExprKind::Paren(subexpression) =>
                    visit_visitable!(vis, subexpression),
                ExprKind::InlineAsm(asm) =>
                    visit_visitable!(vis, asm),
                ExprKind::FormatArgs(f) =>
                    visit_visitable!(vis, f),
                ExprKind::OffsetOf(container, fields) =>
                    visit_visitable!(vis, container, fields),
                ExprKind::Yield(kind) =>
                    visit_visitable!(vis, kind),
                ExprKind::Try(subexpression) =>
                    visit_visitable!(vis, subexpression),
                ExprKind::TryBlock(body, optional_type) =>
                    visit_visitable!(vis, body, optional_type),
                ExprKind::Lit(token) =>
                    visit_visitable!(vis, token),
                ExprKind::IncludedBytes(bytes) =>
                    visit_visitable!(vis, bytes),
                ExprKind::UnsafeBinderCast(kind, expr, ty) =>
                    visit_visitable!(vis, kind, expr, ty),
                ExprKind::DirectConstArg(expr) =>
                    visit_visitable!(vis, expr),
                ExprKind::Err(_guar) => {}
                ExprKind::Dummy => {}
            }

            visit_visitable!(vis, span);
            V::Result::output()
        });

        crate::for_each_ast_visit_hook!($Visitor$(<$lt>)? fn_walk!);
    };
}

common_visitor_and_walkers!(Visitor<'a>);

macro_rules! generate_list_visit_fns {
    ($($visit_fn:ident, $Ty:ty $(, $param:ident: $ParamTy:ty)?;)+) => {
        $(
            #[allow(unused_parens)]
            impl<'a, V: Visitor<'a>> Visitable<'a, V> for ThinVec<$Ty> {
                type Extra = ($($ParamTy)?);

                #[inline]
                fn visit(&'a self, visitor: &mut V, ($($param)?): Self::Extra) -> V::Result {
                    walk_list!(visitor, $visit_fn, self $(, $param)?);
                    V::Result::output()
                }
            }
        )+
    }
}

generate_list_visit_fns! {
    visit_item, Box<Item>;
    visit_foreign_item, Box<ForeignItem>;
    visit_generic_param, GenericParam;
    visit_stmt, Stmt;
    visit_expr, Box<Expr>;
    visit_expr_field, ExprField;
    visit_pat_field, PatField;
    visit_variant, Variant;
    visit_assoc_item, Box<AssocItem>, ctxt: AssocCtxt;
    visit_where_predicate, WherePredicate;
    visit_param, Param;
    visit_field_def, FieldDef;
    visit_arm, Arm;
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
