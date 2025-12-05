//! A `MutVisitor` represents an AST modification; it accepts an AST piece and
//! mutates it in place. So, for instance, macro expansion is a `MutVisitor`
//! that walks over an AST and modifies it.
//!
//! Note: using a `MutVisitor` (other than the `MacroExpander` `MutVisitor`) on
//! an AST before macro expansion is probably a bad idea. For instance,
//! a `MutVisitor` renaming item names in a module will miss all of those
//! that are created by the expansion of a macro.

use std::ops::DerefMut;
use std::panic;

use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::tokenstream::*;
use crate::visit::{AssocCtxt, BoundKind, FnCtxt, LifetimeCtxt, VisitorResult, try_visit};

mod sealed {
    use rustc_ast_ir::visit::VisitorResult;

    /// This is for compatibility with the regular `Visitor`.
    pub trait MutVisitorResult {
        type Result: VisitorResult;
    }

    impl<T> MutVisitorResult for T {
        type Result = ();
    }
}

use sealed::MutVisitorResult;

pub(crate) trait MutVisitable<V: MutVisitor> {
    type Extra: Copy;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra);
}

impl<V: MutVisitor, T: ?Sized> MutVisitable<V> for Box<T>
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        (**self).visit_mut(visitor, extra)
    }
}

impl<V: MutVisitor, T> MutVisitable<V> for Option<T>
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        if let Some(this) = self {
            this.visit_mut(visitor, extra)
        }
    }
}

impl<V: MutVisitor, T> MutVisitable<V> for Spanned<T>
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        let Spanned { span, node } = self;
        span.visit_mut(visitor, ());
        node.visit_mut(visitor, extra);
    }
}

impl<V: MutVisitor, T> MutVisitable<V> for [T]
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        for item in self {
            item.visit_mut(visitor, extra);
        }
    }
}

impl<V: MutVisitor, T> MutVisitable<V> for Vec<T>
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        for item in self {
            item.visit_mut(visitor, extra);
        }
    }
}

impl<V: MutVisitor, T> MutVisitable<V> for (T,)
where
    T: MutVisitable<V>,
{
    type Extra = T::Extra;
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        self.0.visit_mut(visitor, extra);
    }
}

impl<V: MutVisitor, T1, T2> MutVisitable<V> for (T1, T2)
where
    T1: MutVisitable<V, Extra = ()>,
    T2: MutVisitable<V, Extra = ()>,
{
    type Extra = ();
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        self.0.visit_mut(visitor, extra);
        self.1.visit_mut(visitor, extra);
    }
}

impl<V: MutVisitor, T1, T2, T3> MutVisitable<V> for (T1, T2, T3)
where
    T1: MutVisitable<V, Extra = ()>,
    T2: MutVisitable<V, Extra = ()>,
    T3: MutVisitable<V, Extra = ()>,
{
    type Extra = ();
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        self.0.visit_mut(visitor, extra);
        self.1.visit_mut(visitor, extra);
        self.2.visit_mut(visitor, extra);
    }
}

impl<V: MutVisitor, T1, T2, T3, T4> MutVisitable<V> for (T1, T2, T3, T4)
where
    T1: MutVisitable<V, Extra = ()>,
    T2: MutVisitable<V, Extra = ()>,
    T3: MutVisitable<V, Extra = ()>,
    T4: MutVisitable<V, Extra = ()>,
{
    type Extra = ();
    fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
        self.0.visit_mut(visitor, extra);
        self.1.visit_mut(visitor, extra);
        self.2.visit_mut(visitor, extra);
        self.3.visit_mut(visitor, extra);
    }
}

pub trait MutWalkable<V: MutVisitor> {
    fn walk_mut(&mut self, visitor: &mut V);
}

macro_rules! visit_visitable {
    (mut $visitor:expr, $($expr:expr),* $(,)?) => {{
        $(MutVisitable::visit_mut($expr, $visitor, ());)*
    }};
}

macro_rules! visit_visitable_with {
    (mut $visitor:expr, $expr:expr, $extra:expr $(,)?) => {
        MutVisitable::visit_mut($expr, $visitor, $extra)
    };
}

macro_rules! walk_walkable {
    ($visitor:expr, $expr:expr, mut) => {
        MutWalkable::walk_mut($expr, $visitor)
    };
}

macro_rules! impl_visitable {
    (|&mut $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident,
      $extra:ident: $extra_ty:ty| $block:block) => {
        #[allow(unused_parens, non_local_definitions)]
        impl<$vis_ty: MutVisitor> MutVisitable<$vis_ty> for $self_ty {
            type Extra = $extra_ty;
            fn visit_mut(&mut $self, $vis: &mut $vis_ty, $extra: Self::Extra) -> V::Result {
                $block
            }
        }
    };
}

macro_rules! impl_walkable {
    ($(<$K:ident: $Kb:ident>)? |&mut $self:ident: $self_ty:ty,
      $vis:ident: &mut $vis_ty:ident| $block:block) => {
        #[allow(unused_parens, non_local_definitions)]
        impl<$($K: $Kb,)? $vis_ty: MutVisitor> MutWalkable<$vis_ty> for $self_ty {
            fn walk_mut(&mut $self, $vis: &mut $vis_ty) -> V::Result {
                $block
            }
        }
    };
}

macro_rules! impl_visitable_noop {
    (<mut> $($ty:ty,)*) => {
        $(
            impl_visitable!(|&mut self: $ty, _vis: &mut V, _extra: ()| {});
        )*
    };
}

macro_rules! impl_visitable_list {
    (<mut> $($ty:ty,)*) => {
        $(impl<V: MutVisitor, T> MutVisitable<V> for $ty
        where
            for<'a> &'a mut $ty: IntoIterator<Item = &'a mut T>,
            T: MutVisitable<V>,
        {
            type Extra = <T as MutVisitable<V>>::Extra;

            #[inline]
            fn visit_mut(&mut self, visitor: &mut V, extra: Self::Extra) {
                for i in self {
                    i.visit_mut(visitor, extra);
                }
            }
        })*
    }
}

macro_rules! impl_visitable_direct {
    (<mut> $($ty:ty,)*) => {
        $(impl_visitable!(
            |&mut self: $ty, visitor: &mut V, _extra: ()| {
                MutWalkable::walk_mut(self, visitor)
            }
        );)*
    }
}

macro_rules! impl_visitable_calling_walkable {
    (<mut>
        $( fn $method:ident($ty:ty $(, $extra_name:ident: $extra_ty:ty)?); )*
    ) => {
        $(fn $method(&mut self, node: &mut $ty $(, $extra_name:$extra_ty)?) {
            impl_visitable!(|&mut self: $ty, visitor: &mut V, extra: ($($extra_ty)?)| {
                let ($($extra_name)?) = extra;
                visitor.$method(self $(, $extra_name)?);
            });
            walk_walkable!(self, node, mut)
        })*
    }
}

macro_rules! define_named_walk {
    ((mut) $Visitor:ident
        $( pub fn $method:ident($ty:ty); )*
    ) => {
        $(pub fn $method<V: $Visitor>(visitor: &mut V, node: &mut $ty) {
            walk_walkable!(visitor, node, mut)
        })*
    };
}

super::common_visitor_and_walkers!((mut) MutVisitor);

macro_rules! generate_flat_map_visitor_fns {
    ($($name:ident, $Ty:ty, $flat_map_fn:ident$(, $param:ident: $ParamTy:ty)*;)+) => {
        $(
            #[allow(unused_parens)]
            impl<V: MutVisitor> MutVisitable<V> for ThinVec<$Ty> {
                type Extra = ($($ParamTy),*);

                #[inline]
                fn visit_mut(
                    &mut self,
                    visitor: &mut V,
                    ($($param),*): Self::Extra,
                ) -> V::Result {
                    $name(visitor, self $(, $param)*)
                }
            }

            fn $name<V: MutVisitor>(
                vis: &mut V,
                values: &mut ThinVec<$Ty>,
                $(
                    $param: $ParamTy,
                )*
            ) {
                values.flat_map_in_place(|value| vis.$flat_map_fn(value$(,$param)*));
            }
        )+
    }
}

generate_flat_map_visitor_fns! {
    visit_items, Box<Item>, flat_map_item;
    visit_foreign_items, Box<ForeignItem>, flat_map_foreign_item;
    visit_generic_params, GenericParam, flat_map_generic_param;
    visit_stmts, Stmt, flat_map_stmt;
    visit_exprs, Box<Expr>, filter_map_expr;
    visit_expr_fields, ExprField, flat_map_expr_field;
    visit_pat_fields, PatField, flat_map_pat_field;
    visit_variants, Variant, flat_map_variant;
    visit_assoc_items, Box<AssocItem>, flat_map_assoc_item, ctxt: AssocCtxt;
    visit_where_predicates, WherePredicate, flat_map_where_predicate;
    visit_params, Param, flat_map_param;
    visit_field_defs, FieldDef, flat_map_field_def;
    visit_arms, Arm, flat_map_arm;
}

pub fn walk_flat_map_pat_field<T: MutVisitor>(
    vis: &mut T,
    mut fp: PatField,
) -> SmallVec<[PatField; 1]> {
    vis.visit_pat_field(&mut fp);
    smallvec![fp]
}

macro_rules! generate_walk_flat_map_fns {
    ($($fn_name:ident($Ty:ty$(,$extra_name:ident: $ExtraTy:ty)*) => $visit_fn_name:ident;)+) => {$(
        pub fn $fn_name<V: MutVisitor>(vis: &mut V, mut value: $Ty$(,$extra_name: $ExtraTy)*) -> SmallVec<[$Ty; 1]> {
            vis.$visit_fn_name(&mut value$(,$extra_name)*);
            smallvec![value]
        }
    )+};
}

generate_walk_flat_map_fns! {
    walk_flat_map_arm(Arm) => visit_arm;
    walk_flat_map_variant(Variant) => visit_variant;
    walk_flat_map_param(Param) => visit_param;
    walk_flat_map_generic_param(GenericParam) => visit_generic_param;
    walk_flat_map_where_predicate(WherePredicate) => visit_where_predicate;
    walk_flat_map_field_def(FieldDef) => visit_field_def;
    walk_flat_map_expr_field(ExprField) => visit_expr_field;
    walk_flat_map_item(Box<Item>) => visit_item;
    walk_flat_map_foreign_item(Box<ForeignItem>) => visit_foreign_item;
    walk_flat_map_assoc_item(Box<AssocItem>, ctxt: AssocCtxt) => visit_assoc_item;
}

pub fn walk_filter_map_expr<T: MutVisitor>(vis: &mut T, mut e: Box<Expr>) -> Option<Box<Expr>> {
    vis.visit_expr(&mut e);
    Some(e)
}

pub fn walk_flat_map_stmt<T: MutVisitor>(
    vis: &mut T,
    Stmt { kind, span, mut id }: Stmt,
) -> SmallVec<[Stmt; 1]> {
    vis.visit_id(&mut id);
    let mut stmts: SmallVec<[Stmt; 1]> = walk_flat_map_stmt_kind(vis, kind)
        .into_iter()
        .map(|kind| Stmt { id, kind, span })
        .collect();
    match &mut stmts[..] {
        [] => {}
        [stmt] => vis.visit_span(&mut stmt.span),
        _ => panic!(
            "cloning statement `NodeId`s is prohibited by default, \
             the visitor should implement custom statement visiting"
        ),
    }
    stmts
}

fn walk_flat_map_stmt_kind<T: MutVisitor>(vis: &mut T, kind: StmtKind) -> SmallVec<[StmtKind; 1]> {
    match kind {
        StmtKind::Let(mut local) => smallvec![StmtKind::Let({
            vis.visit_local(&mut local);
            local
        })],
        StmtKind::Item(item) => vis.flat_map_item(item).into_iter().map(StmtKind::Item).collect(),
        StmtKind::Expr(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Expr).collect(),
        StmtKind::Semi(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Semi).collect(),
        StmtKind::Empty => smallvec![StmtKind::Empty],
        StmtKind::MacCall(mut mac) => {
            let MacCallStmt { mac: mac_, style: _, attrs, tokens: _ } = mac.deref_mut();
            for attr in attrs {
                vis.visit_attribute(attr);
            }
            vis.visit_mac_call(mac_);
            smallvec![StmtKind::MacCall(mac)]
        }
    }
}
