//! # Categorization
//!
//! The job of the categorization module is to analyze an expression to
//! determine what kind of memory is used in evaluating it (for example,
//! where dereferences occur and what kind of pointer is dereferenced;
//! whether the memory is mutable, etc.).
//!
//! Categorization effectively transforms all of our expressions into
//! expressions of the following forms (the actual enum has many more
//! possibilities, naturally, but they are all variants of these base
//! forms):
//!
//!     E = rvalue    // some computed rvalue
//!       | x         // address of a local variable or argument
//!       | *E        // deref of a ptr
//!       | E.comp    // access to an interior component
//!
//! Imagine a routine ToAddr(Expr) that evaluates an expression and returns an
//! address where the result is to be found. If Expr is a place, then this
//! is the address of the place. If `Expr` is an rvalue, this is the address of
//! some temporary spot in memory where the result is stored.
//!
//! Now, `cat_expr()` classifies the expression `Expr` and the address `A = ToAddr(Expr)`
//! as follows:
//!
//! - `cat`: what kind of expression was this? This is a subset of the
//!   full expression forms which only includes those that we care about
//!   for the purpose of the analysis.
//! - `mutbl`: mutability of the address `A`.
//! - `ty`: the type of data found at the address `A`.
//!
//! The resulting categorization tree differs somewhat from the expressions
//! themselves. For example, auto-derefs are explicit. Also, an index a[b] is
//! decomposed into two operations: a dereference to reach the array data and
//! then an index to jump forward to the relevant item.
//!
//! ## By-reference upvars
//!
//! One part of the codegen which may be non-obvious is that we translate
//! closure upvars into the dereference of a borrowed pointer; this more closely
//! resembles the runtime codegen. So, for example, if we had:
//!
//!     let mut x = 3;
//!     let y = 5;
//!     let inc = || x += y;
//!
//! Then when we categorize `x` (*within* the closure) we would yield a
//! result of `*x'`, effectively, where `x'` is a `Categorization::Upvar` reference
//! tied to `x`. The type of `x'` will be a borrowed pointer.

#![allow(non_camel_case_types)]

pub use self::PointerKind::*;
pub use self::InteriorKind::*;
pub use self::MutabilityCategory::*;
pub use self::AliasableReason::*;
pub use self::Note::*;

use self::Aliasability::*;

use crate::middle::region;
use crate::hir::def_id::{DefId, LocalDefId};
use crate::hir::Node;
use crate::infer::InferCtxt;
use crate::hir::def::{CtorOf, Res, DefKind, CtorKind};
use crate::ty::adjustment;
use crate::ty::{self, DefIdTree, Ty, TyCtxt};
use crate::ty::fold::TypeFoldable;
use crate::ty::layout::VariantIdx;

use crate::hir::{MutImmutable, MutMutable, PatKind};
use crate::hir::pat_util::EnumerateAndAdjustIterator;
use crate::hir;
use syntax::ast::{self, Name};
use syntax::symbol::sym;
use syntax_pos::Span;

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::indexed_vec::Idx;
use std::rc::Rc;
use crate::util::nodemap::ItemLocalSet;

#[derive(Clone, Debug, PartialEq)]
pub enum Categorization<'tcx> {
    Rvalue(ty::Region<'tcx>),            // temporary val, argument is its scope
    ThreadLocal(ty::Region<'tcx>),       // value that cannot move, but still restricted in scope
    StaticItem,
    Upvar(Upvar),                        // upvar referenced by closure env
    Local(hir::HirId),                   // local variable
    Deref(cmt<'tcx>, PointerKind<'tcx>), // deref of a ptr
    Interior(cmt<'tcx>, InteriorKind),   // something interior: field, tuple, etc
    Downcast(cmt<'tcx>, DefId),          // selects a particular enum variant (*1)

    // (*1) downcast is only required if the enum has more than one variant
}

// Represents any kind of upvar
#[derive(Clone, Copy, PartialEq)]
pub struct Upvar {
    pub id: ty::UpvarId,
    pub kind: ty::ClosureKind
}

// different kinds of pointers:
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerKind<'tcx> {
    /// `Box<T>`
    Unique,

    /// `&T`
    BorrowedPtr(ty::BorrowKind, ty::Region<'tcx>),

    /// `*T`
    UnsafePtr(hir::Mutability),
}

// We use the term "interior" to mean "something reachable from the
// base without a pointer dereference", e.g., a field
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteriorKind {
    InteriorField(FieldIndex),
    InteriorElement(InteriorOffsetKind),
}

// Contains index of a field that is actually used for loan path comparisons and
// string representation of the field that should be used only for diagnostics.
#[derive(Clone, Copy, Eq)]
pub struct FieldIndex(pub usize, pub Name);

impl PartialEq for FieldIndex {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl Hash for FieldIndex {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.0.hash(h)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InteriorOffsetKind {
    Index,   // e.g., `array_expr[index_expr]`
    Pattern, // e.g., `fn foo([_, a, _, _]: [A; 4]) { ... }`
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MutabilityCategory {
    McImmutable, // Immutable.
    McDeclared,  // Directly declared as mutable.
    McInherited, // Inherited from the fact that owner is mutable.
}

// A note about the provenance of a `cmt`.  This is used for
// special-case handling of upvars such as mutability inference.
// Upvar categorization can generate a variable number of nested
// derefs.  The note allows detecting them without deep pattern
// matching on the categorization.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Note {
    NoteClosureEnv(ty::UpvarId), // Deref through closure env
    NoteUpvarRef(ty::UpvarId),   // Deref through by-ref upvar
    NoteIndex,                   // Deref as part of desugaring `x[]` into its two components
    NoteNone                     // Nothing special
}

// `cmt`: "Category, Mutability, and Type".
//
// a complete categorization of a value indicating where it originated
// and how it is located, as well as the mutability of the memory in
// which the value is stored.
//
// *WARNING* The field `cmt.type` is NOT necessarily the same as the
// result of `node_type(cmt.id)`.
//
// (FIXME: rewrite the following comment given that `@x` managed
// pointers have been obsolete for quite some time.)
//
// This is because the `id` is always the `id` of the node producing the
// type; in an expression like `*x`, the type of this deref node is the
// deref'd type (`T`), but in a pattern like `@x`, the `@x` pattern is
// again a dereference, but its type is the type *before* the
// dereference (`@T`). So use `cmt.ty` to find the type of the value in
// a consistent fashion. For more details, see the method `cat_pattern`
#[derive(Clone, Debug, PartialEq)]
pub struct cmt_<'tcx> {
    pub hir_id: hir::HirId,        // HIR id of expr/pat producing this value
    pub span: Span,                // span of same expr/pat
    pub cat: Categorization<'tcx>, // categorization of expr
    pub mutbl: MutabilityCategory, // mutability of expr as place
    pub ty: Ty<'tcx>,              // type of the expr (*see WARNING above*)
    pub note: Note,                // Note about the provenance of this cmt
}

pub type cmt<'tcx> = Rc<cmt_<'tcx>>;

pub enum ImmutabilityBlame<'tcx> {
    ImmLocal(hir::HirId),
    ClosureEnv(LocalDefId),
    LocalDeref(hir::HirId),
    AdtFieldDeref(&'tcx ty::AdtDef, &'tcx ty::FieldDef)
}

impl<'tcx> cmt_<'tcx> {
    fn resolve_field(&self, field_index: usize) -> Option<(&'tcx ty::AdtDef, &'tcx ty::FieldDef)>
    {
        let adt_def = match self.ty.sty {
            ty::Adt(def, _) => def,
            ty::Tuple(..) => return None,
            // closures get `Categorization::Upvar` rather than `Categorization::Interior`
            _ =>  bug!("interior cmt {:?} is not an ADT", self)
        };
        let variant_def = match self.cat {
            Categorization::Downcast(_, variant_did) => {
                adt_def.variant_with_id(variant_did)
            }
            _ => {
                assert_eq!(adt_def.variants.len(), 1);
                &adt_def.variants[VariantIdx::new(0)]
            }
        };
        Some((adt_def, &variant_def.fields[field_index]))
    }

    pub fn immutability_blame(&self) -> Option<ImmutabilityBlame<'tcx>> {
        match self.cat {
            Categorization::Deref(ref base_cmt, BorrowedPtr(ty::ImmBorrow, _)) => {
                // try to figure out where the immutable reference came from
                match base_cmt.cat {
                    Categorization::Local(hir_id) =>
                        Some(ImmutabilityBlame::LocalDeref(hir_id)),
                    Categorization::Interior(ref base_cmt, InteriorField(field_index)) => {
                        base_cmt.resolve_field(field_index.0).map(|(adt_def, field_def)| {
                            ImmutabilityBlame::AdtFieldDeref(adt_def, field_def)
                        })
                    }
                    Categorization::Upvar(Upvar { id, .. }) => {
                        if let NoteClosureEnv(..) = self.note {
                            Some(ImmutabilityBlame::ClosureEnv(id.closure_expr_id))
                        } else {
                            None
                        }
                    }
                    _ => None
                }
            }
            Categorization::Local(hir_id) => {
                Some(ImmutabilityBlame::ImmLocal(hir_id))
            }
            Categorization::Rvalue(..) |
            Categorization::Upvar(..) |
            Categorization::Deref(_, UnsafePtr(..)) => {
                // This should not be reachable up to inference limitations.
                None
            }
            Categorization::Interior(ref base_cmt, _) |
            Categorization::Downcast(ref base_cmt, _) |
            Categorization::Deref(ref base_cmt, _) => {
                base_cmt.immutability_blame()
            }
            Categorization::ThreadLocal(..) |
            Categorization::StaticItem => {
                // Do we want to do something here?
                None
            }
        }
    }
}

pub trait HirNode {
    fn hir_id(&self) -> hir::HirId;
    fn span(&self) -> Span;
}

impl HirNode for hir::Expr {
    fn hir_id(&self) -> hir::HirId { self.hir_id }
    fn span(&self) -> Span { self.span }
}

impl HirNode for hir::Pat {
    fn hir_id(&self) -> hir::HirId { self.hir_id }
    fn span(&self) -> Span { self.span }
}

#[derive(Clone)]
pub struct MemCategorizationContext<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub body_owner: DefId,
    pub upvars: Option<&'tcx FxIndexMap<hir::HirId, hir::Upvar>>,
    pub region_scope_tree: &'a region::ScopeTree,
    pub tables: &'a ty::TypeckTables<'tcx>,
    rvalue_promotable_map: Option<&'tcx ItemLocalSet>,
    infcx: Option<&'a InferCtxt<'a, 'tcx>>,
}

pub type McResult<T> = Result<T, ()>;

impl MutabilityCategory {
    pub fn from_mutbl(m: hir::Mutability) -> MutabilityCategory {
        let ret = match m {
            MutImmutable => McImmutable,
            MutMutable => McDeclared
        };
        debug!("MutabilityCategory::{}({:?}) => {:?}",
               "from_mutbl", m, ret);
        ret
    }

    pub fn from_borrow_kind(borrow_kind: ty::BorrowKind) -> MutabilityCategory {
        let ret = match borrow_kind {
            ty::ImmBorrow => McImmutable,
            ty::UniqueImmBorrow => McImmutable,
            ty::MutBorrow => McDeclared,
        };
        debug!("MutabilityCategory::{}({:?}) => {:?}",
               "from_borrow_kind", borrow_kind, ret);
        ret
    }

    fn from_pointer_kind(base_mutbl: MutabilityCategory,
                         ptr: PointerKind<'_>) -> MutabilityCategory {
        let ret = match ptr {
            Unique => {
                base_mutbl.inherit()
            }
            BorrowedPtr(borrow_kind, _) => {
                MutabilityCategory::from_borrow_kind(borrow_kind)
            }
            UnsafePtr(m) => {
                MutabilityCategory::from_mutbl(m)
            }
        };
        debug!("MutabilityCategory::{}({:?}, {:?}) => {:?}",
               "from_pointer_kind", base_mutbl, ptr, ret);
        ret
    }

    fn from_local(
        tcx: TyCtxt<'_>,
        tables: &ty::TypeckTables<'_>,
        id: hir::HirId,
    ) -> MutabilityCategory {
        let ret = match tcx.hir().get(id) {
            Node::Binding(p) => match p.node {
                PatKind::Binding(..) => {
                    let bm = *tables.pat_binding_modes()
                                    .get(p.hir_id)
                                    .expect("missing binding mode");
                    if bm == ty::BindByValue(hir::MutMutable) {
                        McDeclared
                    } else {
                        McImmutable
                    }
                }
                _ => span_bug!(p.span, "expected identifier pattern")
            },
            _ => span_bug!(tcx.hir().span(id), "expected identifier pattern")
        };
        debug!("MutabilityCategory::{}(tcx, id={:?}) => {:?}",
               "from_local", id, ret);
        ret
    }

    pub fn inherit(&self) -> MutabilityCategory {
        let ret = match *self {
            McImmutable => McImmutable,
            McDeclared => McInherited,
            McInherited => McInherited,
        };
        debug!("{:?}.inherit() => {:?}", self, ret);
        ret
    }

    pub fn is_mutable(&self) -> bool {
        let ret = match *self {
            McImmutable => false,
            McInherited => true,
            McDeclared => true,
        };
        debug!("{:?}.is_mutable() => {:?}", self, ret);
        ret
    }

    pub fn is_immutable(&self) -> bool {
        let ret = match *self {
            McImmutable => true,
            McDeclared | McInherited => false
        };
        debug!("{:?}.is_immutable() => {:?}", self, ret);
        ret
    }

    pub fn to_user_str(&self) -> &'static str {
        match *self {
            McDeclared | McInherited => "mutable",
            McImmutable => "immutable",
        }
    }
}

impl<'a, 'tcx> MemCategorizationContext<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        body_owner: DefId,
        region_scope_tree: &'a region::ScopeTree,
        tables: &'a ty::TypeckTables<'tcx>,
        rvalue_promotable_map: Option<&'tcx ItemLocalSet>,
    ) -> MemCategorizationContext<'a, 'tcx> {
        MemCategorizationContext {
            tcx,
            body_owner,
            upvars: tcx.upvars(body_owner),
            region_scope_tree,
            tables,
            rvalue_promotable_map,
            infcx: None
        }
    }
}

impl<'a, 'tcx> MemCategorizationContext<'a, 'tcx> {
    /// Creates a `MemCategorizationContext` during type inference.
    /// This is used during upvar analysis and a few other places.
    /// Because the typeck tables are not yet complete, the results
    /// from the analysis must be used with caution:
    ///
    /// - rvalue promotions are not known, so the lifetimes of
    ///   temporaries may be overly conservative;
    /// - similarly, as the results of upvar analysis are not yet
    ///   known, the results around upvar accesses may be incorrect.
    pub fn with_infer(
        infcx: &'a InferCtxt<'a, 'tcx>,
        body_owner: DefId,
        region_scope_tree: &'a region::ScopeTree,
        tables: &'a ty::TypeckTables<'tcx>,
    ) -> MemCategorizationContext<'a, 'tcx> {
        let tcx = infcx.tcx;

        // Subtle: we can't do rvalue promotion analysis until the
        // typeck phase is complete, which means that you can't trust
        // the rvalue lifetimes that result, but that's ok, since we
        // don't need to know those during type inference.
        let rvalue_promotable_map = None;

        MemCategorizationContext {
            tcx,
            body_owner,
            upvars: tcx.upvars(body_owner),
            region_scope_tree,
            tables,
            rvalue_promotable_map,
            infcx: Some(infcx),
        }
    }

    pub fn type_is_copy_modulo_regions(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> bool {
        self.infcx.map(|infcx| infcx.type_is_copy_modulo_regions(param_env, ty, span))
            .or_else(|| {
                if (param_env, ty).has_local_value() {
                    None
                } else {
                    Some(ty.is_copy_modulo_regions(self.tcx, param_env, span))
                }
            })
            .unwrap_or(true)
    }

    fn resolve_vars_if_possible<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        self.infcx.map(|infcx| infcx.resolve_vars_if_possible(value))
            .unwrap_or_else(|| value.clone())
    }

    fn is_tainted_by_errors(&self) -> bool {
        self.infcx.map_or(false, |infcx| infcx.is_tainted_by_errors())
    }

    fn resolve_type_vars_or_error(&self,
                                  id: hir::HirId,
                                  ty: Option<Ty<'tcx>>)
                                  -> McResult<Ty<'tcx>> {
        match ty {
            Some(ty) => {
                let ty = self.resolve_vars_if_possible(&ty);
                if ty.references_error() || ty.is_ty_var() {
                    debug!("resolve_type_vars_or_error: error from {:?}", ty);
                    Err(())
                } else {
                    Ok(ty)
                }
            }
            // FIXME
            None if self.is_tainted_by_errors() => Err(()),
            None => {
                bug!("no type for node {}: {} in mem_categorization",
                     id, self.tcx.hir().node_to_string(id));
            }
        }
    }

    pub fn node_ty(&self,
                   hir_id: hir::HirId)
                   -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(hir_id,
                                        self.tables.node_type_opt(hir_id))
    }

    pub fn expr_ty(&self, expr: &hir::Expr) -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(expr.hir_id, self.tables.expr_ty_opt(expr))
    }

    pub fn expr_ty_adjusted(&self, expr: &hir::Expr) -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(expr.hir_id, self.tables.expr_ty_adjusted_opt(expr))
    }

    /// Returns the type of value that this pattern matches against.
    /// Some non-obvious cases:
    ///
    /// - a `ref x` binding matches against a value of type `T` and gives
    ///   `x` the type `&T`; we return `T`.
    /// - a pattern with implicit derefs (thanks to default binding
    ///   modes #42640) may look like `Some(x)` but in fact have
    ///   implicit deref patterns attached (e.g., it is really
    ///   `&Some(x)`). In that case, we return the "outermost" type
    ///   (e.g., `&Option<T>).
    pub fn pat_ty_adjusted(&self, pat: &hir::Pat) -> McResult<Ty<'tcx>> {
        // Check for implicit `&` types wrapping the pattern; note
        // that these are never attached to binding patterns, so
        // actually this is somewhat "disjoint" from the code below
        // that aims to account for `ref x`.
        if let Some(vec) = self.tables.pat_adjustments().get(pat.hir_id) {
            if let Some(first_ty) = vec.first() {
                debug!("pat_ty(pat={:?}) found adjusted ty `{:?}`", pat, first_ty);
                return Ok(first_ty);
            }
        }

        self.pat_ty_unadjusted(pat)
    }


    /// Like `pat_ty`, but ignores implicit `&` patterns.
    fn pat_ty_unadjusted(&self, pat: &hir::Pat) -> McResult<Ty<'tcx>> {
        let base_ty = self.node_ty(pat.hir_id)?;
        debug!("pat_ty(pat={:?}) base_ty={:?}", pat, base_ty);

        // This code detects whether we are looking at a `ref x`,
        // and if so, figures out what the type *being borrowed* is.
        let ret_ty = match pat.node {
            PatKind::Binding(..) => {
                let bm = *self.tables
                              .pat_binding_modes()
                              .get(pat.hir_id)
                              .expect("missing binding mode");

                if let ty::BindByReference(_) = bm {
                    // a bind-by-ref means that the base_ty will be the type of the ident itself,
                    // but what we want here is the type of the underlying value being borrowed.
                    // So peel off one-level, turning the &T into T.
                    match base_ty.builtin_deref(false) {
                        Some(t) => t.ty,
                        None => {
                            debug!("By-ref binding of non-derefable type {:?}", base_ty);
                            return Err(());
                        }
                    }
                } else {
                    base_ty
                }
            }
            _ => base_ty,
        };
        debug!("pat_ty(pat={:?}) ret_ty={:?}", pat, ret_ty);

        Ok(ret_ty)
    }

    pub fn cat_expr(&self, expr: &hir::Expr) -> McResult<cmt_<'tcx>> {
        // This recursion helper avoids going through *too many*
        // adjustments, since *only* non-overloaded deref recurses.
        fn helper<'a, 'tcx>(
            mc: &MemCategorizationContext<'a, 'tcx>,
            expr: &hir::Expr,
            adjustments: &[adjustment::Adjustment<'tcx>],
        ) -> McResult<cmt_<'tcx>> {
            match adjustments.split_last() {
                None => mc.cat_expr_unadjusted(expr),
                Some((adjustment, previous)) => {
                    mc.cat_expr_adjusted_with(expr, || helper(mc, expr, previous), adjustment)
                }
            }
        }

        helper(self, expr, self.tables.expr_adjustments(expr))
    }

    pub fn cat_expr_adjusted(&self, expr: &hir::Expr,
                             previous: cmt_<'tcx>,
                             adjustment: &adjustment::Adjustment<'tcx>)
                             -> McResult<cmt_<'tcx>> {
        self.cat_expr_adjusted_with(expr, || Ok(previous), adjustment)
    }

    fn cat_expr_adjusted_with<F>(&self, expr: &hir::Expr,
                                 previous: F,
                                 adjustment: &adjustment::Adjustment<'tcx>)
                                 -> McResult<cmt_<'tcx>>
        where F: FnOnce() -> McResult<cmt_<'tcx>>
    {
        debug!("cat_expr_adjusted_with({:?}): {:?}", adjustment, expr);
        let target = self.resolve_vars_if_possible(&adjustment.target);
        match adjustment.kind {
            adjustment::Adjust::Deref(overloaded) => {
                // Equivalent to *expr or something similar.
                let base = Rc::new(if let Some(deref) = overloaded {
                    let ref_ty = self.tcx.mk_ref(deref.region, ty::TypeAndMut {
                        ty: target,
                        mutbl: deref.mutbl,
                    });
                    self.cat_rvalue_node(expr.hir_id, expr.span, ref_ty)
                } else {
                    previous()?
                });
                self.cat_deref(expr, base, NoteNone)
            }

            adjustment::Adjust::NeverToAny |
            adjustment::Adjust::Pointer(_) |
            adjustment::Adjust::Borrow(_) => {
                // Result is an rvalue.
                Ok(self.cat_rvalue_node(expr.hir_id, expr.span, target))
            }
        }
    }

    pub fn cat_expr_unadjusted(&self, expr: &hir::Expr) -> McResult<cmt_<'tcx>> {
        debug!("cat_expr: id={} expr={:?}", expr.hir_id, expr);

        let expr_ty = self.expr_ty(expr)?;
        match expr.node {
            hir::ExprKind::Unary(hir::UnDeref, ref e_base) => {
                if self.tables.is_method_call(expr) {
                    self.cat_overloaded_place(expr, e_base, NoteNone)
                } else {
                    let base_cmt = Rc::new(self.cat_expr(&e_base)?);
                    self.cat_deref(expr, base_cmt, NoteNone)
                }
            }

            hir::ExprKind::Field(ref base, f_ident) => {
                let base_cmt = Rc::new(self.cat_expr(&base)?);
                debug!("cat_expr(cat_field): id={} expr={:?} base={:?}",
                       expr.hir_id,
                       expr,
                       base_cmt);
                let f_index = self.tcx.field_index(expr.hir_id, self.tables);
                Ok(self.cat_field(expr, base_cmt, f_index, f_ident, expr_ty))
            }

            hir::ExprKind::Index(ref base, _) => {
                if self.tables.is_method_call(expr) {
                    // If this is an index implemented by a method call, then it
                    // will include an implicit deref of the result.
                    // The call to index() returns a `&T` value, which
                    // is an rvalue. That is what we will be
                    // dereferencing.
                    self.cat_overloaded_place(expr, base, NoteIndex)
                } else {
                    let base_cmt = Rc::new(self.cat_expr(&base)?);
                    self.cat_index(expr, base_cmt, expr_ty, InteriorOffsetKind::Index)
                }
            }

            hir::ExprKind::Path(ref qpath) => {
                let res = self.tables.qpath_res(qpath, expr.hir_id);
                self.cat_res(expr.hir_id, expr.span, expr_ty, res)
            }

            hir::ExprKind::Type(ref e, _) => {
                self.cat_expr(&e)
            }

            hir::ExprKind::AddrOf(..) | hir::ExprKind::Call(..) |
            hir::ExprKind::Assign(..) | hir::ExprKind::AssignOp(..) |
            hir::ExprKind::Closure(..) | hir::ExprKind::Ret(..) |
            hir::ExprKind::Unary(..) | hir::ExprKind::Yield(..) |
            hir::ExprKind::MethodCall(..) | hir::ExprKind::Cast(..) | hir::ExprKind::DropTemps(..) |
            hir::ExprKind::Array(..) | hir::ExprKind::Tup(..) |
            hir::ExprKind::Binary(..) | hir::ExprKind::While(..) |
            hir::ExprKind::Block(..) | hir::ExprKind::Loop(..) | hir::ExprKind::Match(..) |
            hir::ExprKind::Lit(..) | hir::ExprKind::Break(..) |
            hir::ExprKind::Continue(..) | hir::ExprKind::Struct(..) | hir::ExprKind::Repeat(..) |
            hir::ExprKind::InlineAsm(..) | hir::ExprKind::Box(..) | hir::ExprKind::Err => {
                Ok(self.cat_rvalue_node(expr.hir_id, expr.span, expr_ty))
            }
        }
    }

    pub fn cat_res(&self,
                   hir_id: hir::HirId,
                   span: Span,
                   expr_ty: Ty<'tcx>,
                   res: Res)
                   -> McResult<cmt_<'tcx>> {
        debug!("cat_res: id={:?} expr={:?} def={:?}",
               hir_id, expr_ty, res);

        match res {
            Res::Def(DefKind::Ctor(..), _)
            | Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::ConstParam, _)
            | Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::Method, _)
            | Res::SelfCtor(..) => {
                Ok(self.cat_rvalue_node(hir_id, span, expr_ty))
            }

            Res::Def(DefKind::Static, def_id) => {
                // `#[thread_local]` statics may not outlive the current function, but
                // they also cannot be moved out of.
                let is_thread_local = self.tcx.get_attrs(def_id)[..]
                    .iter()
                    .any(|attr| attr.check_name(sym::thread_local));

                let cat = if is_thread_local {
                    let re = self.temporary_scope(hir_id.local_id);
                    Categorization::ThreadLocal(re)
                } else {
                    Categorization::StaticItem
                };

                Ok(cmt_ {
                    hir_id,
                    span,
                    cat,
                    mutbl: match self.tcx.static_mutability(def_id).unwrap() {
                        hir::MutImmutable => McImmutable,
                        hir::MutMutable => McDeclared,
                    },
                    ty:expr_ty,
                    note: NoteNone
                })
            }

            Res::Local(var_id) => {
                if self.upvars.map_or(false, |upvars| upvars.contains_key(&var_id)) {
                    self.cat_upvar(hir_id, span, var_id)
                } else {
                    Ok(cmt_ {
                        hir_id,
                        span,
                        cat: Categorization::Local(var_id),
                        mutbl: MutabilityCategory::from_local(self.tcx, self.tables, var_id),
                        ty: expr_ty,
                        note: NoteNone
                    })
                }
            }

            def => span_bug!(span, "unexpected definition in memory categorization: {:?}", def)
        }
    }

    // Categorize an upvar, complete with invisible derefs of closure
    // environment and upvar reference as appropriate.
    fn cat_upvar(
        &self,
        hir_id: hir::HirId,
        span: Span,
        var_id: hir::HirId,
    ) -> McResult<cmt_<'tcx>> {
        // An upvar can have up to 3 components. We translate first to a
        // `Categorization::Upvar`, which is itself a fiction -- it represents the reference to the
        // field from the environment.
        //
        // `Categorization::Upvar`.  Next, we add a deref through the implicit
        // environment pointer with an anonymous free region 'env and
        // appropriate borrow kind for closure kinds that take self by
        // reference.  Finally, if the upvar was captured
        // by-reference, we add a deref through that reference.  The
        // region of this reference is an inference variable 'up that
        // was previously generated and recorded in the upvar borrow
        // map.  The borrow kind bk is inferred by based on how the
        // upvar is used.
        //
        // This results in the following table for concrete closure
        // types:
        //
        //                | move                 | ref
        // ---------------+----------------------+-------------------------------
        // Fn             | copied -> &'env      | upvar -> &'env -> &'up bk
        // FnMut          | copied -> &'env mut  | upvar -> &'env mut -> &'up bk
        // FnOnce         | copied               | upvar -> &'up bk

        let closure_expr_def_id = self.body_owner;
        let fn_hir_id = self.tcx.hir().local_def_id_to_hir_id(
            LocalDefId::from_def_id(closure_expr_def_id),
        );
        let ty = self.node_ty(fn_hir_id)?;
        let kind = match ty.sty {
            ty::Generator(..) => ty::ClosureKind::FnOnce,
            ty::Closure(closure_def_id, closure_substs) => {
                match self.infcx {
                    // During upvar inference we may not know the
                    // closure kind, just use the LATTICE_BOTTOM value.
                    Some(infcx) =>
                        infcx.closure_kind(closure_def_id, closure_substs)
                             .unwrap_or(ty::ClosureKind::LATTICE_BOTTOM),

                    None =>
                        closure_substs.closure_kind(closure_def_id, self.tcx.global_tcx()),
                }
            }
            _ => span_bug!(span, "unexpected type for fn in mem_categorization: {:?}", ty),
        };

        let upvar_id = ty::UpvarId {
            var_path: ty::UpvarPath { hir_id: var_id },
            closure_expr_id: closure_expr_def_id.to_local(),
        };

        let var_ty = self.node_ty(var_id)?;

        // Mutability of original variable itself
        let var_mutbl = MutabilityCategory::from_local(self.tcx, self.tables, var_id);

        // Construct the upvar. This represents access to the field
        // from the environment (perhaps we should eventually desugar
        // this field further, but it will do for now).
        let cmt_result = cmt_ {
            hir_id,
            span,
            cat: Categorization::Upvar(Upvar {id: upvar_id, kind: kind}),
            mutbl: var_mutbl,
            ty: var_ty,
            note: NoteNone
        };

        // If this is a `FnMut` or `Fn` closure, then the above is
        // conceptually a `&mut` or `&` reference, so we have to add a
        // deref.
        let cmt_result = match kind {
            ty::ClosureKind::FnOnce => {
                cmt_result
            }
            ty::ClosureKind::FnMut => {
                self.env_deref(hir_id, span, upvar_id, var_mutbl, ty::MutBorrow, cmt_result)
            }
            ty::ClosureKind::Fn => {
                self.env_deref(hir_id, span, upvar_id, var_mutbl, ty::ImmBorrow, cmt_result)
            }
        };

        // If this is a by-ref capture, then the upvar we loaded is
        // actually a reference, so we have to add an implicit deref
        // for that.
        let upvar_capture = self.tables.upvar_capture(upvar_id);
        let cmt_result = match upvar_capture {
            ty::UpvarCapture::ByValue => {
                cmt_result
            }
            ty::UpvarCapture::ByRef(upvar_borrow) => {
                let ptr = BorrowedPtr(upvar_borrow.kind, upvar_borrow.region);
                cmt_ {
                    hir_id,
                    span,
                    cat: Categorization::Deref(Rc::new(cmt_result), ptr),
                    mutbl: MutabilityCategory::from_borrow_kind(upvar_borrow.kind),
                    ty: var_ty,
                    note: NoteUpvarRef(upvar_id)
                }
            }
        };

        let ret = cmt_result;
        debug!("cat_upvar ret={:?}", ret);
        Ok(ret)
    }

    fn env_deref(&self,
                 hir_id: hir::HirId,
                 span: Span,
                 upvar_id: ty::UpvarId,
                 upvar_mutbl: MutabilityCategory,
                 env_borrow_kind: ty::BorrowKind,
                 cmt_result: cmt_<'tcx>)
                 -> cmt_<'tcx>
    {
        // Region of environment pointer
        let env_region = self.tcx.mk_region(ty::ReFree(ty::FreeRegion {
            // The environment of a closure is guaranteed to
            // outlive any bindings introduced in the body of the
            // closure itself.
            scope: upvar_id.closure_expr_id.to_def_id(),
            bound_region: ty::BrEnv
        }));

        let env_ptr = BorrowedPtr(env_borrow_kind, env_region);

        let var_ty = cmt_result.ty;

        // We need to add the env deref.  This means
        // that the above is actually immutable and
        // has a ref type.  However, nothing should
        // actually look at the type, so we can get
        // away with stuffing a `Error` in there
        // instead of bothering to construct a proper
        // one.
        let cmt_result = cmt_ {
            mutbl: McImmutable,
            ty: self.tcx.types.err,
            ..cmt_result
        };

        let mut deref_mutbl = MutabilityCategory::from_borrow_kind(env_borrow_kind);

        // Issue #18335. If variable is declared as immutable, override the
        // mutability from the environment and substitute an `&T` anyway.
        match upvar_mutbl {
            McImmutable => { deref_mutbl = McImmutable; }
            McDeclared | McInherited => { }
        }

        let ret = cmt_ {
            hir_id,
            span,
            cat: Categorization::Deref(Rc::new(cmt_result), env_ptr),
            mutbl: deref_mutbl,
            ty: var_ty,
            note: NoteClosureEnv(upvar_id)
        };

        debug!("env_deref ret {:?}", ret);

        ret
    }

    /// Returns the lifetime of a temporary created by expr with id `id`.
    /// This could be `'static` if `id` is part of a constant expression.
    pub fn temporary_scope(&self, id: hir::ItemLocalId) -> ty::Region<'tcx> {
        let scope = self.region_scope_tree.temporary_scope(id);
        self.tcx.mk_region(match scope {
            Some(scope) => ty::ReScope(scope),
            None => ty::ReStatic
        })
    }

    pub fn cat_rvalue_node(&self,
                           hir_id: hir::HirId,
                           span: Span,
                           expr_ty: Ty<'tcx>)
                           -> cmt_<'tcx> {
        debug!("cat_rvalue_node(id={:?}, span={:?}, expr_ty={:?})",
               hir_id, span, expr_ty);

        let promotable = self.rvalue_promotable_map.as_ref().map(|m| m.contains(&hir_id.local_id))
                                                            .unwrap_or(false);

        debug!("cat_rvalue_node: promotable = {:?}", promotable);

        // Always promote `[T; 0]` (even when e.g., borrowed mutably).
        let promotable = match expr_ty.sty {
            ty::Array(_, len) if len.assert_usize(self.tcx) == Some(0) => true,
            _ => promotable,
        };

        debug!("cat_rvalue_node: promotable = {:?} (2)", promotable);

        // Compute maximum lifetime of this rvalue. This is 'static if
        // we can promote to a constant, otherwise equal to enclosing temp
        // lifetime.
        let re = if promotable {
            self.tcx.lifetimes.re_static
        } else {
            self.temporary_scope(hir_id.local_id)
        };
        let ret = self.cat_rvalue(hir_id, span, re, expr_ty);
        debug!("cat_rvalue_node ret {:?}", ret);
        ret
    }

    pub fn cat_rvalue(&self,
                      cmt_hir_id: hir::HirId,
                      span: Span,
                      temp_scope: ty::Region<'tcx>,
                      expr_ty: Ty<'tcx>) -> cmt_<'tcx> {
        let ret = cmt_ {
            hir_id: cmt_hir_id,
            span:span,
            cat:Categorization::Rvalue(temp_scope),
            mutbl:McDeclared,
            ty:expr_ty,
            note: NoteNone
        };
        debug!("cat_rvalue ret {:?}", ret);
        ret
    }

    pub fn cat_field<N: HirNode>(&self,
                                 node: &N,
                                 base_cmt: cmt<'tcx>,
                                 f_index: usize,
                                 f_ident: ast::Ident,
                                 f_ty: Ty<'tcx>)
                                 -> cmt_<'tcx> {
        let ret = cmt_ {
            hir_id: node.hir_id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: Categorization::Interior(base_cmt,
                                          InteriorField(FieldIndex(f_index, f_ident.name))),
            ty: f_ty,
            note: NoteNone
        };
        debug!("cat_field ret {:?}", ret);
        ret
    }

    fn cat_overloaded_place(
        &self,
        expr: &hir::Expr,
        base: &hir::Expr,
        note: Note,
    ) -> McResult<cmt_<'tcx>> {
        debug!("cat_overloaded_place(expr={:?}, base={:?}, note={:?})",
               expr,
               base,
               note);

        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
        let place_ty = self.expr_ty(expr)?;
        let base_ty = self.expr_ty_adjusted(base)?;

        let (region, mutbl) = match base_ty.sty {
            ty::Ref(region, _, mutbl) => (region, mutbl),
            _ => span_bug!(expr.span, "cat_overloaded_place: base is not a reference")
        };
        let ref_ty = self.tcx.mk_ref(region, ty::TypeAndMut {
            ty: place_ty,
            mutbl,
        });

        let base_cmt = Rc::new(self.cat_rvalue_node(expr.hir_id, expr.span, ref_ty));
        self.cat_deref(expr, base_cmt, note)
    }

    pub fn cat_deref(
        &self,
        node: &impl HirNode,
        base_cmt: cmt<'tcx>,
        note: Note,
    ) -> McResult<cmt_<'tcx>> {
        debug!("cat_deref: base_cmt={:?}", base_cmt);

        let base_cmt_ty = base_cmt.ty;
        let deref_ty = match base_cmt_ty.builtin_deref(true) {
            Some(mt) => mt.ty,
            None => {
                debug!("Explicit deref of non-derefable type: {:?}", base_cmt_ty);
                return Err(());
            }
        };

        let ptr = match base_cmt.ty.sty {
            ty::Adt(def, ..) if def.is_box() => Unique,
            ty::RawPtr(ref mt) => UnsafePtr(mt.mutbl),
            ty::Ref(r, _, mutbl) => {
                let bk = ty::BorrowKind::from_mutbl(mutbl);
                BorrowedPtr(bk, r)
            }
            _ => bug!("unexpected type in cat_deref: {:?}", base_cmt.ty)
        };
        let ret = cmt_ {
            hir_id: node.hir_id(),
            span: node.span(),
            // For unique ptrs, we inherit mutability from the owning reference.
            mutbl: MutabilityCategory::from_pointer_kind(base_cmt.mutbl, ptr),
            cat: Categorization::Deref(base_cmt, ptr),
            ty: deref_ty,
            note: note,
        };
        debug!("cat_deref ret {:?}", ret);
        Ok(ret)
    }

    fn cat_index<N: HirNode>(&self,
                             elt: &N,
                             base_cmt: cmt<'tcx>,
                             element_ty: Ty<'tcx>,
                             context: InteriorOffsetKind)
                             -> McResult<cmt_<'tcx>> {
        //! Creates a cmt for an indexing operation (`[]`).
        //!
        //! One subtle aspect of indexing that may not be
        //! immediately obvious: for anything other than a fixed-length
        //! vector, an operation like `x[y]` actually consists of two
        //! disjoint (from the point of view of borrowck) operations.
        //! The first is a deref of `x` to create a pointer `p` that points
        //! at the first element in the array. The second operation is
        //! an index which adds `y*sizeof(T)` to `p` to obtain the
        //! pointer to `x[y]`. `cat_index` will produce a resulting
        //! cmt containing both this deref and the indexing,
        //! presuming that `base_cmt` is not of fixed-length type.
        //!
        //! # Parameters
        //! - `elt`: the HIR node being indexed
        //! - `base_cmt`: the cmt of `elt`

        let interior_elem = InteriorElement(context);
        let ret = self.cat_imm_interior(elt, base_cmt, element_ty, interior_elem);
        debug!("cat_index ret {:?}", ret);
        return Ok(ret);
    }

    pub fn cat_imm_interior<N:HirNode>(&self,
                                       node: &N,
                                       base_cmt: cmt<'tcx>,
                                       interior_ty: Ty<'tcx>,
                                       interior: InteriorKind)
                                       -> cmt_<'tcx> {
        let ret = cmt_ {
            hir_id: node.hir_id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: Categorization::Interior(base_cmt, interior),
            ty: interior_ty,
            note: NoteNone
        };
        debug!("cat_imm_interior ret={:?}", ret);
        ret
    }

    pub fn cat_downcast_if_needed<N:HirNode>(&self,
                                             node: &N,
                                             base_cmt: cmt<'tcx>,
                                             variant_did: DefId)
                                             -> cmt<'tcx> {
        // univariant enums do not need downcasts
        let base_did = self.tcx.parent(variant_did).unwrap();
        if self.tcx.adt_def(base_did).variants.len() != 1 {
            let base_ty = base_cmt.ty;
            let ret = Rc::new(cmt_ {
                hir_id: node.hir_id(),
                span: node.span(),
                mutbl: base_cmt.mutbl.inherit(),
                cat: Categorization::Downcast(base_cmt, variant_did),
                ty: base_ty,
                note: NoteNone
            });
            debug!("cat_downcast ret={:?}", ret);
            ret
        } else {
            debug!("cat_downcast univariant={:?}", base_cmt);
            base_cmt
        }
    }

    pub fn cat_pattern<F>(&self, cmt: cmt<'tcx>, pat: &hir::Pat, mut op: F) -> McResult<()>
        where F: FnMut(cmt<'tcx>, &hir::Pat),
    {
        self.cat_pattern_(cmt, pat, &mut op)
    }

    // FIXME(#19596) This is a workaround, but there should be a better way to do this
    fn cat_pattern_<F>(&self, mut cmt: cmt<'tcx>, pat: &hir::Pat, op: &mut F) -> McResult<()>
        where F : FnMut(cmt<'tcx>, &hir::Pat)
    {
        // Here, `cmt` is the categorization for the value being
        // matched and pat is the pattern it is being matched against.
        //
        // In general, the way that this works is that we walk down
        // the pattern, constructing a cmt that represents the path
        // that will be taken to reach the value being matched.
        //
        // When we encounter named bindings, we take the cmt that has
        // been built up and pass it off to guarantee_valid() so that
        // we can be sure that the binding will remain valid for the
        // duration of the arm.
        //
        // (*2) There is subtlety concerning the correspondence between
        // pattern ids and types as compared to *expression* ids and
        // types. This is explained briefly. on the definition of the
        // type `cmt`, so go off and read what it says there, then
        // come back and I'll dive into a bit more detail here. :) OK,
        // back?
        //
        // In general, the id of the cmt should be the node that
        // "produces" the value---patterns aren't executable code
        // exactly, but I consider them to "execute" when they match a
        // value, and I consider them to produce the value that was
        // matched. So if you have something like:
        //
        // (FIXME: `@@3` is not legal code anymore!)
        //
        //     let x = @@3;
        //     match x {
        //       @@y { ... }
        //     }
        //
        // In this case, the cmt and the relevant ids would be:
        //
        //     CMT             Id                  Type of Id Type of cmt
        //
        //     local(x)->@->@
        //     ^~~~~~~^        `x` from discr      @@int      @@int
        //     ^~~~~~~~~~^     `@@y` pattern node  @@int      @int
        //     ^~~~~~~~~~~~~^  `@y` pattern node   @int       int
        //
        // You can see that the types of the id and the cmt are in
        // sync in the first line, because that id is actually the id
        // of an expression. But once we get to pattern ids, the types
        // step out of sync again. So you'll see below that we always
        // get the type of the *subpattern* and use that.

        debug!("cat_pattern(pat={:?}, cmt={:?})", pat, cmt);

        // If (pattern) adjustments are active for this pattern, adjust the `cmt` correspondingly.
        // `cmt`s are constructed differently from patterns. For example, in
        //
        // ```
        // match foo {
        //     &&Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the pattern `&&Some(x,)` is represented as `Ref { Ref { TupleStruct }}`. To build the
        // corresponding `cmt` we start with a `cmt` for `foo`, and then, by traversing the
        // pattern, try to answer the question: given the address of `foo`, how is `x` reached?
        //
        // `&&Some(x,)` `cmt_foo`
        //  `&Some(x,)` `deref { cmt_foo}`
        //   `Some(x,)` `deref { deref { cmt_foo }}`
        //        (x,)` `field0 { deref { deref { cmt_foo }}}` <- resulting cmt
        //
        // The above example has no adjustments. If the code were instead the (after adjustments,
        // equivalent) version
        //
        // ```
        // match foo {
        //     Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // Then we see that to get the same result, we must start with `deref { deref { cmt_foo }}`
        // instead of `cmt_foo` since the pattern is now `Some(x,)` and not `&&Some(x,)`, even
        // though its assigned type is that of `&&Some(x,)`.
        for _ in 0..self.tables
                        .pat_adjustments()
                        .get(pat.hir_id)
                        .map(|v| v.len())
                        .unwrap_or(0)
        {
            debug!("cat_pattern: applying adjustment to cmt={:?}", cmt);
            cmt = Rc::new(self.cat_deref(pat, cmt, NoteNone)?);
        }
        let cmt = cmt; // lose mutability
        debug!("cat_pattern: applied adjustment derefs to get cmt={:?}", cmt);

        // Invoke the callback, but only now, after the `cmt` has adjusted.
        //
        // To see that this makes sense, consider `match &Some(3) { Some(x) => { ... }}`. In that
        // case, the initial `cmt` will be that for `&Some(3)` and the pattern is `Some(x)`. We
        // don't want to call `op` with these incompatible values. As written, what happens instead
        // is that `op` is called with the adjusted cmt (that for `*&Some(3)`) and the pattern
        // `Some(x)` (which matches). Recursing once more, `*&Some(3)` and the pattern `Some(x)`
        // result in the cmt `Downcast<Some>(*&Some(3)).0` associated to `x` and invoke `op` with
        // that (where the `ref` on `x` is implied).
        op(cmt.clone(), pat);

        match pat.node {
            PatKind::TupleStruct(ref qpath, ref subpats, ddpos) => {
                let res = self.tables.qpath_res(qpath, pat.hir_id);
                let (cmt, expected_len) = match res {
                    Res::Err => {
                        debug!("access to unresolvable pattern {:?}", pat);
                        return Err(())
                    }
                    Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), variant_ctor_did) => {
                        let variant_did = self.tcx.parent(variant_ctor_did).unwrap();
                        let enum_did = self.tcx.parent(variant_did).unwrap();
                        (self.cat_downcast_if_needed(pat, cmt, variant_did),
                         self.tcx.adt_def(enum_did)
                             .variant_with_ctor_id(variant_ctor_did).fields.len())
                    }
                    Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), _)
                    | Res::SelfCtor(..) => {
                        let ty = self.pat_ty_unadjusted(&pat)?;
                        match ty.sty {
                            ty::Adt(adt_def, _) => {
                                (cmt, adt_def.non_enum_variant().fields.len())
                            }
                            _ => {
                                span_bug!(pat.span,
                                          "tuple struct pattern unexpected type {:?}", ty);
                            }
                        }
                    }
                    def => {
                        debug!(
                            "tuple struct pattern didn't resolve to variant or struct {:?} at {:?}",
                            def,
                            pat.span,
                        );
                        self.tcx.sess.delay_span_bug(pat.span, &format!(
                            "tuple struct pattern didn't resolve to variant or struct {:?}",
                            def,
                        ));
                        return Err(());
                    }
                };

                for (i, subpat) in subpats.iter().enumerate_and_adjust(expected_len, ddpos) {
                    let subpat_ty = self.pat_ty_adjusted(&subpat)?; // see (*2)
                    let interior = InteriorField(FieldIndex(i, sym::integer(i)));
                    let subcmt = Rc::new(
                        self.cat_imm_interior(pat, cmt.clone(), subpat_ty, interior));
                    self.cat_pattern_(subcmt, &subpat, op)?;
                }
            }

            PatKind::Struct(ref qpath, ref field_pats, _) => {
                // {f1: p1, ..., fN: pN}
                let res = self.tables.qpath_res(qpath, pat.hir_id);
                let cmt = match res {
                    Res::Err => {
                        debug!("access to unresolvable pattern {:?}", pat);
                        return Err(())
                    }
                    Res::Def(DefKind::Ctor(CtorOf::Variant, _), variant_ctor_did) => {
                        let variant_did = self.tcx.parent(variant_ctor_did).unwrap();
                        self.cat_downcast_if_needed(pat, cmt, variant_did)
                    }
                    Res::Def(DefKind::Variant, variant_did) => {
                        self.cat_downcast_if_needed(pat, cmt, variant_did)
                    }
                    _ => cmt,
                };

                for fp in field_pats {
                    let field_ty = self.pat_ty_adjusted(&fp.node.pat)?; // see (*2)
                    let f_index = self.tcx.field_index(fp.node.hir_id, self.tables);
                    let cmt_field = Rc::new(self.cat_field(pat, cmt.clone(), f_index,
                                                           fp.node.ident, field_ty));
                    self.cat_pattern_(cmt_field, &fp.node.pat, op)?;
                }
            }

            PatKind::Binding(.., Some(ref subpat)) => {
                self.cat_pattern_(cmt, &subpat, op)?;
            }

            PatKind::Tuple(ref subpats, ddpos) => {
                // (p1, ..., pN)
                let ty = self.pat_ty_unadjusted(&pat)?;
                let expected_len = match ty.sty {
                    ty::Tuple(ref tys) => tys.len(),
                    _ => span_bug!(pat.span, "tuple pattern unexpected type {:?}", ty),
                };
                for (i, subpat) in subpats.iter().enumerate_and_adjust(expected_len, ddpos) {
                    let subpat_ty = self.pat_ty_adjusted(&subpat)?; // see (*2)
                    let interior = InteriorField(FieldIndex(i, sym::integer(i)));
                    let subcmt = Rc::new(
                        self.cat_imm_interior(pat, cmt.clone(), subpat_ty, interior));
                    self.cat_pattern_(subcmt, &subpat, op)?;
                }
            }

            PatKind::Box(ref subpat) | PatKind::Ref(ref subpat, _) => {
                // box p1, &p1, &mut p1.  we can ignore the mutability of
                // PatKind::Ref since that information is already contained
                // in the type.
                let subcmt = Rc::new(self.cat_deref(pat, cmt, NoteNone)?);
                self.cat_pattern_(subcmt, &subpat, op)?;
            }

            PatKind::Slice(ref before, ref slice, ref after) => {
                let element_ty = match cmt.ty.builtin_index() {
                    Some(ty) => ty,
                    None => {
                        debug!("Explicit index of non-indexable type {:?}", cmt);
                        return Err(());
                    }
                };
                let context = InteriorOffsetKind::Pattern;
                let elt_cmt = Rc::new(self.cat_index(pat, cmt, element_ty, context)?);
                for before_pat in before {
                    self.cat_pattern_(elt_cmt.clone(), &before_pat, op)?;
                }
                if let Some(ref slice_pat) = *slice {
                    self.cat_pattern_(elt_cmt.clone(), &slice_pat, op)?;
                }
                for after_pat in after {
                    self.cat_pattern_(elt_cmt.clone(), &after_pat, op)?;
                }
            }

            PatKind::Path(_) | PatKind::Binding(.., None) |
            PatKind::Lit(..) | PatKind::Range(..) | PatKind::Wild => {
                // always ok
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Aliasability {
    FreelyAliasable(AliasableReason),
    NonAliasable,
    ImmutableUnique(Box<Aliasability>),
}

#[derive(Copy, Clone, Debug)]
pub enum AliasableReason {
    AliasableBorrowed,
    AliasableStatic,
    AliasableStaticMut,
}

impl<'tcx> cmt_<'tcx> {
    pub fn guarantor(&self) -> cmt_<'tcx> {
        //! Returns `self` after stripping away any derefs or
        //! interior content. The return value is basically the `cmt` which
        //! determines how long the value in `self` remains live.

        match self.cat {
            Categorization::Rvalue(..) |
            Categorization::StaticItem |
            Categorization::ThreadLocal(..) |
            Categorization::Local(..) |
            Categorization::Deref(_, UnsafePtr(..)) |
            Categorization::Deref(_, BorrowedPtr(..)) |
            Categorization::Upvar(..) => {
                (*self).clone()
            }
            Categorization::Downcast(ref b, _) |
            Categorization::Interior(ref b, _) |
            Categorization::Deref(ref b, Unique) => {
                b.guarantor()
            }
        }
    }

    /// Returns `FreelyAliasable(_)` if this place represents a freely aliasable pointer type.
    pub fn freely_aliasable(&self) -> Aliasability {
        // Maybe non-obvious: copied upvars can only be considered
        // non-aliasable in once closures, since any other kind can be
        // aliased and eventually recused.

        match self.cat {
            Categorization::Deref(ref b, BorrowedPtr(ty::MutBorrow, _)) |
            Categorization::Deref(ref b, BorrowedPtr(ty::UniqueImmBorrow, _)) |
            Categorization::Deref(ref b, Unique) |
            Categorization::Downcast(ref b, _) |
            Categorization::Interior(ref b, _) => {
                // Aliasability depends on base cmt
                b.freely_aliasable()
            }

            Categorization::Rvalue(..) |
            Categorization::ThreadLocal(..) |
            Categorization::Local(..) |
            Categorization::Upvar(..) |
            Categorization::Deref(_, UnsafePtr(..)) => { // yes, it's aliasable, but...
                NonAliasable
            }

            Categorization::StaticItem => {
                if self.mutbl.is_mutable() {
                    FreelyAliasable(AliasableStaticMut)
                } else {
                    FreelyAliasable(AliasableStatic)
                }
            }

            Categorization::Deref(_, BorrowedPtr(ty::ImmBorrow, _)) => {
                FreelyAliasable(AliasableBorrowed)
            }
        }
    }

    // Digs down through one or two layers of deref and grabs the
    // Categorization of the cmt for the upvar if a note indicates there is
    // one.
    pub fn upvar_cat(&self) -> Option<&Categorization<'tcx>> {
        match self.note {
            NoteClosureEnv(..) | NoteUpvarRef(..) => {
                Some(match self.cat {
                    Categorization::Deref(ref inner, _) => {
                        match inner.cat {
                            Categorization::Deref(ref inner, _) => &inner.cat,
                            Categorization::Upvar(..) => &inner.cat,
                            _ => bug!()
                        }
                    }
                    _ => bug!()
                })
            }
            NoteIndex | NoteNone => None
        }
    }

    pub fn descriptive_string(&self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
        match self.cat {
            Categorization::StaticItem => {
                "static item".into()
            }
            Categorization::ThreadLocal(..) => {
                "thread-local static item".into()
            }
            Categorization::Rvalue(..) => {
                "non-place".into()
            }
            Categorization::Local(vid) => {
                if tcx.hir().is_argument(vid) {
                    "argument"
                } else {
                    "local variable"
                }.into()
            }
            Categorization::Deref(_, pk) => {
                match self.upvar_cat() {
                    Some(&Categorization::Upvar(ref var)) => {
                        var.to_string().into()
                    }
                    Some(_) => bug!(),
                    None => {
                        match pk {
                            Unique => {
                                "`Box` content"
                            }
                            UnsafePtr(..) => {
                                "dereference of raw pointer"
                            }
                            BorrowedPtr(..) => {
                                match self.note {
                                    NoteIndex => "indexed content",
                                    _ => "borrowed content"
                                }
                            }
                        }.into()
                    }
                }
            }
            Categorization::Interior(_, InteriorField(..)) => {
                "field".into()
            }
            Categorization::Interior(_, InteriorElement(InteriorOffsetKind::Index)) => {
                "indexed content".into()
            }
            Categorization::Interior(_, InteriorElement(InteriorOffsetKind::Pattern)) => {
                "pattern-bound indexed content".into()
            }
            Categorization::Upvar(ref var) => {
                var.to_string().into()
            }
            Categorization::Downcast(ref cmt, _) => {
                cmt.descriptive_string(tcx).into()
            }
        }
    }
}

pub fn ptr_sigil(ptr: PointerKind<'_>) -> &'static str {
    match ptr {
        Unique => "Box",
        BorrowedPtr(ty::ImmBorrow, _) => "&",
        BorrowedPtr(ty::MutBorrow, _) => "&mut",
        BorrowedPtr(ty::UniqueImmBorrow, _) => "&unique",
        UnsafePtr(_) => "*",
    }
}

impl fmt::Debug for InteriorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            InteriorField(FieldIndex(_, info)) => write!(f, "{}", info),
            InteriorElement(..) => write!(f, "[]"),
        }
    }
}

impl fmt::Debug for Upvar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}/{:?}", self.id, self.kind)
    }
}

impl fmt::Display for Upvar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ty::ClosureKind::Fn => "Fn",
            ty::ClosureKind::FnMut => "FnMut",
            ty::ClosureKind::FnOnce => "FnOnce",
        };
        write!(f, "captured outer variable in an `{}` closure", kind)
    }
}
