// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

pub use self::terr_vstore_kind::*;
pub use self::type_err::*;
pub use self::BuiltinBound::*;
pub use self::InferTy::*;
pub use self::InferRegion::*;
pub use self::ImplOrTraitItemId::*;
pub use self::ClosureKind::*;
pub use self::ast_ty_to_ty_cache_entry::*;
pub use self::Variance::*;
pub use self::AutoAdjustment::*;
pub use self::Representability::*;
pub use self::UnsizeKind::*;
pub use self::AutoRef::*;
pub use self::ExprKind::*;
pub use self::DtorKind::*;
pub use self::ExplicitSelfCategory::*;
pub use self::FnOutput::*;
pub use self::Region::*;
pub use self::ImplOrTraitItemContainer::*;
pub use self::BorrowKind::*;
pub use self::ImplOrTraitItem::*;
pub use self::BoundRegion::*;
pub use self::sty::*;
pub use self::IntVarValue::*;
pub use self::ExprAdjustment::*;
pub use self::vtable_origin::*;
pub use self::MethodOrigin::*;
pub use self::CopyImplementationError::*;

use back::svh::Svh;
use session::Session;
use lint;
use metadata::csearch;
use middle;
use middle::check_const;
use middle::const_eval;
use middle::def::{self, DefMap, ExportMap};
use middle::dependency_format;
use middle::lang_items::{FnTraitLangItem, FnMutTraitLangItem};
use middle::lang_items::{FnOnceTraitLangItem, TyDescStructLangItem};
use middle::mem_categorization as mc;
use middle::region;
use middle::resolve_lifetime;
use middle::infer;
use middle::stability;
use middle::subst::{self, Subst, Substs, VecPerParamSpace};
use middle::traits;
use middle::ty;
use middle::ty_fold::{self, TypeFoldable, TypeFolder};
use middle::ty_walk::TypeWalker;
use util::ppaux::{note_and_explain_region, bound_region_ptr_to_string};
use util::ppaux::ty_to_string;
use util::ppaux::{Repr, UserString};
use util::common::{memoized, ErrorReported};
use util::nodemap::{NodeMap, NodeSet, DefIdMap, DefIdSet};
use util::nodemap::{FnvHashMap};

use arena::TypedArena;
use std::borrow::{Borrow, Cow};
use std::cell::{Cell, RefCell};
use std::cmp;
use std::fmt;
use std::hash::{Hash, SipHasher, Hasher};
#[cfg(stage0)] use std::hash::Writer;
use std::mem;
use std::ops;
use std::rc::Rc;
use std::vec::{CowVec, IntoIter};
use collections::enum_set::{EnumSet, CLike};
use std::collections::{HashMap, HashSet};
use syntax::abi;
use syntax::ast::{CrateNum, DefId, Ident, ItemTrait, LOCAL_CRATE};
use syntax::ast::{MutImmutable, MutMutable, Name, NamedField, NodeId};
use syntax::ast::{StmtExpr, StmtSemi, StructField, UnnamedField, Visibility};
use syntax::ast_util::{self, is_local, lit_is_str, local_def, PostExpansionMethod};
use syntax::attr::{self, AttrMetaMethods};
use syntax::codemap::Span;
use syntax::parse::token::{self, InternedString, special_idents};
use syntax::{ast, ast_map};

pub type Disr = u64;

pub const INITIAL_DISCRIMINANT_VALUE: Disr = 0;

// Data types

/// The complete set of all analyses described in this module. This is
/// produced by the driver and fed to trans and later passes.
pub struct CrateAnalysis<'tcx> {
    pub export_map: ExportMap,
    pub exported_items: middle::privacy::ExportedItems,
    pub public_items: middle::privacy::PublicItems,
    pub ty_cx: ty::ctxt<'tcx>,
    pub reachable: NodeSet,
    pub name: String,
    pub glob_map: Option<GlobMap>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct field<'tcx> {
    pub name: ast::Name,
    pub mt: mt<'tcx>
}

#[derive(Clone, Copy, Debug)]
pub enum ImplOrTraitItemContainer {
    TraitContainer(ast::DefId),
    ImplContainer(ast::DefId),
}

impl ImplOrTraitItemContainer {
    pub fn id(&self) -> ast::DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ImplOrTraitItem<'tcx> {
    MethodTraitItem(Rc<Method<'tcx>>),
    TypeTraitItem(Rc<AssociatedType>),
}

impl<'tcx> ImplOrTraitItem<'tcx> {
    fn id(&self) -> ImplOrTraitItemId {
        match *self {
            MethodTraitItem(ref method) => MethodTraitItemId(method.def_id),
            TypeTraitItem(ref associated_type) => {
                TypeTraitItemId(associated_type.def_id)
            }
        }
    }

    pub fn def_id(&self) -> ast::DefId {
        match *self {
            MethodTraitItem(ref method) => method.def_id,
            TypeTraitItem(ref associated_type) => associated_type.def_id,
        }
    }

    pub fn name(&self) -> ast::Name {
        match *self {
            MethodTraitItem(ref method) => method.name,
            TypeTraitItem(ref associated_type) => associated_type.name,
        }
    }

    pub fn container(&self) -> ImplOrTraitItemContainer {
        match *self {
            MethodTraitItem(ref method) => method.container,
            TypeTraitItem(ref associated_type) => associated_type.container,
        }
    }

    pub fn as_opt_method(&self) -> Option<Rc<Method<'tcx>>> {
        match *self {
            MethodTraitItem(ref m) => Some((*m).clone()),
            TypeTraitItem(_) => None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ImplOrTraitItemId {
    MethodTraitItemId(ast::DefId),
    TypeTraitItemId(ast::DefId),
}

impl ImplOrTraitItemId {
    pub fn def_id(&self) -> ast::DefId {
        match *self {
            MethodTraitItemId(def_id) => def_id,
            TypeTraitItemId(def_id) => def_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Method<'tcx> {
    pub name: ast::Name,
    pub generics: Generics<'tcx>,
    pub predicates: GenericPredicates<'tcx>,
    pub fty: BareFnTy<'tcx>,
    pub explicit_self: ExplicitSelfCategory,
    pub vis: ast::Visibility,
    pub def_id: ast::DefId,
    pub container: ImplOrTraitItemContainer,

    // If this method is provided, we need to know where it came from
    pub provided_source: Option<ast::DefId>
}

impl<'tcx> Method<'tcx> {
    pub fn new(name: ast::Name,
               generics: ty::Generics<'tcx>,
               predicates: GenericPredicates<'tcx>,
               fty: BareFnTy<'tcx>,
               explicit_self: ExplicitSelfCategory,
               vis: ast::Visibility,
               def_id: ast::DefId,
               container: ImplOrTraitItemContainer,
               provided_source: Option<ast::DefId>)
               -> Method<'tcx> {
       Method {
            name: name,
            generics: generics,
            predicates: predicates,
            fty: fty,
            explicit_self: explicit_self,
            vis: vis,
            def_id: def_id,
            container: container,
            provided_source: provided_source
        }
    }

    pub fn container_id(&self) -> ast::DefId {
        match self.container {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AssociatedType {
    pub name: ast::Name,
    pub vis: ast::Visibility,
    pub def_id: ast::DefId,
    pub container: ImplOrTraitItemContainer,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct mt<'tcx> {
    pub ty: Ty<'tcx>,
    pub mutbl: ast::Mutability,
}

#[derive(Clone, Copy, Debug)]
pub struct field_ty {
    pub name: Name,
    pub id: DefId,
    pub vis: ast::Visibility,
    pub origin: ast::DefId,  // The DefId of the struct in which the field is declared.
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, PartialEq, Eq, Hash)]
pub struct creader_cache_key {
    pub cnum: CrateNum,
    pub pos: uint,
    pub len: uint
}

#[derive(Copy)]
pub enum ast_ty_to_ty_cache_entry<'tcx> {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(Ty<'tcx>)  /* resolved to a type, irrespective of region */
}

#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable)]
pub struct ItemVariances {
    pub types: VecPerParamSpace<Variance>,
    pub regions: VecPerParamSpace<Variance>,
}

#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable, Debug, Copy)]
pub enum Variance {
    Covariant,      // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,      // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant,  // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,      // T<A> <: T<B>            -- e.g., unused type parameter
}

#[derive(Clone, Debug)]
pub enum AutoAdjustment<'tcx> {
    AdjustReifyFnPointer(ast::DefId), // go from a fn-item type to a fn-pointer type
    AdjustDerefRef(AutoDerefRef<'tcx>)
}

#[derive(Clone, PartialEq, Debug)]
pub enum UnsizeKind<'tcx> {
    // [T, ..n] -> [T], the uint field is n.
    UnsizeLength(uint),
    // An unsize coercion applied to the tail field of a struct.
    // The uint is the index of the type parameter which is unsized.
    UnsizeStruct(Box<UnsizeKind<'tcx>>, uint),
    UnsizeVtable(TyTrait<'tcx>, /* the self type of the trait */ Ty<'tcx>)
}

#[derive(Clone, Debug)]
pub struct AutoDerefRef<'tcx> {
    pub autoderefs: uint,
    pub autoref: Option<AutoRef<'tcx>>
}

#[derive(Clone, PartialEq, Debug)]
pub enum AutoRef<'tcx> {
    /// Convert from T to &T
    /// The third field allows us to wrap other AutoRef adjustments.
    AutoPtr(Region, ast::Mutability, Option<Box<AutoRef<'tcx>>>),

    /// Convert [T, ..n] to [T] (or similar, depending on the kind)
    AutoUnsize(UnsizeKind<'tcx>),

    /// Convert Box<[T, ..n]> to Box<[T]> or something similar in a Box.
    /// With DST and Box a library type, this should be replaced by UnsizeStruct.
    AutoUnsizeUniq(UnsizeKind<'tcx>),

    /// Convert from T to *T
    /// Value to thin pointer
    /// The second field allows us to wrap other AutoRef adjustments.
    AutoUnsafe(ast::Mutability, Option<Box<AutoRef<'tcx>>>),
}

// Ugly little helper function. The first bool in the returned tuple is true if
// there is an 'unsize to trait object' adjustment at the bottom of the
// adjustment. If that is surrounded by an AutoPtr, then we also return the
// region of the AutoPtr (in the third argument). The second bool is true if the
// adjustment is unique.
fn autoref_object_region(autoref: &AutoRef) -> (bool, bool, Option<Region>) {
    fn unsize_kind_is_object(k: &UnsizeKind) -> bool {
        match k {
            &UnsizeVtable(..) => true,
            &UnsizeStruct(box ref k, _) => unsize_kind_is_object(k),
            _ => false
        }
    }

    match autoref {
        &AutoUnsize(ref k) => (unsize_kind_is_object(k), false, None),
        &AutoUnsizeUniq(ref k) => (unsize_kind_is_object(k), true, None),
        &AutoPtr(adj_r, _, Some(box ref autoref)) => {
            let (b, u, r) = autoref_object_region(autoref);
            if r.is_some() || u {
                (b, u, r)
            } else {
                (b, u, Some(adj_r))
            }
        }
        &AutoUnsafe(_, Some(box ref autoref)) => autoref_object_region(autoref),
        _ => (false, false, None)
    }
}

// If the adjustment introduces a borrowed reference to a trait object, then
// returns the region of the borrowed reference.
pub fn adjusted_object_region(adj: &AutoAdjustment) -> Option<Region> {
    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            let (b, _, r) = autoref_object_region(autoref);
            if b {
                r
            } else {
                None
            }
        }
        _ => None
    }
}

// Returns true if there is a trait cast at the bottom of the adjustment.
pub fn adjust_is_object(adj: &AutoAdjustment) -> bool {
    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            let (b, _, _) = autoref_object_region(autoref);
            b
        }
        _ => false
    }
}

// If possible, returns the type expected from the given adjustment. This is not
// possible if the adjustment depends on the type of the adjusted expression.
pub fn type_of_adjust<'tcx>(cx: &ctxt<'tcx>, adj: &AutoAdjustment<'tcx>) -> Option<Ty<'tcx>> {
    fn type_of_autoref<'tcx>(cx: &ctxt<'tcx>, autoref: &AutoRef<'tcx>) -> Option<Ty<'tcx>> {
        match autoref {
            &AutoUnsize(ref k) => match k {
                &UnsizeVtable(TyTrait { ref principal, ref bounds }, _) => {
                    Some(mk_trait(cx, principal.clone(), bounds.clone()))
                }
                _ => None
            },
            &AutoUnsizeUniq(ref k) => match k {
                &UnsizeVtable(TyTrait { ref principal, ref bounds }, _) => {
                    Some(mk_uniq(cx, mk_trait(cx, principal.clone(), bounds.clone())))
                }
                _ => None
            },
            &AutoPtr(r, m, Some(box ref autoref)) => {
                match type_of_autoref(cx, autoref) {
                    Some(ty) => Some(mk_rptr(cx, cx.mk_region(r), mt {mutbl: m, ty: ty})),
                    None => None
                }
            }
            &AutoUnsafe(m, Some(box ref autoref)) => {
                match type_of_autoref(cx, autoref) {
                    Some(ty) => Some(mk_ptr(cx, mt {mutbl: m, ty: ty})),
                    None => None
                }
            }
            _ => None
        }
    }

    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            type_of_autoref(cx, autoref)
        }
        _ => None
    }
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Debug)]
pub struct param_index {
    pub space: subst::ParamSpace,
    pub index: uint
}

#[derive(Clone, Debug)]
pub enum MethodOrigin<'tcx> {
    // fully statically resolved method
    MethodStatic(ast::DefId),

    // fully statically resolved closure invocation
    MethodStaticClosure(ast::DefId),

    // method invoked on a type parameter with a bounded trait
    MethodTypeParam(MethodParam<'tcx>),

    // method invoked on a trait instance
    MethodTraitObject(MethodObject<'tcx>),

}

// details for a method invoked with a receiver whose type is a type parameter
// with a bounded trait.
#[derive(Clone, Debug)]
pub struct MethodParam<'tcx> {
    // the precise trait reference that occurs as a bound -- this may
    // be a supertrait of what the user actually typed. Note that it
    // never contains bound regions; those regions should have been
    // instantiated with fresh variables at this point.
    pub trait_ref: Rc<ty::TraitRef<'tcx>>,

    // index of uint in the list of trait items. Note that this is NOT
    // the index into the vtable, because the list of trait items
    // includes associated types.
    pub method_num: uint,

    /// The impl for the trait from which the method comes. This
    /// should only be used for certain linting/heuristic purposes
    /// since there is no guarantee that this is Some in every
    /// situation that it could/should be.
    pub impl_def_id: Option<ast::DefId>,
}

// details for a method invoked with a receiver whose type is an object
#[derive(Clone, Debug)]
pub struct MethodObject<'tcx> {
    // the (super)trait containing the method to be invoked
    pub trait_ref: Rc<ty::TraitRef<'tcx>>,

    // the actual base trait id of the object
    pub object_trait_id: ast::DefId,

    // index of the method to be invoked amongst the trait's items
    pub method_num: uint,

    // index into the actual runtime vtable.
    // the vtable is formed by concatenating together the method lists of
    // the base object trait and all supertraits; this is the index into
    // that vtable
    pub vtable_index: uint,
}

#[derive(Clone)]
pub struct MethodCallee<'tcx> {
    pub origin: MethodOrigin<'tcx>,
    pub ty: Ty<'tcx>,
    pub substs: subst::Substs<'tcx>
}

/// With method calls, we store some extra information in
/// side tables (i.e method_map). We use
/// MethodCall as a key to index into these tables instead of
/// just directly using the expression's NodeId. The reason
/// for this being that we may apply adjustments (coercions)
/// with the resulting expression also needing to use the
/// side tables. The problem with this is that we don't
/// assign a separate NodeId to this new expression
/// and so it would clash with the base expression if both
/// needed to add to the side tables. Thus to disambiguate
/// we also keep track of whether there's an adjustment in
/// our key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MethodCall {
    pub expr_id: ast::NodeId,
    pub adjustment: ExprAdjustment
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable, Copy)]
pub enum ExprAdjustment {
    NoAdjustment,
    AutoDeref(uint),
    AutoObject
}

impl MethodCall {
    pub fn expr(id: ast::NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            adjustment: NoAdjustment
        }
    }

    pub fn autoobject(id: ast::NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            adjustment: AutoObject
        }
    }

    pub fn autoderef(expr_id: ast::NodeId, autoderef: uint) -> MethodCall {
        MethodCall {
            expr_id: expr_id,
            adjustment: AutoDeref(1 + autoderef)
        }
    }
}

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
pub type MethodMap<'tcx> = RefCell<FnvHashMap<MethodCall, MethodCallee<'tcx>>>;

pub type vtable_param_res<'tcx> = Vec<vtable_origin<'tcx>>;

// Resolutions for bounds of all parameters, left to right, for a given path.
pub type vtable_res<'tcx> = VecPerParamSpace<vtable_param_res<'tcx>>;

#[derive(Clone)]
pub enum vtable_origin<'tcx> {
    /*
      Statically known vtable. def_id gives the impl item
      from whence comes the vtable, and tys are the type substs.
      vtable_res is the vtable itself.
     */
    vtable_static(ast::DefId, subst::Substs<'tcx>, vtable_res<'tcx>),

    /*
      Dynamic vtable, comes from a parameter that has a bound on it:
      fn foo<T:quux,baz,bar>(a: T) -- a's vtable would have a
      vtable_param origin

      The first argument is the param index (identifying T in the example),
      and the second is the bound number (identifying baz)
     */
    vtable_param(param_index, uint),

    /*
      Vtable automatically generated for a closure. The def ID is the
      ID of the closure expression.
     */
    vtable_closure(ast::DefId),

    /*
      Asked to determine the vtable for ty_err. This is the value used
      for the vtables of `Self` in a virtual call like `foo.bar()`
      where `foo` is of object type. The same value is also used when
      type errors occur.
     */
    vtable_error,
}


// For every explicit cast into an object type, maps from the cast
// expr to the associated trait ref.
pub type ObjectCastMap<'tcx> = RefCell<NodeMap<ty::PolyTraitRef<'tcx>>>;

/// A restriction that certain types must be the same size. The use of
/// `transmute` gives rise to these restrictions. These generally
/// cannot be checked until trans; therefore, each call to `transmute`
/// will push one or more such restriction into the
/// `transmute_restrictions` vector during `intrinsicck`. They are
/// then checked during `trans` by the fn `check_intrinsics`.
#[derive(Copy)]
pub struct TransmuteRestriction<'tcx> {
    /// The span whence the restriction comes.
    pub span: Span,

    /// The type being transmuted from.
    pub original_from: Ty<'tcx>,

    /// The type being transmuted to.
    pub original_to: Ty<'tcx>,

    /// The type being transmuted from, with all type parameters
    /// substituted for an arbitrary representative. Not to be shown
    /// to the end user.
    pub substituted_from: Ty<'tcx>,

    /// The type being transmuted to, with all type parameters
    /// substituted for an arbitrary representative. Not to be shown
    /// to the end user.
    pub substituted_to: Ty<'tcx>,

    /// NodeId of the transmute intrinsic.
    pub id: ast::NodeId,
}

/// Internal storage
pub struct CtxtArenas<'tcx> {
    type_: TypedArena<TyS<'tcx>>,
    substs: TypedArena<Substs<'tcx>>,
    bare_fn: TypedArena<BareFnTy<'tcx>>,
    region: TypedArena<Region>,
}

impl<'tcx> CtxtArenas<'tcx> {
    pub fn new() -> CtxtArenas<'tcx> {
        CtxtArenas {
            type_: TypedArena::new(),
            substs: TypedArena::new(),
            bare_fn: TypedArena::new(),
            region: TypedArena::new(),
        }
    }
}

pub struct CommonTypes<'tcx> {
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub int: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub uint: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub err: Ty<'tcx>,
}

/// The data structure to keep track of all the information that typechecker
/// generates so that so that it can be reused and doesn't have to be redone
/// later on.
pub struct ctxt<'tcx> {
    /// The arenas that types etc are allocated from.
    arenas: &'tcx CtxtArenas<'tcx>,

    /// Specifically use a speedy hash algorithm for this hash map, it's used
    /// quite often.
    // FIXME(eddyb) use a FnvHashSet<InternedTy<'tcx>> when equivalent keys can
    // queried from a HashSet.
    interner: RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>,

    // FIXME as above, use a hashset if equivalent elements can be queried.
    substs_interner: RefCell<FnvHashMap<&'tcx Substs<'tcx>, &'tcx Substs<'tcx>>>,
    bare_fn_interner: RefCell<FnvHashMap<&'tcx BareFnTy<'tcx>, &'tcx BareFnTy<'tcx>>>,
    region_interner: RefCell<FnvHashMap<&'tcx Region, &'tcx Region>>,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    pub sess: Session,
    pub def_map: DefMap,

    pub named_region_map: resolve_lifetime::NamedRegionMap,

    pub region_maps: middle::region::RegionMaps,

    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    pub node_types: RefCell<NodeMap<Ty<'tcx>>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    pub item_substs: RefCell<NodeMap<ItemSubsts<'tcx>>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub impl_or_trait_items: RefCell<DefIdMap<ImplOrTraitItem<'tcx>>>,

    /// Maps from a trait def-id to a list of the def-ids of its trait items
    pub trait_item_def_ids: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItemId>>>>,

    /// A cache for the trait_items() routine
    pub trait_items_cache: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItem<'tcx>>>>>,

    pub impl_trait_cache: RefCell<DefIdMap<Option<Rc<ty::TraitRef<'tcx>>>>>,

    pub trait_refs: RefCell<NodeMap<Rc<TraitRef<'tcx>>>>,
    pub trait_defs: RefCell<DefIdMap<Rc<TraitDef<'tcx>>>>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated predicates.
    pub predicates: RefCell<DefIdMap<GenericPredicates<'tcx>>>,

    /// Maps from node-id of a trait object cast (like `foo as
    /// Box<Trait>`) to the trait reference.
    pub object_cast_map: ObjectCastMap<'tcx>,

    pub map: ast_map::Map<'tcx>,
    pub intrinsic_defs: RefCell<DefIdMap<Ty<'tcx>>>,
    pub freevars: RefCell<FreevarMap>,
    pub tcache: RefCell<DefIdMap<TypeScheme<'tcx>>>,
    pub rcache: RefCell<FnvHashMap<creader_cache_key, Ty<'tcx>>>,
    pub short_names_cache: RefCell<FnvHashMap<Ty<'tcx>, String>>,
    pub tc_cache: RefCell<FnvHashMap<Ty<'tcx>, TypeContents>>,
    pub ast_ty_to_ty_cache: RefCell<NodeMap<ast_ty_to_ty_cache_entry<'tcx>>>,
    pub enum_var_cache: RefCell<DefIdMap<Rc<Vec<Rc<VariantInfo<'tcx>>>>>>,
    pub ty_param_defs: RefCell<NodeMap<TypeParameterDef<'tcx>>>,
    pub adjustments: RefCell<NodeMap<AutoAdjustment<'tcx>>>,
    pub normalized_cache: RefCell<FnvHashMap<Ty<'tcx>, Ty<'tcx>>>,
    pub lang_items: middle::lang_items::LanguageItems,
    /// A mapping of fake provided method def_ids to the default implementation
    pub provided_method_sources: RefCell<DefIdMap<ast::DefId>>,
    pub struct_fields: RefCell<DefIdMap<Rc<Vec<field_ty>>>>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub item_variance_map: RefCell<DefIdMap<Rc<ItemVariances>>>,

    /// True if the variance has been computed yet; false otherwise.
    pub variance_computed: Cell<bool>,

    /// A mapping from the def ID of an enum or struct type to the def ID
    /// of the method that implements its destructor. If the type is not
    /// present in this map, it does not have a destructor. This map is
    /// populated during the coherence phase of typechecking.
    pub destructor_for_type: RefCell<DefIdMap<ast::DefId>>,

    /// A method will be in this list if and only if it is a destructor.
    pub destructors: RefCell<DefIdSet>,

    /// Maps a trait onto a list of impls of that trait.
    pub trait_impls: RefCell<DefIdMap<Rc<RefCell<Vec<ast::DefId>>>>>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: RefCell<DefIdMap<Rc<Vec<ast::DefId>>>>,

    /// Maps a DefId of an impl to a list of its items.
    /// Note that this contains all of the impls that we know about,
    /// including ones in other crates. It's not clear that this is the best
    /// way to do it.
    pub impl_items: RefCell<DefIdMap<Vec<ImplOrTraitItemId>>>,

    /// Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    /// present in this set can be warned about.
    pub used_unsafe: RefCell<NodeSet>,

    /// Set of nodes which mark locals as mutable which end up getting used at
    /// some point. Local variable definitions not in this set can be warned
    /// about.
    pub used_mut_nodes: RefCell<NodeSet>,

    /// The set of external nominal types whose implementations have been read.
    /// This is used for lazy resolution of methods.
    pub populated_external_types: RefCell<DefIdSet>,

    /// The set of external traits whose implementations have been read. This
    /// is used for lazy resolution of traits.
    pub populated_external_traits: RefCell<DefIdSet>,

    /// Borrows
    pub upvar_capture_map: RefCell<UpvarCaptureMap>,

    /// These two caches are used by const_eval when decoding external statics
    /// and variants that are found.
    pub extern_const_statics: RefCell<DefIdMap<ast::NodeId>>,
    pub extern_const_variants: RefCell<DefIdMap<ast::NodeId>>,

    pub method_map: MethodMap<'tcx>,

    pub dependency_formats: RefCell<dependency_format::Dependencies>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kinds: RefCell<DefIdMap<ClosureKind>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_tys: RefCell<DefIdMap<ClosureTy<'tcx>>>,

    pub node_lint_levels: RefCell<FnvHashMap<(ast::NodeId, lint::LintId),
                                              lint::LevelSource>>,

    /// The types that must be asserted to be the same size for `transmute`
    /// to be valid. We gather up these restrictions in the intrinsicck pass
    /// and check them in trans.
    pub transmute_restrictions: RefCell<Vec<TransmuteRestriction<'tcx>>>,

    /// Maps any item's def-id to its stability index.
    pub stability: RefCell<stability::Index>,

    /// Maps def IDs to true if and only if they're associated types.
    pub associated_types: RefCell<DefIdMap<bool>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the representation hints for struct definitions.
    pub repr_hint_cache: RefCell<DefIdMap<Rc<Vec<attr::ReprAttr>>>>,

    /// Caches whether types are known to impl Copy. Note that type
    /// parameters are never placed into this cache, because their
    /// results are dependent on the parameter environment.
    pub type_impls_copy_cache: RefCell<HashMap<Ty<'tcx>,bool>>,

    /// Caches whether types are known to impl Sized. Note that type
    /// parameters are never placed into this cache, because their
    /// results are dependent on the parameter environment.
    pub type_impls_sized_cache: RefCell<HashMap<Ty<'tcx>,bool>>,

    /// Caches whether traits are object safe
    pub object_safety_cache: RefCell<DefIdMap<bool>>,

    /// Maps Expr NodeId's to their constant qualification.
    pub const_qualif_map: RefCell<NodeMap<check_const::ConstQualif>>,
}

// Flags that we track on types. These flags are propagated upwards
// through the type during type construction, so that we can quickly
// check whether the type has various kinds of types in it without
// recursing over the type itself.
bitflags! {
    flags TypeFlags: u32 {
        const NO_TYPE_FLAGS       = 0b0,
        const HAS_PARAMS          = 0b1,
        const HAS_SELF            = 0b10,
        const HAS_TY_INFER        = 0b100,
        const HAS_RE_INFER        = 0b1000,
        const HAS_RE_LATE_BOUND   = 0b10000,
        const HAS_REGIONS         = 0b100000,
        const HAS_TY_ERR          = 0b1000000,
        const HAS_PROJECTION      = 0b10000000,
        const NEEDS_SUBST   = HAS_PARAMS.bits | HAS_SELF.bits | HAS_REGIONS.bits,
    }
}

macro_rules! sty_debug_print {
    ($ctxt: expr, $($variant: ident),*) => {{
        // curious inner module to allow variant names to be used as
        // variable names.
        mod inner {
            use middle::ty;
            #[derive(Copy)]
            struct DebugStat {
                total: uint,
                region_infer: uint,
                ty_infer: uint,
                both_infer: uint,
            }

            pub fn go(tcx: &ty::ctxt) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*


                for (_, t) in &*tcx.interner.borrow() {
                    let variant = match t.sty {
                        ty::ty_bool | ty::ty_char | ty::ty_int(..) | ty::ty_uint(..) |
                            ty::ty_float(..) | ty::ty_str => continue,
                        ty::ty_err => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let region = t.flags.intersects(ty::HAS_RE_INFER);
                    let ty = t.flags.intersects(ty::HAS_TY_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if region { total.region_infer += 1; variant.region_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if region && ty { total.both_infer += 1; variant.both_infer += 1 }
                }
                println!("Ty interner             total           ty region  both");
                $(println!("    {:18}: {uses:6} {usespc:4.1}%, \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                           stringify!($variant),
                           uses = $variant.total,
                           usespc = $variant.total as f64 * 100.0 / total.total as f64,
                           ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                           region = $variant.region_infer as f64 * 100.0  / total.total as f64,
                           both = $variant.both_infer as f64 * 100.0  / total.total as f64);
                  )*
                println!("                  total {uses:6}        \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                         uses = total.total,
                         ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                         region = total.region_infer as f64 * 100.0  / total.total as f64,
                         both = total.both_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($ctxt)
    }}
}

impl<'tcx> ctxt<'tcx> {
    pub fn print_debug_stats(&self) {
        sty_debug_print!(
            self,
            ty_enum, ty_uniq, ty_vec, ty_ptr, ty_rptr, ty_bare_fn, ty_trait,
            ty_struct, ty_closure, ty_tup, ty_param, ty_open, ty_infer, ty_projection);

        println!("Substs interner: #{}", self.substs_interner.borrow().len());
        println!("BareFnTy interner: #{}", self.bare_fn_interner.borrow().len());
        println!("Region interner: #{}", self.region_interner.borrow().len());
    }
}

#[derive(Debug)]
pub struct TyS<'tcx> {
    pub sty: sty<'tcx>,
    pub flags: TypeFlags,

    // the maximal depth of any bound regions appearing in this type.
    region_depth: u32,
}

impl fmt::Debug for TypeFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.bits)
    }
}

impl<'tcx> PartialEq for TyS<'tcx> {
    fn eq(&self, other: &TyS<'tcx>) -> bool {
        // (self as *const _) == (other as *const _)
        (self as *const TyS<'tcx>) == (other as *const TyS<'tcx>)
    }
}
impl<'tcx> Eq for TyS<'tcx> {}

#[cfg(stage0)]
impl<'tcx, S: Writer + Hasher> Hash<S> for TyS<'tcx> {
    fn hash(&self, s: &mut S) {
        (self as *const _).hash(s)
    }
}
#[cfg(not(stage0))]
impl<'tcx> Hash for TyS<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const _).hash(s)
    }
}

pub type Ty<'tcx> = &'tcx TyS<'tcx>;

/// An entry in the type interner.
pub struct InternedTy<'tcx> {
    ty: Ty<'tcx>
}

// NB: An InternedTy compares and hashes as a sty.
impl<'tcx> PartialEq for InternedTy<'tcx> {
    fn eq(&self, other: &InternedTy<'tcx>) -> bool {
        self.ty.sty == other.ty.sty
    }
}

impl<'tcx> Eq for InternedTy<'tcx> {}

#[cfg(stage0)]
impl<'tcx, S: Writer + Hasher> Hash<S> for InternedTy<'tcx> {
    fn hash(&self, s: &mut S) {
        self.ty.sty.hash(s)
    }
}
#[cfg(not(stage0))]
impl<'tcx> Hash for InternedTy<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.ty.sty.hash(s)
    }
}

impl<'tcx> Borrow<sty<'tcx>> for InternedTy<'tcx> {
    fn borrow<'a>(&'a self) -> &'a sty<'tcx> {
        &self.ty.sty
    }
}

pub fn type_has_params(ty: Ty) -> bool {
    ty.flags.intersects(HAS_PARAMS)
}
pub fn type_has_self(ty: Ty) -> bool {
    ty.flags.intersects(HAS_SELF)
}
pub fn type_has_ty_infer(ty: Ty) -> bool {
    ty.flags.intersects(HAS_TY_INFER)
}
pub fn type_needs_infer(ty: Ty) -> bool {
    ty.flags.intersects(HAS_TY_INFER | HAS_RE_INFER)
}
pub fn type_has_projection(ty: Ty) -> bool {
    ty.flags.intersects(HAS_PROJECTION)
}

pub fn type_has_late_bound_regions(ty: Ty) -> bool {
    ty.flags.intersects(HAS_RE_LATE_BOUND)
}

/// An "escaping region" is a bound region whose binder is not part of `t`.
///
/// So, for example, consider a type like the following, which has two binders:
///
///    for<'a> fn(x: for<'b> fn(&'a int, &'b int))
///    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ outer scope
///                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~  inner scope
///
/// This type has *bound regions* (`'a`, `'b`), but it does not have escaping regions, because the
/// binders of both `'a` and `'b` are part of the type itself. However, if we consider the *inner
/// fn type*, that type has an escaping region: `'a`.
///
/// Note that what I'm calling an "escaping region" is often just called a "free region". However,
/// we already use the term "free region". It refers to the regions that we use to represent bound
/// regions on a fn definition while we are typechecking its body.
///
/// To clarify, conceptually there is no particular difference between an "escaping" region and a
/// "free" region. However, there is a big difference in practice. Basically, when "entering" a
/// binding level, one is generally required to do some sort of processing to a bound region, such
/// as replacing it with a fresh/skolemized region, or making an entry in the environment to
/// represent the scope to which it is attached, etc. An escaping region represents a bound region
/// for which this processing has not yet been done.
pub fn type_has_escaping_regions(ty: Ty) -> bool {
    type_escapes_depth(ty, 0)
}

pub fn type_escapes_depth(ty: Ty, depth: u32) -> bool {
    ty.region_depth > depth
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BareFnTy<'tcx> {
    pub unsafety: ast::Unsafety,
    pub abi: abi::Abi,
    pub sig: PolyFnSig<'tcx>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ClosureTy<'tcx> {
    pub unsafety: ast::Unsafety,
    pub abi: abi::Abi,
    pub sig: PolyFnSig<'tcx>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FnOutput<'tcx> {
    FnConverging(Ty<'tcx>),
    FnDiverging
}

impl<'tcx> FnOutput<'tcx> {
    pub fn diverges(&self) -> bool {
        *self == FnDiverging
    }

    pub fn unwrap(self) -> Ty<'tcx> {
        match self {
            ty::FnConverging(t) => t,
            ty::FnDiverging => unreachable!()
        }
    }
}

pub type PolyFnOutput<'tcx> = Binder<FnOutput<'tcx>>;

impl<'tcx> PolyFnOutput<'tcx> {
    pub fn diverges(&self) -> bool {
        self.0.diverges()
    }
}

/// Signature of a function type, which I have arbitrarily
/// decided to use to refer to the input/output types.
///
/// - `inputs` is the list of arguments and their modes.
/// - `output` is the return type.
/// - `variadic` indicates whether this is a variadic function. (only true for foreign fns)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FnSig<'tcx> {
    pub inputs: Vec<Ty<'tcx>>,
    pub output: FnOutput<'tcx>,
    pub variadic: bool
}

pub type PolyFnSig<'tcx> = Binder<FnSig<'tcx>>;

impl<'tcx> PolyFnSig<'tcx> {
    pub fn inputs(&self) -> ty::Binder<Vec<Ty<'tcx>>> {
        ty::Binder(self.0.inputs.clone())
    }
    pub fn input(&self, index: uint) -> ty::Binder<Ty<'tcx>> {
        ty::Binder(self.0.inputs[index])
    }
    pub fn output(&self) -> ty::Binder<FnOutput<'tcx>> {
        ty::Binder(self.0.output.clone())
    }
    pub fn variadic(&self) -> bool {
        self.0.variadic
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ParamTy {
    pub space: subst::ParamSpace,
    pub idx: u32,
    pub name: ast::Name,
}

/// A [De Bruijn index][dbi] is a standard means of representing
/// regions (and perhaps later types) in a higher-ranked setting. In
/// particular, imagine a type like this:
///
///     for<'a> fn(for<'b> fn(&'b int, &'a int), &'a char)
///     ^          ^            |        |         |
///     |          |            |        |         |
///     |          +------------+ 1      |         |
///     |                                |         |
///     +--------------------------------+ 2       |
///     |                                          |
///     +------------------------------------------+ 1
///
/// In this type, there are two binders (the outer fn and the inner
/// fn). We need to be able to determine, for any given region, which
/// fn type it is bound by, the inner or the outer one. There are
/// various ways you can do this, but a De Bruijn index is one of the
/// more convenient and has some nice properties. The basic idea is to
/// count the number of binders, inside out. Some examples should help
/// clarify what I mean.
///
/// Let's start with the reference type `&'b int` that is the first
/// argument to the inner function. This region `'b` is assigned a De
/// Bruijn index of 1, meaning "the innermost binder" (in this case, a
/// fn). The region `'a` that appears in the second argument type (`&'a
/// int`) would then be assigned a De Bruijn index of 2, meaning "the
/// second-innermost binder". (These indices are written on the arrays
/// in the diagram).
///
/// What is interesting is that De Bruijn index attached to a particular
/// variable will vary depending on where it appears. For example,
/// the final type `&'a char` also refers to the region `'a` declared on
/// the outermost fn. But this time, this reference is not nested within
/// any other binders (i.e., it is not an argument to the inner fn, but
/// rather the outer one). Therefore, in this case, it is assigned a
/// De Bruijn index of 1, because the innermost binder in that location
/// is the outer fn.
///
/// [dbi]: http://en.wikipedia.org/wiki/De_Bruijn_index
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, Copy)]
pub struct DebruijnIndex {
    // We maintain the invariant that this is never 0. So 1 indicates
    // the innermost binder. To ensure this, create with `DebruijnIndex::new`.
    pub depth: u32,
}

/// Representation of regions:
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum Region {
    // Region bound in a type or fn declaration which will be
    // substituted 'early' -- that is, at the same time when type
    // parameters are substituted.
    ReEarlyBound(/* param id */ ast::NodeId,
                 subst::ParamSpace,
                 /*index*/ u32,
                 ast::Name),

    // Region bound in a function scope, which will be substituted when the
    // function is called.
    ReLateBound(DebruijnIndex, BoundRegion),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    ReFree(FreeRegion),

    /// A concrete region naming some statically determined extent
    /// (e.g. an expression or sequence of statements) within the
    /// current function.
    ReScope(region::CodeExtent),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    ReStatic,

    /// A region variable.  Should not exist after typeck.
    ReInfer(InferRegion),

    /// Empty lifetime is for data that is never accessed.
    /// Bottom in the region lattice. We treat ReEmpty somewhat
    /// specially; at least right now, we do not generate instances of
    /// it during the GLB computations, but rather
    /// generate an error instead. This is to improve error messages.
    /// The only way to get an instance of ReEmpty is to have a region
    /// variable with no constraints.
    ReEmpty,
}

/// Upvars do not get their own node-id. Instead, we use the pair of
/// the original var id (that is, the root variable that is referenced
/// by the upvar) and the id of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UpvarId {
    pub var_id: ast::NodeId,
    pub closure_expr_id: ast::NodeId,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable, Copy)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    ImmBorrow,

    /// Data must be immutable but not aliasable.  This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when you the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    ///    let x: &mut int = ...;
    ///    let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///    struct Env { x: & &mut int }
    ///    let x: &mut int = ...;
    ///    let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///    struct Env { x: & &mut int }
    ///    let x: &mut int = ...;
    ///    let y = (&mut Env { &mut x }, fn_ptr); // changed from &x to &mut x
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    UniqueImmBorrow,

    /// Data is mutable and not aliasable.
    MutBorrow
}

/// Information describing the capture of an upvar. This is computed
/// during `typeck`, specifically by `regionck`.
#[derive(PartialEq, Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum UpvarCapture {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by reference.
    ByRef(UpvarBorrow),
}

#[derive(PartialEq, Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub struct UpvarBorrow {
    /// The kind of borrow: by-ref upvars have access to shared
    /// immutable borrows, which are not part of the normal language
    /// syntax.
    pub kind: BorrowKind,

    /// Region of the resulting reference.
    pub region: ty::Region,
}

pub type UpvarCaptureMap = FnvHashMap<UpvarId, UpvarCapture>;

impl Region {
    pub fn is_bound(&self) -> bool {
        match *self {
            ty::ReEarlyBound(..) => true,
            ty::ReLateBound(..) => true,
            _ => false
        }
    }

    pub fn escapes_depth(&self, depth: u32) -> bool {
        match *self {
            ty::ReLateBound(debruijn, _) => debruijn.depth > depth,
            _ => false,
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Debug, Copy)]
/// A "free" region `fr` can be interpreted as "some region
/// at least as big as the scope `fr.scope`".
pub struct FreeRegion {
    pub scope: region::DestructionScopeData,
    pub bound_region: BoundRegion
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum BoundRegion {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(u32),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The def-id is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(ast::DefId, ast::Name),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(u32),

    // Anonymous region for the implicit env pointer parameter
    // to a closure
    BrEnv
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in libsyntax/ast.rs as well.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum sty<'tcx> {
    ty_bool,
    ty_char,
    ty_int(ast::IntTy),
    ty_uint(ast::UintTy),
    ty_float(ast::FloatTy),
    /// Substs here, possibly against intuition, *may* contain `ty_param`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `ty_enum` corresponds to an enum
    /// definition and not a concrete use of it. To get the correct `ty_enum`
    /// from the tcx, use the `NodeId` from the `ast::Ty` and look it up in
    /// the `ast_ty_to_ty_cache`. This is probably true for `ty_struct` as
    /// well.`
    ty_enum(DefId, &'tcx Substs<'tcx>),
    ty_uniq(Ty<'tcx>),
    ty_str,
    ty_vec(Ty<'tcx>, Option<uint>), // Second field is length.
    ty_ptr(mt<'tcx>),
    ty_rptr(&'tcx Region, mt<'tcx>),

    // If the def-id is Some(_), then this is the type of a specific
    // fn item. Otherwise, if None(_), it a fn pointer type.
    ty_bare_fn(Option<DefId>, &'tcx BareFnTy<'tcx>),

    ty_trait(Box<TyTrait<'tcx>>),
    ty_struct(DefId, &'tcx Substs<'tcx>),

    ty_closure(DefId, &'tcx Region, &'tcx Substs<'tcx>),

    ty_tup(Vec<Ty<'tcx>>),

    ty_projection(ProjectionTy<'tcx>),
    ty_param(ParamTy), // type parameter

    ty_open(Ty<'tcx>), // A deref'ed fat pointer, i.e., a dynamically sized value
                       // and its size. Only ever used in trans. It is not necessary
                       // earlier since we don't need to distinguish a DST with its
                       // size (e.g., in a deref) vs a DST with the size elsewhere (
                       // e.g., in a field).

    ty_infer(InferTy), // something used only during inference/typeck
    ty_err, // Also only used during inference/typeck, to represent
            // the type of an erroneous expression (helps cut down
            // on non-useful type error messages)
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TyTrait<'tcx> {
    pub principal: ty::PolyTraitRef<'tcx>,
    pub bounds: ExistentialBounds<'tcx>,
}

impl<'tcx> TyTrait<'tcx> {
    pub fn principal_def_id(&self) -> ast::DefId {
        self.principal.0.def_id
    }

    /// Object types don't have a self-type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self-type. A common choice is `mk_err()`
    /// or some skolemized type.
    pub fn principal_trait_ref_with_self_ty(&self,
                                            tcx: &ctxt<'tcx>,
                                            self_ty: Ty<'tcx>)
                                            -> ty::PolyTraitRef<'tcx>
    {
        // otherwise the escaping regions would be captured by the binder
        assert!(!self_ty.has_escaping_regions());

        ty::Binder(Rc::new(ty::TraitRef {
            def_id: self.principal.0.def_id,
            substs: tcx.mk_substs(self.principal.0.substs.with_self_ty(self_ty)),
        }))
    }

    pub fn projection_bounds_with_self_ty(&self,
                                          tcx: &ctxt<'tcx>,
                                          self_ty: Ty<'tcx>)
                                          -> Vec<ty::PolyProjectionPredicate<'tcx>>
    {
        // otherwise the escaping regions would be captured by the binders
        assert!(!self_ty.has_escaping_regions());

        self.bounds.projection_bounds.iter()
            .map(|in_poly_projection_predicate| {
                let in_projection_ty = &in_poly_projection_predicate.0.projection_ty;
                let substs = tcx.mk_substs(in_projection_ty.trait_ref.substs.with_self_ty(self_ty));
                let trait_ref =
                    Rc::new(ty::TraitRef::new(in_projection_ty.trait_ref.def_id,
                                              substs));
                let projection_ty = ty::ProjectionTy {
                    trait_ref: trait_ref,
                    item_name: in_projection_ty.item_name
                };
                ty::Binder(ty::ProjectionPredicate {
                    projection_ty: projection_ty,
                    ty: in_poly_projection_predicate.0.ty
                })
            })
            .collect()
    }
}

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where clause:
///
///     T : Foo<U>
///
/// This would be represented by a trait-reference where the def-id is the
/// def-id for the trait `Foo` and the substs defines `T` as parameter 0 in the
/// `SelfSpace` and `U` as parameter 0 in the `TypeSpace`.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the substitutions.
///
/// Note that a `TraitRef` introduces a level of region binding, to
/// account for higher-ranked trait bounds like `T : for<'a> Foo<&'a
/// U>` or higher-ranked object types.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

pub type PolyTraitRef<'tcx> = Binder<Rc<TraitRef<'tcx>>>;

impl<'tcx> PolyTraitRef<'tcx> {
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.0.self_ty()
    }

    pub fn def_id(&self) -> ast::DefId {
        self.0.def_id
    }

    pub fn substs(&self) -> &'tcx Substs<'tcx> {
        // FIXME(#20664) every use of this fn is probably a bug, it should yield Binder<>
        self.0.substs
    }

    pub fn input_types(&self) -> &[Ty<'tcx>] {
        // FIXME(#20664) every use of this fn is probably a bug, it should yield Binder<>
        self.0.input_types()
    }

    pub fn to_poly_trait_predicate(&self) -> PolyTraitPredicate<'tcx> {
        // Note that we preserve binding levels
        Binder(TraitPredicate { trait_ref: self.0.clone() })
    }
}

/// Binder is a binder for higher-ranked lifetimes. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a int)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<TraitRef>`). Note that when we skolemize, instantiate,
/// erase, or otherwise "discharge" these bound regions, we change the
/// type from `Binder<T>` to just `T` (see
/// e.g. `liberate_late_bound_regions`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Binder<T>(pub T);

#[derive(Clone, Copy, PartialEq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
}

#[derive(Clone, Copy, Debug)]
pub enum terr_vstore_kind {
    terr_vec,
    terr_str,
    terr_fn,
    terr_trait
}

#[derive(Clone, Copy, Debug)]
pub struct expected_found<T> {
    pub expected: T,
    pub found: T
}

// Data structures used in type unification
#[derive(Clone, Copy, Debug)]
pub enum type_err<'tcx> {
    terr_mismatch,
    terr_unsafety_mismatch(expected_found<ast::Unsafety>),
    terr_abi_mismatch(expected_found<abi::Abi>),
    terr_mutability,
    terr_box_mutability,
    terr_ptr_mutability,
    terr_ref_mutability,
    terr_vec_mutability,
    terr_tuple_size(expected_found<uint>),
    terr_fixed_array_size(expected_found<uint>),
    terr_ty_param_size(expected_found<uint>),
    terr_arg_count,
    terr_regions_does_not_outlive(Region, Region),
    terr_regions_not_same(Region, Region),
    terr_regions_no_overlap(Region, Region),
    terr_regions_insufficiently_polymorphic(BoundRegion, Region),
    terr_regions_overly_polymorphic(BoundRegion, Region),
    terr_sorts(expected_found<Ty<'tcx>>),
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::FloatTy>),
    terr_traits(expected_found<ast::DefId>),
    terr_builtin_bounds(expected_found<BuiltinBounds>),
    terr_variadic_mismatch(expected_found<bool>),
    terr_cyclic_ty,
    terr_convergence_mismatch(expected_found<bool>),
    terr_projection_name_mismatched(expected_found<ast::Name>),
    terr_projection_bounds_length(expected_found<uint>),
}

/// Bounds suitable for a named type parameter like `A` in `fn foo<A>`
/// as well as the existential type parameter in an object type.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct ParamBounds<'tcx> {
    pub region_bounds: Vec<ty::Region>,
    pub builtin_bounds: BuiltinBounds,
    pub trait_bounds: Vec<PolyTraitRef<'tcx>>,
    pub projection_bounds: Vec<PolyProjectionPredicate<'tcx>>,
}

/// Bounds suitable for an existentially quantified type parameter
/// such as those that appear in object types or closure types. The
/// major difference between this case and `ParamBounds` is that
/// general purpose trait bounds are omitted and there must be
/// *exactly one* region.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct ExistentialBounds<'tcx> {
    pub region_bound: ty::Region,
    pub builtin_bounds: BuiltinBounds,
    pub projection_bounds: Vec<PolyProjectionPredicate<'tcx>>,
}

pub type BuiltinBounds = EnumSet<BuiltinBound>;

#[derive(Clone, RustcEncodable, PartialEq, Eq, RustcDecodable, Hash,
           Debug, Copy)]
#[repr(uint)]
pub enum BuiltinBound {
    BoundSend,
    BoundSized,
    BoundCopy,
    BoundSync,
}

pub fn empty_builtin_bounds() -> BuiltinBounds {
    EnumSet::new()
}

pub fn all_builtin_bounds() -> BuiltinBounds {
    let mut set = EnumSet::new();
    set.insert(BoundSend);
    set.insert(BoundSized);
    set.insert(BoundSync);
    set
}

/// An existential bound that does not implement any traits.
pub fn region_existential_bound<'tcx>(r: ty::Region) -> ExistentialBounds<'tcx> {
    ty::ExistentialBounds { region_bound: r,
                            builtin_bounds: empty_builtin_bounds(),
                            projection_bounds: Vec::new() }
}

impl CLike for BuiltinBound {
    fn to_usize(&self) -> uint {
        *self as uint
    }
    fn from_usize(v: uint) -> BuiltinBound {
        unsafe { mem::transmute(v) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVid {
    pub index: u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntVid {
    pub index: u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FloatVid {
    pub index: u32
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct RegionVid {
    pub index: u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid),
    FloatVar(FloatVid),

    /// A `FreshTy` is one that is generated as a replacement for an
    /// unbound type variable. This is convenient for caching etc. See
    /// `middle::infer::freshen` for more details.
    FreshTy(u32),

    // FIXME -- once integral fallback is impl'd, we should remove
    // this type. It's only needed to prevent spurious errors for
    // integers whose type winds up never being constrained.
    FreshIntTy(u32),
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum UnconstrainedNumeric {
    UnconstrainedFloat,
    UnconstrainedInt,
    Neither,
}


#[derive(Clone, RustcEncodable, RustcDecodable, Eq, Hash, Debug, Copy)]
pub enum InferRegion {
    ReVar(RegionVid),
    ReSkolemized(u32, BoundRegion)
}

impl cmp::PartialEq for InferRegion {
    fn eq(&self, other: &InferRegion) -> bool {
        match ((*self), *other) {
            (ReVar(rva), ReVar(rvb)) => {
                rva == rvb
            }
            (ReSkolemized(rva, _), ReSkolemized(rvb, _)) => {
                rva == rvb
            }
            _ => false
        }
    }
    fn ne(&self, other: &InferRegion) -> bool {
        !((*self) == (*other))
    }
}

impl fmt::Debug for TyVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "_#{}t", self.index)
    }
}

impl fmt::Debug for IntVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_#{}i", self.index)
    }
}

impl fmt::Debug for FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_#{}f", self.index)
    }
}

impl fmt::Debug for RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "'_#{}r", self.index)
    }
}

impl<'tcx> fmt::Debug for FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({:?}; variadic: {})->{:?}", self.inputs, self.variadic, self.output)
    }
}

impl fmt::Debug for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TyVar(ref v) => v.fmt(f),
            IntVar(ref v) => v.fmt(f),
            FloatVar(ref v) => v.fmt(f),
            FreshTy(v) => write!(f, "FreshTy({:?})", v),
            FreshIntTy(v) => write!(f, "FreshIntTy({:?})", v),
        }
    }
}

impl fmt::Debug for IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IntType(ref v) => v.fmt(f),
            UintType(ref v) => v.fmt(f),
        }
    }
}

/// Default region to use for the bound of objects that are
/// supplied as the value for this type parameter. This is derived
/// from `T:'a` annotations appearing in the type definition.  If
/// this is `None`, then the default is inherited from the
/// surrounding context. See RFC #599 for details.
#[derive(Copy, Clone, Debug)]
pub enum ObjectLifetimeDefault {
    /// Require an explicit annotation. Occurs when multiple
    /// `T:'a` constraints are found.
    Ambiguous,

    /// Use the given region as the default.
    Specific(Region),
}

#[derive(Clone, Debug)]
pub struct TypeParameterDef<'tcx> {
    pub name: ast::Name,
    pub def_id: ast::DefId,
    pub space: subst::ParamSpace,
    pub index: u32,
    pub bounds: ParamBounds<'tcx>,
    pub default: Option<Ty<'tcx>>,
    pub object_lifetime_default: Option<ObjectLifetimeDefault>,
}

#[derive(RustcEncodable, RustcDecodable, Clone, Debug)]
pub struct RegionParameterDef {
    pub name: ast::Name,
    pub def_id: ast::DefId,
    pub space: subst::ParamSpace,
    pub index: u32,
    pub bounds: Vec<ty::Region>,
}

impl RegionParameterDef {
    pub fn to_early_bound_region(&self) -> ty::Region {
        ty::ReEarlyBound(self.def_id.node, self.space, self.index, self.name)
    }
}

/// Information about the formal type/lifetime parameters associated
/// with an item or method. Analogous to ast::Generics.
#[derive(Clone, Debug)]
pub struct Generics<'tcx> {
    pub types: VecPerParamSpace<TypeParameterDef<'tcx>>,
    pub regions: VecPerParamSpace<RegionParameterDef>,
}

impl<'tcx> Generics<'tcx> {
    pub fn empty() -> Generics<'tcx> {
        Generics {
            types: VecPerParamSpace::empty(),
            regions: VecPerParamSpace::empty(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty() && self.regions.is_empty()
    }

    pub fn has_type_params(&self, space: subst::ParamSpace) -> bool {
        !self.types.is_empty_in(space)
    }

    pub fn has_region_params(&self, space: subst::ParamSpace) -> bool {
        !self.regions.is_empty_in(space)
    }
}

/// Bounds on generics.
#[derive(Clone, Debug)]
pub struct GenericPredicates<'tcx> {
    pub predicates: VecPerParamSpace<Predicate<'tcx>>,
}

impl<'tcx> GenericPredicates<'tcx> {
    pub fn empty() -> GenericPredicates<'tcx> {
        GenericPredicates {
            predicates: VecPerParamSpace::empty(),
        }
    }

    pub fn instantiate(&self, tcx: &ty::ctxt<'tcx>, substs: &Substs<'tcx>)
                       -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates {
            predicates: self.predicates.subst(tcx, substs),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Predicate<'tcx> {
    /// Corresponds to `where Foo : Bar<A,B,C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the parameters in the `TypeSpace`.
    Trait(PolyTraitPredicate<'tcx>),

    /// where `T1 == T2`.
    Equate(PolyEquatePredicate<'tcx>),

    /// where 'a : 'b
    RegionOutlives(PolyRegionOutlivesPredicate),

    /// where T : 'a
    TypeOutlives(PolyTypeOutlivesPredicate<'tcx>),

    /// where <T as TraitRef>::Name == X, approximately.
    /// See `ProjectionPredicate` struct for details.
    Projection(PolyProjectionPredicate<'tcx>),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: Rc<TraitRef<'tcx>>
}
pub type PolyTraitPredicate<'tcx> = ty::Binder<TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn def_id(&self) -> ast::DefId {
        self.trait_ref.def_id
    }

    pub fn input_types(&self) -> &[Ty<'tcx>] {
        self.trait_ref.substs.types.as_slice()
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
    }
}

impl<'tcx> PolyTraitPredicate<'tcx> {
    pub fn def_id(&self) -> ast::DefId {
        self.0.def_id()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct EquatePredicate<'tcx>(pub Ty<'tcx>, pub Ty<'tcx>); // `0 == 1`
pub type PolyEquatePredicate<'tcx> = ty::Binder<EquatePredicate<'tcx>>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OutlivesPredicate<A,B>(pub A, pub B); // `A : B`
pub type PolyOutlivesPredicate<A,B> = ty::Binder<OutlivesPredicate<A,B>>;
pub type PolyRegionOutlivesPredicate = PolyOutlivesPredicate<ty::Region, ty::Region>;
pub type PolyTypeOutlivesPredicate<'tcx> = PolyOutlivesPredicate<Ty<'tcx>, ty::Region>;

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T : TraitRef<..., Item=Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T : TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of Form
/// #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: ProjectionTy<'tcx>,
    pub ty: Ty<'tcx>,
}

pub type PolyProjectionPredicate<'tcx> = Binder<ProjectionPredicate<'tcx>>;

impl<'tcx> PolyProjectionPredicate<'tcx> {
    pub fn item_name(&self) -> ast::Name {
        self.0.projection_ty.item_name // safe to skip the binder to access a name
    }

    pub fn sort_key(&self) -> (ast::DefId, ast::Name) {
        self.0.projection_ty.sort_key()
    }
}

/// Represents the projection of an associated type. In explicit UFCS
/// form this would be written `<T as Trait<..>>::N`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ProjectionTy<'tcx> {
    /// The trait reference `T as Trait<..>`.
    pub trait_ref: Rc<ty::TraitRef<'tcx>>,

    /// The name `N` of the associated type.
    pub item_name: ast::Name,
}

impl<'tcx> ProjectionTy<'tcx> {
    pub fn sort_key(&self) -> (ast::DefId, ast::Name) {
        (self.trait_ref.def_id, self.item_name)
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for Rc<TraitRef<'tcx>> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        assert!(!self.has_escaping_regions());
        ty::Binder(self.clone())
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        // We are just preserving the binder levels here
        ty::Binder(self.0.trait_ref.clone())
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        // Note: unlike with TraitRef::to_poly_trait_ref(),
        // self.0.trait_ref is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        ty::Binder(self.0.projection_ty.trait_ref.clone())
    }
}

pub trait AsPredicate<'tcx> {
    fn as_predicate(&self) -> Predicate<'tcx>;
}

impl<'tcx> AsPredicate<'tcx> for Rc<TraitRef<'tcx>> {
    fn as_predicate(&self) -> Predicate<'tcx> {
        // we're about to add a binder, so let's check that we don't
        // accidentally capture anything, or else that might be some
        // weird debruijn accounting.
        assert!(!self.has_escaping_regions());

        ty::Predicate::Trait(ty::Binder(ty::TraitPredicate {
            trait_ref: self.clone()
        }))
    }
}

impl<'tcx> AsPredicate<'tcx> for PolyTraitRef<'tcx> {
    fn as_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(self.to_poly_trait_predicate())
    }
}

impl<'tcx> AsPredicate<'tcx> for PolyEquatePredicate<'tcx> {
    fn as_predicate(&self) -> Predicate<'tcx> {
        Predicate::Equate(self.clone())
    }
}

impl<'tcx> AsPredicate<'tcx> for PolyRegionOutlivesPredicate {
    fn as_predicate(&self) -> Predicate<'tcx> {
        Predicate::RegionOutlives(self.clone())
    }
}

impl<'tcx> AsPredicate<'tcx> for PolyTypeOutlivesPredicate<'tcx> {
    fn as_predicate(&self) -> Predicate<'tcx> {
        Predicate::TypeOutlives(self.clone())
    }
}

impl<'tcx> AsPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn as_predicate(&self) -> Predicate<'tcx> {
        Predicate::Projection(self.clone())
    }
}

impl<'tcx> Predicate<'tcx> {
    /// Iterates over the types in this predicate. Note that in all
    /// cases this is skipping over a binder, so late-bound regions
    /// with depth 0 are bound by the predicate.
    pub fn walk_tys(&self) -> IntoIter<Ty<'tcx>> {
        let vec: Vec<_> = match *self {
            ty::Predicate::Trait(ref data) => {
                data.0.trait_ref.substs.types.as_slice().to_vec()
            }
            ty::Predicate::Equate(ty::Binder(ref data)) => {
                vec![data.0, data.1]
            }
            ty::Predicate::TypeOutlives(ty::Binder(ref data)) => {
                vec![data.0]
            }
            ty::Predicate::RegionOutlives(..) => {
                vec![]
            }
            ty::Predicate::Projection(ref data) => {
                let trait_inputs = data.0.projection_ty.trait_ref.substs.types.as_slice();
                trait_inputs.iter()
                            .cloned()
                            .chain(Some(data.0.ty).into_iter())
                            .collect()
            }
        };

        // The only reason to collect into a vector here is that I was
        // too lazy to make the full (somewhat complicated) iterator
        // type that would be needed here. But I wanted this fn to
        // return an iterator conceptually, rather than a `Vec`, so as
        // to be closer to `Ty::walk`.
        vec.into_iter()
    }

    pub fn has_escaping_regions(&self) -> bool {
        match *self {
            Predicate::Trait(ref trait_ref) => trait_ref.has_escaping_regions(),
            Predicate::Equate(ref p) => p.has_escaping_regions(),
            Predicate::RegionOutlives(ref p) => p.has_escaping_regions(),
            Predicate::TypeOutlives(ref p) => p.has_escaping_regions(),
            Predicate::Projection(ref p) => p.has_escaping_regions(),
        }
    }

    pub fn to_opt_poly_trait_ref(&self) -> Option<PolyTraitRef<'tcx>> {
        match *self {
            Predicate::Trait(ref t) => {
                Some(t.to_poly_trait_ref())
            }
            Predicate::Projection(..) |
            Predicate::Equate(..) |
            Predicate::RegionOutlives(..) |
            Predicate::TypeOutlives(..) => {
                None
            }
        }
    }
}

/// Represents the bounds declared on a particular set of type
/// parameters.  Should eventually be generalized into a flag list of
/// where clauses.  You can obtain a `InstantiatedPredicates` list from a
/// `GenericPredicates` by using the `instantiate` method. Note that this method
/// reflects an important semantic invariant of `InstantiatedPredicates`: while
/// the `GenericPredicates` are expressed in terms of the bound type
/// parameters of the impl/trait/whatever, an `InstantiatedPredicates` instance
/// represented a set of bounds for some particular instantiation,
/// meaning that the generic parameters have been substituted with
/// their values.
///
/// Example:
///
///     struct Foo<T,U:Bar<T>> { ... }
///
/// Here, the `GenericPredicates` for `Foo` would contain a list of bounds like
/// `[[], [U:Bar<T>]]`.  Now if there were some particular reference
/// like `Foo<int,uint>`, then the `InstantiatedPredicates` would be `[[],
/// [uint:Bar<int>]]`.
#[derive(Clone, Debug)]
pub struct InstantiatedPredicates<'tcx> {
    pub predicates: VecPerParamSpace<Predicate<'tcx>>,
}

impl<'tcx> InstantiatedPredicates<'tcx> {
    pub fn empty() -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates { predicates: VecPerParamSpace::empty() }
    }

    pub fn has_escaping_regions(&self) -> bool {
        self.predicates.any(|p| p.has_escaping_regions())
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

impl<'tcx> TraitRef<'tcx> {
    pub fn new(def_id: ast::DefId, substs: &'tcx Substs<'tcx>) -> TraitRef<'tcx> {
        TraitRef { def_id: def_id, substs: substs }
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.self_ty().unwrap()
    }

    pub fn input_types(&self) -> &[Ty<'tcx>] {
        // Select only the "input types" from a trait-reference. For
        // now this is all the types that appear in the
        // trait-reference, but it should eventually exclude
        // associated types.
        self.substs.types.as_slice()
    }
}

/// When type checking, we use the `ParameterEnvironment` to track
/// details about the type/lifetime parameters that are in scope.
/// It primarily stores the bounds information.
///
/// Note: This information might seem to be redundant with the data in
/// `tcx.ty_param_defs`, but it is not. That table contains the
/// parameter definitions from an "outside" perspective, but this
/// struct will contain the bounds for a parameter as seen from inside
/// the function body. Currently the only real distinction is that
/// bound lifetime parameters are replaced with free ones, but in the
/// future I hope to refine the representation of types so as to make
/// more distinctions clearer.
#[derive(Clone)]
pub struct ParameterEnvironment<'a, 'tcx:'a> {
    pub tcx: &'a ctxt<'tcx>,

    /// See `construct_free_substs` for details.
    pub free_substs: Substs<'tcx>,

    /// Each type parameter has an implicit region bound that
    /// indicates it must outlive at least the function body (the user
    /// may specify stronger requirements). This field indicates the
    /// region of the callee.
    pub implicit_region_bound: ty::Region,

    /// Obligations that the caller must satisfy. This is basically
    /// the set of bounds on the in-scope type parameters, translated
    /// into Obligations.
    pub caller_bounds: Vec<ty::Predicate<'tcx>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,
}

impl<'a, 'tcx> ParameterEnvironment<'a, 'tcx> {
    pub fn with_caller_bounds(&self,
                              caller_bounds: Vec<ty::Predicate<'tcx>>)
                              -> ParameterEnvironment<'a,'tcx>
    {
        ParameterEnvironment {
            tcx: self.tcx,
            free_substs: self.free_substs.clone(),
            implicit_region_bound: self.implicit_region_bound,
            caller_bounds: caller_bounds,
            selection_cache: traits::SelectionCache::new(),
        }
    }

    pub fn for_item(cx: &'a ctxt<'tcx>, id: NodeId) -> ParameterEnvironment<'a, 'tcx> {
        match cx.map.find(id) {
            Some(ast_map::NodeImplItem(ref impl_item)) => {
                match **impl_item {
                    ast::MethodImplItem(ref method) => {
                        let method_def_id = ast_util::local_def(id);
                        match ty::impl_or_trait_item(cx, method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                let method_bounds = &method_ty.predicates;
                                construct_parameter_environment(
                                    cx,
                                    method.span,
                                    method_generics,
                                    method_bounds,
                                    method.pe_body().id)
                            }
                            TypeTraitItem(_) => {
                                cx.sess
                                  .bug("ParameterEnvironment::for_item(): \
                                        can't create a parameter environment \
                                        for type trait items")
                            }
                        }
                    }
                    ast::TypeImplItem(_) => {
                        cx.sess.bug("ParameterEnvironment::for_item(): \
                                     can't create a parameter environment \
                                     for type impl items")
                    }
                }
            }
            Some(ast_map::NodeTraitItem(trait_method)) => {
                match *trait_method {
                    ast::RequiredMethod(ref required) => {
                        cx.sess.span_bug(required.span,
                                         "ParameterEnvironment::for_item():
                                          can't create a parameter \
                                          environment for required trait \
                                          methods")
                    }
                    ast::ProvidedMethod(ref method) => {
                        let method_def_id = ast_util::local_def(id);
                        match ty::impl_or_trait_item(cx, method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                let method_bounds = &method_ty.predicates;
                                construct_parameter_environment(
                                    cx,
                                    method.span,
                                    method_generics,
                                    method_bounds,
                                    method.pe_body().id)
                            }
                            TypeTraitItem(_) => {
                                cx.sess
                                  .bug("ParameterEnvironment::for_item(): \
                                        can't create a parameter environment \
                                        for type trait items")
                            }
                        }
                    }
                    ast::TypeTraitItem(_) => {
                        cx.sess.bug("ParameterEnvironment::from_item(): \
                                     can't create a parameter environment \
                                     for type trait items")
                    }
                }
            }
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ast::ItemFn(_, _, _, _, ref body) => {
                        // We assume this is a function.
                        let fn_def_id = ast_util::local_def(id);
                        let fn_scheme = lookup_item_type(cx, fn_def_id);
                        let fn_predicates = lookup_predicates(cx, fn_def_id);

                        construct_parameter_environment(cx,
                                                        item.span,
                                                        &fn_scheme.generics,
                                                        &fn_predicates,
                                                        body.id)
                    }
                    ast::ItemEnum(..) |
                    ast::ItemStruct(..) |
                    ast::ItemImpl(..) |
                    ast::ItemConst(..) |
                    ast::ItemStatic(..) => {
                        let def_id = ast_util::local_def(id);
                        let scheme = lookup_item_type(cx, def_id);
                        let predicates = lookup_predicates(cx, def_id);
                        construct_parameter_environment(cx,
                                                        item.span,
                                                        &scheme.generics,
                                                        &predicates,
                                                        id)
                    }
                    _ => {
                        cx.sess.span_bug(item.span,
                                         "ParameterEnvironment::from_item():
                                          can't create a parameter \
                                          environment for this kind of item")
                    }
                }
            }
            Some(ast_map::NodeExpr(..)) => {
                // This is a convenience to allow closures to work.
                ParameterEnvironment::for_item(cx, cx.map.get_parent(id))
            }
            _ => {
                cx.sess.bug(&format!("ParameterEnvironment::from_item(): \
                                     `{}` is not an item",
                                    cx.map.node_to_string(id))[])
            }
        }
    }
}

/// A "type scheme", in ML terminology, is a type combined with some
/// set of generic types that the type is, well, generic over. In Rust
/// terms, it is the "type" of a fn item or struct -- this type will
/// include various generic parameters that must be substituted when
/// the item/struct is referenced. That is called converting the type
/// scheme to a monotype.
///
/// - `generics`: the set of type parameters and their bounds
/// - `ty`: the base types, which may reference the parameters defined
///   in `generics`
///
/// Note that TypeSchemes are also sometimes called "polytypes" (and
/// in fact this struct used to carry that name, so you may find some
/// stray references in a comment or something). We try to reserve the
/// "poly" prefix to refer to higher-ranked things, as in
/// `PolyTraitRef`.
///
/// Note that each item also comes with predicates, see
/// `lookup_predicates`.
#[derive(Clone, Debug)]
pub struct TypeScheme<'tcx> {
    pub generics: Generics<'tcx>,
    pub ty: Ty<'tcx>,
}

/// As `TypeScheme` but for a trait ref.
pub struct TraitDef<'tcx> {
    pub unsafety: ast::Unsafety,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait wil
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    /// Generic type definitions. Note that `Self` is listed in here
    /// as having a single bound, the trait itself (e.g., in the trait
    /// `Eq`, there is a single bound `Self : Eq`). This is so that
    /// default methods get to assume that the `Self` parameters
    /// implements the trait.
    pub generics: Generics<'tcx>,

    /// The "supertrait" bounds.
    pub bounds: ParamBounds<'tcx>,

    pub trait_ref: Rc<ty::TraitRef<'tcx>>,

    /// A list of the associated types defined in this trait. Useful
    /// for resolving `X::Foo` type markers.
    pub associated_type_names: Vec<ast::Name>,
}

/// Records the substitutions used to translate the polytype for an
/// item into the monotype of an item reference.
#[derive(Clone)]
pub struct ItemSubsts<'tcx> {
    pub substs: Substs<'tcx>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, RustcEncodable, RustcDecodable)]
pub enum ClosureKind {
    FnClosureKind,
    FnMutClosureKind,
    FnOnceClosureKind,
}

impl ClosureKind {
    pub fn trait_did(&self, cx: &ctxt) -> ast::DefId {
        let result = match *self {
            FnClosureKind => cx.lang_items.require(FnTraitLangItem),
            FnMutClosureKind => {
                cx.lang_items.require(FnMutTraitLangItem)
            }
            FnOnceClosureKind => {
                cx.lang_items.require(FnOnceTraitLangItem)
            }
        };
        match result {
            Ok(trait_did) => trait_did,
            Err(err) => cx.sess.fatal(&err[..]),
        }
    }
}

pub trait ClosureTyper<'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.param_env().tcx
    }

    fn param_env<'a>(&'a self) -> &'a ty::ParameterEnvironment<'a, 'tcx>;

    /// Is this a `Fn`, `FnMut` or `FnOnce` closure? During typeck,
    /// returns `None` if the kind of this closure has not yet been
    /// inferred.
    fn closure_kind(&self,
                    def_id: ast::DefId)
                    -> Option<ty::ClosureKind>;

    /// Returns the argument/return types of this closure.
    fn closure_type(&self,
                    def_id: ast::DefId,
                    substs: &subst::Substs<'tcx>)
                    -> ty::ClosureTy<'tcx>;

    /// Returns the set of all upvars and their transformed
    /// types. During typeck, maybe return `None` if the upvar types
    /// have not yet been inferred.
    fn closure_upvars(&self,
                      def_id: ast::DefId,
                      substs: &Substs<'tcx>)
                      -> Option<Vec<ClosureUpvar<'tcx>>>;
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(arena: &'tcx TypedArena<TyS<'tcx>>,
           interner: &mut FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>)
           -> CommonTypes<'tcx>
    {
        CommonTypes {
            bool: intern_ty(arena, interner, ty_bool),
            char: intern_ty(arena, interner, ty_char),
            err: intern_ty(arena, interner, ty_err),
            int: intern_ty(arena, interner, ty_int(ast::TyIs(false))),
            i8: intern_ty(arena, interner, ty_int(ast::TyI8)),
            i16: intern_ty(arena, interner, ty_int(ast::TyI16)),
            i32: intern_ty(arena, interner, ty_int(ast::TyI32)),
            i64: intern_ty(arena, interner, ty_int(ast::TyI64)),
            uint: intern_ty(arena, interner, ty_uint(ast::TyUs(false))),
            u8: intern_ty(arena, interner, ty_uint(ast::TyU8)),
            u16: intern_ty(arena, interner, ty_uint(ast::TyU16)),
            u32: intern_ty(arena, interner, ty_uint(ast::TyU32)),
            u64: intern_ty(arena, interner, ty_uint(ast::TyU64)),
            f32: intern_ty(arena, interner, ty_float(ast::TyF32)),
            f64: intern_ty(arena, interner, ty_float(ast::TyF64)),
        }
    }
}

pub fn mk_ctxt<'tcx>(s: Session,
                     arenas: &'tcx CtxtArenas<'tcx>,
                     dm: DefMap,
                     named_region_map: resolve_lifetime::NamedRegionMap,
                     map: ast_map::Map<'tcx>,
                     freevars: RefCell<FreevarMap>,
                     region_maps: middle::region::RegionMaps,
                     lang_items: middle::lang_items::LanguageItems,
                     stability: stability::Index) -> ctxt<'tcx>
{
    let mut interner = FnvHashMap();
    let common_types = CommonTypes::new(&arenas.type_, &mut interner);

    ctxt {
        arenas: arenas,
        interner: RefCell::new(interner),
        substs_interner: RefCell::new(FnvHashMap()),
        bare_fn_interner: RefCell::new(FnvHashMap()),
        region_interner: RefCell::new(FnvHashMap()),
        types: common_types,
        named_region_map: named_region_map,
        item_variance_map: RefCell::new(DefIdMap()),
        variance_computed: Cell::new(false),
        sess: s,
        def_map: dm,
        region_maps: region_maps,
        node_types: RefCell::new(FnvHashMap()),
        item_substs: RefCell::new(NodeMap()),
        trait_refs: RefCell::new(NodeMap()),
        trait_defs: RefCell::new(DefIdMap()),
        predicates: RefCell::new(DefIdMap()),
        object_cast_map: RefCell::new(NodeMap()),
        map: map,
        intrinsic_defs: RefCell::new(DefIdMap()),
        freevars: freevars,
        tcache: RefCell::new(DefIdMap()),
        rcache: RefCell::new(FnvHashMap()),
        short_names_cache: RefCell::new(FnvHashMap()),
        tc_cache: RefCell::new(FnvHashMap()),
        ast_ty_to_ty_cache: RefCell::new(NodeMap()),
        enum_var_cache: RefCell::new(DefIdMap()),
        impl_or_trait_items: RefCell::new(DefIdMap()),
        trait_item_def_ids: RefCell::new(DefIdMap()),
        trait_items_cache: RefCell::new(DefIdMap()),
        impl_trait_cache: RefCell::new(DefIdMap()),
        ty_param_defs: RefCell::new(NodeMap()),
        adjustments: RefCell::new(NodeMap()),
        normalized_cache: RefCell::new(FnvHashMap()),
        lang_items: lang_items,
        provided_method_sources: RefCell::new(DefIdMap()),
        struct_fields: RefCell::new(DefIdMap()),
        destructor_for_type: RefCell::new(DefIdMap()),
        destructors: RefCell::new(DefIdSet()),
        trait_impls: RefCell::new(DefIdMap()),
        inherent_impls: RefCell::new(DefIdMap()),
        impl_items: RefCell::new(DefIdMap()),
        used_unsafe: RefCell::new(NodeSet()),
        used_mut_nodes: RefCell::new(NodeSet()),
        populated_external_types: RefCell::new(DefIdSet()),
        populated_external_traits: RefCell::new(DefIdSet()),
        upvar_capture_map: RefCell::new(FnvHashMap()),
        extern_const_statics: RefCell::new(DefIdMap()),
        extern_const_variants: RefCell::new(DefIdMap()),
        method_map: RefCell::new(FnvHashMap()),
        dependency_formats: RefCell::new(FnvHashMap()),
        closure_kinds: RefCell::new(DefIdMap()),
        closure_tys: RefCell::new(DefIdMap()),
        node_lint_levels: RefCell::new(FnvHashMap()),
        transmute_restrictions: RefCell::new(Vec::new()),
        stability: RefCell::new(stability),
        associated_types: RefCell::new(DefIdMap()),
        selection_cache: traits::SelectionCache::new(),
        repr_hint_cache: RefCell::new(DefIdMap()),
        type_impls_copy_cache: RefCell::new(HashMap::new()),
        type_impls_sized_cache: RefCell::new(HashMap::new()),
        object_safety_cache: RefCell::new(DefIdMap()),
        const_qualif_map: RefCell::new(NodeMap()),
   }
}

// Type constructors

impl<'tcx> ctxt<'tcx> {
    pub fn mk_substs(&self, substs: Substs<'tcx>) -> &'tcx Substs<'tcx> {
        if let Some(substs) = self.substs_interner.borrow().get(&substs) {
            return *substs;
        }

        let substs = self.arenas.substs.alloc(substs);
        self.substs_interner.borrow_mut().insert(substs, substs);
        substs
    }

    pub fn mk_bare_fn(&self, bare_fn: BareFnTy<'tcx>) -> &'tcx BareFnTy<'tcx> {
        if let Some(bare_fn) = self.bare_fn_interner.borrow().get(&bare_fn) {
            return *bare_fn;
        }

        let bare_fn = self.arenas.bare_fn.alloc(bare_fn);
        self.bare_fn_interner.borrow_mut().insert(bare_fn, bare_fn);
        bare_fn
    }

    pub fn mk_region(&self, region: Region) -> &'tcx Region {
        if let Some(region) = self.region_interner.borrow().get(&region) {
            return *region;
        }

        let region = self.arenas.region.alloc(region);
        self.region_interner.borrow_mut().insert(region, region);
        region
    }

    pub fn closure_kind(&self, def_id: ast::DefId) -> ty::ClosureKind {
        self.closure_kinds.borrow()[def_id]
    }

    pub fn closure_type(&self,
                        def_id: ast::DefId,
                        substs: &subst::Substs<'tcx>)
                        -> ty::ClosureTy<'tcx>
    {
        self.closure_tys.borrow()[def_id].subst(self, substs)
    }
}

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for Ty above).
pub fn mk_t<'tcx>(cx: &ctxt<'tcx>, st: sty<'tcx>) -> Ty<'tcx> {
    let mut interner = cx.interner.borrow_mut();
    intern_ty(&cx.arenas.type_, &mut *interner, st)
}

fn intern_ty<'tcx>(type_arena: &'tcx TypedArena<TyS<'tcx>>,
                   interner: &mut FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>,
                   st: sty<'tcx>)
                   -> Ty<'tcx>
{
    match interner.get(&st) {
        Some(ty) => return *ty,
        _ => ()
    }

    let flags = FlagComputation::for_sty(&st);

    let ty = match () {
        () => type_arena.alloc(TyS { sty: st,
                                     flags: flags.flags,
                                     region_depth: flags.depth, }),
    };

    debug!("Interned type: {:?} Pointer: {:?}",
           ty, ty as *const _);

    interner.insert(InternedTy { ty: ty }, ty);

    ty
}

struct FlagComputation {
    flags: TypeFlags,

    // maximum depth of any bound region that we have seen thus far
    depth: u32,
}

impl FlagComputation {
    fn new() -> FlagComputation {
        FlagComputation { flags: NO_TYPE_FLAGS, depth: 0 }
    }

    fn for_sty(st: &sty) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_sty(st);
        result
    }

    fn add_flags(&mut self, flags: TypeFlags) {
        self.flags = self.flags | flags;
    }

    fn add_depth(&mut self, depth: u32) {
        if depth > self.depth {
            self.depth = depth;
        }
    }

    /// Adds the flags/depth from a set of types that appear within the current type, but within a
    /// region binder.
    fn add_bound_computation(&mut self, computation: &FlagComputation) {
        self.add_flags(computation.flags);

        // The types that contributed to `computation` occurred within
        // a region binder, so subtract one from the region depth
        // within when adding the depth to `self`.
        let depth = computation.depth;
        if depth > 0 {
            self.add_depth(depth - 1);
        }
    }

    fn add_sty(&mut self, st: &sty) {
        match st {
            &ty_bool |
            &ty_char |
            &ty_int(_) |
            &ty_float(_) |
            &ty_uint(_) |
            &ty_str => {
            }

            // You might think that we could just return ty_err for
            // any type containing ty_err as a component, and get
            // rid of the HAS_TY_ERR flag -- likewise for ty_bot (with
            // the exception of function types that return bot).
            // But doing so caused sporadic memory corruption, and
            // neither I (tjc) nor nmatsakis could figure out why,
            // so we're doing it this way.
            &ty_err => {
                self.add_flags(HAS_TY_ERR)
            }

            &ty_param(ref p) => {
                if p.space == subst::SelfSpace {
                    self.add_flags(HAS_SELF);
                } else {
                    self.add_flags(HAS_PARAMS);
                }
            }

            &ty_closure(_, region, substs) => {
                self.add_region(*region);
                self.add_substs(substs);
            }

            &ty_infer(_) => {
                self.add_flags(HAS_TY_INFER)
            }

            &ty_enum(_, substs) | &ty_struct(_, substs) => {
                self.add_substs(substs);
            }

            &ty_projection(ref data) => {
                self.add_flags(HAS_PROJECTION);
                self.add_projection_ty(data);
            }

            &ty_trait(box TyTrait { ref principal, ref bounds }) => {
                let mut computation = FlagComputation::new();
                computation.add_substs(principal.0.substs);
                for projection_bound in &bounds.projection_bounds {
                    let mut proj_computation = FlagComputation::new();
                    proj_computation.add_projection_predicate(&projection_bound.0);
                    computation.add_bound_computation(&proj_computation);
                }
                self.add_bound_computation(&computation);

                self.add_bounds(bounds);
            }

            &ty_uniq(tt) | &ty_vec(tt, _) | &ty_open(tt) => {
                self.add_ty(tt)
            }

            &ty_ptr(ref m) => {
                self.add_ty(m.ty);
            }

            &ty_rptr(r, ref m) => {
                self.add_region(*r);
                self.add_ty(m.ty);
            }

            &ty_tup(ref ts) => {
                self.add_tys(&ts[..]);
            }

            &ty_bare_fn(_, ref f) => {
                self.add_fn_sig(&f.sig);
            }
        }
    }

    fn add_ty(&mut self, ty: Ty) {
        self.add_flags(ty.flags);
        self.add_depth(ty.region_depth);
    }

    fn add_tys(&mut self, tys: &[Ty]) {
        for &ty in tys {
            self.add_ty(ty);
        }
    }

    fn add_fn_sig(&mut self, fn_sig: &PolyFnSig) {
        let mut computation = FlagComputation::new();

        computation.add_tys(&fn_sig.0.inputs[]);

        if let ty::FnConverging(output) = fn_sig.0.output {
            computation.add_ty(output);
        }

        self.add_bound_computation(&computation);
    }

    fn add_region(&mut self, r: Region) {
        self.add_flags(HAS_REGIONS);
        match r {
            ty::ReInfer(_) => { self.add_flags(HAS_RE_INFER); }
            ty::ReLateBound(debruijn, _) => {
                self.add_flags(HAS_RE_LATE_BOUND);
                self.add_depth(debruijn.depth);
            }
            _ => { }
        }
    }

    fn add_projection_predicate(&mut self, projection_predicate: &ProjectionPredicate) {
        self.add_projection_ty(&projection_predicate.projection_ty);
        self.add_ty(projection_predicate.ty);
    }

    fn add_projection_ty(&mut self, projection_ty: &ProjectionTy) {
        self.add_substs(projection_ty.trait_ref.substs);
    }

    fn add_substs(&mut self, substs: &Substs) {
        self.add_tys(substs.types.as_slice());
        match substs.regions {
            subst::ErasedRegions => {}
            subst::NonerasedRegions(ref regions) => {
                for &r in regions.iter() {
                    self.add_region(r);
                }
            }
        }
    }

    fn add_bounds(&mut self, bounds: &ExistentialBounds) {
        self.add_region(bounds.region_bound);
    }
}

pub fn mk_mach_int<'tcx>(tcx: &ctxt<'tcx>, tm: ast::IntTy) -> Ty<'tcx> {
    match tm {
        ast::TyIs(_)   => tcx.types.int,
        ast::TyI8   => tcx.types.i8,
        ast::TyI16  => tcx.types.i16,
        ast::TyI32  => tcx.types.i32,
        ast::TyI64  => tcx.types.i64,
    }
}

pub fn mk_mach_uint<'tcx>(tcx: &ctxt<'tcx>, tm: ast::UintTy) -> Ty<'tcx> {
    match tm {
        ast::TyUs(_)   => tcx.types.uint,
        ast::TyU8   => tcx.types.u8,
        ast::TyU16  => tcx.types.u16,
        ast::TyU32  => tcx.types.u32,
        ast::TyU64  => tcx.types.u64,
    }
}

pub fn mk_mach_float<'tcx>(tcx: &ctxt<'tcx>, tm: ast::FloatTy) -> Ty<'tcx> {
    match tm {
        ast::TyF32  => tcx.types.f32,
        ast::TyF64  => tcx.types.f64,
    }
}

pub fn mk_str<'tcx>(cx: &ctxt<'tcx>) -> Ty<'tcx> {
    mk_t(cx, ty_str)
}

pub fn mk_str_slice<'tcx>(cx: &ctxt<'tcx>, r: &'tcx Region, m: ast::Mutability) -> Ty<'tcx> {
    mk_rptr(cx, r,
            mt {
                ty: mk_t(cx, ty_str),
                mutbl: m
            })
}

pub fn mk_enum<'tcx>(cx: &ctxt<'tcx>, did: ast::DefId, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_enum(did, substs))
}

pub fn mk_uniq<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> { mk_t(cx, ty_uniq(ty)) }

pub fn mk_ptr<'tcx>(cx: &ctxt<'tcx>, tm: mt<'tcx>) -> Ty<'tcx> { mk_t(cx, ty_ptr(tm)) }

pub fn mk_rptr<'tcx>(cx: &ctxt<'tcx>, r: &'tcx Region, tm: mt<'tcx>) -> Ty<'tcx> {
    mk_t(cx, ty_rptr(r, tm))
}

pub fn mk_mut_rptr<'tcx>(cx: &ctxt<'tcx>, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutMutable})
}
pub fn mk_imm_rptr<'tcx>(cx: &ctxt<'tcx>, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_mut_ptr<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutMutable})
}

pub fn mk_imm_ptr<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_nil_ptr<'tcx>(cx: &ctxt<'tcx>) -> Ty<'tcx> {
    mk_ptr(cx, mt {ty: mk_nil(cx), mutbl: ast::MutImmutable})
}

pub fn mk_vec<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>, sz: Option<uint>) -> Ty<'tcx> {
    mk_t(cx, ty_vec(ty, sz))
}

pub fn mk_slice<'tcx>(cx: &ctxt<'tcx>, r: &'tcx Region, tm: mt<'tcx>) -> Ty<'tcx> {
    mk_rptr(cx, r,
            mt {
                ty: mk_vec(cx, tm.ty, None),
                mutbl: tm.mutbl
            })
}

pub fn mk_tup<'tcx>(cx: &ctxt<'tcx>, ts: Vec<Ty<'tcx>>) -> Ty<'tcx> {
    mk_t(cx, ty_tup(ts))
}

pub fn mk_nil<'tcx>(cx: &ctxt<'tcx>) -> Ty<'tcx> {
    mk_tup(cx, Vec::new())
}

pub fn mk_bare_fn<'tcx>(cx: &ctxt<'tcx>,
                        opt_def_id: Option<ast::DefId>,
                        fty: &'tcx BareFnTy<'tcx>) -> Ty<'tcx> {
    mk_t(cx, ty_bare_fn(opt_def_id, fty))
}

pub fn mk_ctor_fn<'tcx>(cx: &ctxt<'tcx>,
                        def_id: ast::DefId,
                        input_tys: &[Ty<'tcx>],
                        output: Ty<'tcx>) -> Ty<'tcx> {
    let input_args = input_tys.iter().cloned().collect();
    mk_bare_fn(cx,
               Some(def_id),
               cx.mk_bare_fn(BareFnTy {
                   unsafety: ast::Unsafety::Normal,
                   abi: abi::Rust,
                   sig: ty::Binder(FnSig {
                    inputs: input_args,
                    output: ty::FnConverging(output),
                    variadic: false
                   })
                }))
}

pub fn mk_trait<'tcx>(cx: &ctxt<'tcx>,
                      principal: ty::PolyTraitRef<'tcx>,
                      bounds: ExistentialBounds<'tcx>)
                      -> Ty<'tcx>
{
    assert!(bound_list_is_sorted(&bounds.projection_bounds));

    let inner = box TyTrait {
        principal: principal,
        bounds: bounds
    };
    mk_t(cx, ty_trait(inner))
}

fn bound_list_is_sorted(bounds: &[ty::PolyProjectionPredicate]) -> bool {
    bounds.len() == 0 ||
        bounds[1..].iter().enumerate().all(
            |(index, bound)| bounds[index].sort_key() <= bound.sort_key())
}

pub fn sort_bounds_list(bounds: &mut [ty::PolyProjectionPredicate]) {
    bounds.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()))
}

pub fn mk_projection<'tcx>(cx: &ctxt<'tcx>,
                           trait_ref: Rc<ty::TraitRef<'tcx>>,
                           item_name: ast::Name)
                           -> Ty<'tcx> {
    // take a copy of substs so that we own the vectors inside
    let inner = ProjectionTy { trait_ref: trait_ref, item_name: item_name };
    mk_t(cx, ty_projection(inner))
}

pub fn mk_struct<'tcx>(cx: &ctxt<'tcx>, struct_id: ast::DefId,
                       substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_struct(struct_id, substs))
}

pub fn mk_closure<'tcx>(cx: &ctxt<'tcx>, closure_id: ast::DefId,
                        region: &'tcx Region, substs: &'tcx Substs<'tcx>)
                        -> Ty<'tcx> {
    mk_t(cx, ty_closure(closure_id, region, substs))
}

pub fn mk_var<'tcx>(cx: &ctxt<'tcx>, v: TyVid) -> Ty<'tcx> {
    mk_infer(cx, TyVar(v))
}

pub fn mk_int_var<'tcx>(cx: &ctxt<'tcx>, v: IntVid) -> Ty<'tcx> {
    mk_infer(cx, IntVar(v))
}

pub fn mk_float_var<'tcx>(cx: &ctxt<'tcx>, v: FloatVid) -> Ty<'tcx> {
    mk_infer(cx, FloatVar(v))
}

pub fn mk_infer<'tcx>(cx: &ctxt<'tcx>, it: InferTy) -> Ty<'tcx> {
    mk_t(cx, ty_infer(it))
}

pub fn mk_param<'tcx>(cx: &ctxt<'tcx>,
                      space: subst::ParamSpace,
                      index: u32,
                      name: ast::Name) -> Ty<'tcx> {
    mk_t(cx, ty_param(ParamTy { space: space, idx: index, name: name }))
}

pub fn mk_self_type<'tcx>(cx: &ctxt<'tcx>) -> Ty<'tcx> {
    mk_param(cx, subst::SelfSpace, 0, special_idents::type_self.name)
}

pub fn mk_param_from_def<'tcx>(cx: &ctxt<'tcx>, def: &TypeParameterDef) -> Ty<'tcx> {
    mk_param(cx, def.space, def.index, def.name)
}

pub fn mk_open<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> { mk_t(cx, ty_open(ty)) }

impl<'tcx> TyS<'tcx> {
    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```notrust
    /// int => { int }
    /// Foo<Bar<int>> => { Foo<Bar<int>>, Bar<int>, int }
    /// [int] => { [int], int }
    /// ```
    pub fn walk(&'tcx self) -> TypeWalker<'tcx> {
        TypeWalker::new(self)
    }

    /// Iterator that walks types reachable from `self`, in
    /// depth-first order. Note that this is a shallow walk. For
    /// example:
    ///
    /// ```notrust
    /// int => { }
    /// Foo<Bar<int>> => { Bar<int>, int }
    /// [int] => { int }
    /// ```
    pub fn walk_children(&'tcx self) -> TypeWalker<'tcx> {
        // Walks type reachable from `self` but not `self
        let mut walker = self.walk();
        let r = walker.next();
        assert_eq!(r, Some(self));
        walker
    }

    pub fn as_opt_param_ty(&self) -> Option<ty::ParamTy> {
        match self.sty {
            ty::ty_param(ref d) => Some(d.clone()),
            _ => None,
        }
    }
}

pub fn walk_ty<'tcx, F>(ty_root: Ty<'tcx>, mut f: F)
    where F: FnMut(Ty<'tcx>),
{
    for ty in ty_root.walk() {
        f(ty);
    }
}

/// Walks `ty` and any types appearing within `ty`, invoking the
/// callback `f` on each type. If the callback returns false, then the
/// children of the current type are ignored.
///
/// Note: prefer `ty.walk()` where possible.
pub fn maybe_walk_ty<'tcx,F>(ty_root: Ty<'tcx>, mut f: F)
    where F : FnMut(Ty<'tcx>) -> bool
{
    let mut walker = ty_root.walk();
    while let Some(ty) = walker.next() {
        if !f(ty) {
            walker.skip_current_subtree();
        }
    }
}

// Folds types from the bottom up.
pub fn fold_ty<'tcx, F>(cx: &ctxt<'tcx>, t0: Ty<'tcx>,
                        fldop: F)
                        -> Ty<'tcx> where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
{
    let mut f = ty_fold::BottomUpFolder {tcx: cx, fldop: fldop};
    f.fold_ty(t0)
}

impl ParamTy {
    pub fn new(space: subst::ParamSpace,
               index: u32,
               name: ast::Name)
               -> ParamTy {
        ParamTy { space: space, idx: index, name: name }
    }

    pub fn for_self() -> ParamTy {
        ParamTy::new(subst::SelfSpace, 0, special_idents::type_self.name)
    }

    pub fn for_def(def: &TypeParameterDef) -> ParamTy {
        ParamTy::new(def.space, def.index, def.name)
    }

    pub fn to_ty<'tcx>(self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        ty::mk_param(tcx, self.space, self.idx, self.name)
    }

    pub fn is_self(&self) -> bool {
        self.space == subst::SelfSpace && self.idx == 0
    }
}

impl<'tcx> ItemSubsts<'tcx> {
    pub fn empty() -> ItemSubsts<'tcx> {
        ItemSubsts { substs: Substs::empty() }
    }

    pub fn is_noop(&self) -> bool {
        self.substs.is_noop()
    }
}

impl<'tcx> ParamBounds<'tcx> {
    pub fn empty() -> ParamBounds<'tcx> {
        ParamBounds {
            builtin_bounds: empty_builtin_bounds(),
            trait_bounds: Vec::new(),
            region_bounds: Vec::new(),
            projection_bounds: Vec::new(),
        }
    }
}

// Type utilities

pub fn type_is_nil(ty: Ty) -> bool {
    match ty.sty {
        ty_tup(ref tys) => tys.is_empty(),
        _ => false
    }
}

pub fn type_is_error(ty: Ty) -> bool {
    ty.flags.intersects(HAS_TY_ERR)
}

pub fn type_needs_subst(ty: Ty) -> bool {
    ty.flags.intersects(NEEDS_SUBST)
}

pub fn trait_ref_contains_error(tref: &ty::TraitRef) -> bool {
    tref.substs.types.any(|&ty| type_is_error(ty))
}

pub fn type_is_ty_var(ty: Ty) -> bool {
    match ty.sty {
        ty_infer(TyVar(_)) => true,
        _ => false
    }
}

pub fn type_is_bool(ty: Ty) -> bool { ty.sty == ty_bool }

pub fn type_is_self(ty: Ty) -> bool {
    match ty.sty {
        ty_param(ref p) => p.space == subst::SelfSpace,
        _ => false
    }
}

fn type_is_slice(ty: Ty) -> bool {
    match ty.sty {
        ty_ptr(mt) | ty_rptr(_, mt) => match mt.ty.sty {
            ty_vec(_, None) | ty_str => true,
            _ => false,
        },
        _ => false
    }
}

pub fn type_is_vec(ty: Ty) -> bool {
    match ty.sty {
        ty_vec(..) => true,
        ty_ptr(mt{ty, ..}) | ty_rptr(_, mt{ty, ..}) |
        ty_uniq(ty) => match ty.sty {
            ty_vec(_, None) => true,
            _ => false
        },
        _ => false
    }
}

pub fn type_is_structural(ty: Ty) -> bool {
    match ty.sty {
      ty_struct(..) | ty_tup(_) | ty_enum(..) |
      ty_vec(_, Some(_)) | ty_closure(..) => true,
      _ => type_is_slice(ty) | type_is_trait(ty)
    }
}

pub fn type_is_simd(cx: &ctxt, ty: Ty) -> bool {
    match ty.sty {
        ty_struct(did, _) => lookup_simd(cx, did),
        _ => false
    }
}

pub fn sequence_element_type<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty_vec(ty, _) => ty,
        ty_str => mk_mach_uint(cx, ast::TyU8),
        ty_open(ty) => sequence_element_type(cx, ty),
        _ => cx.sess.bug(&format!("sequence_element_type called on non-sequence value: {}",
                                 ty_to_string(cx, ty))[]),
    }
}

pub fn simd_type<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty_struct(did, substs) => {
            let fields = lookup_struct_fields(cx, did);
            lookup_field_type(cx, did, fields[0].id, substs)
        }
        _ => panic!("simd_type called on invalid type")
    }
}

pub fn simd_size(cx: &ctxt, ty: Ty) -> uint {
    match ty.sty {
        ty_struct(did, _) => {
            let fields = lookup_struct_fields(cx, did);
            fields.len()
        }
        _ => panic!("simd_size called on invalid type")
    }
}

pub fn type_is_region_ptr(ty: Ty) -> bool {
    match ty.sty {
        ty_rptr(..) => true,
        _ => false
    }
}

pub fn type_is_unsafe_ptr(ty: Ty) -> bool {
    match ty.sty {
      ty_ptr(_) => return true,
      _ => return false
    }
}

pub fn type_is_unique(ty: Ty) -> bool {
    match ty.sty {
        ty_uniq(_) => true,
        _ => false
    }
}

/*
 A scalar type is one that denotes an atomic datum, with no sub-components.
 (A ty_ptr is scalar because it represents a non-managed pointer, so its
 contents are abstract to rustc.)
*/
pub fn type_is_scalar(ty: Ty) -> bool {
    match ty.sty {
      ty_bool | ty_char | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_infer(IntVar(_)) | ty_infer(FloatVar(_)) |
      ty_bare_fn(..) | ty_ptr(_) => true,
      _ => false
    }
}

/// Returns true if this type is a floating point type and false otherwise.
pub fn type_is_floating_point(ty: Ty) -> bool {
    match ty.sty {
        ty_float(_) => true,
        _ => false,
    }
}

/// Type contents is how the type checker reasons about kinds.
/// They track what kinds of things are found within a type.  You can
/// think of them as kind of an "anti-kind".  They track the kinds of values
/// and thinks that are contained in types.  Having a larger contents for
/// a type tends to rule that type *out* from various kinds.  For example,
/// a type that contains a reference is not sendable.
///
/// The reason we compute type contents and not kinds is that it is
/// easier for me (nmatsakis) to think about what is contained within
/// a type than to think about what is *not* contained within a type.
#[derive(Clone, Copy)]
pub struct TypeContents {
    pub bits: u64
}

macro_rules! def_type_content_sets {
    (mod $mname:ident { $($name:ident = $bits:expr),+ }) => {
        #[allow(non_snake_case)]
        mod $mname {
            use middle::ty::TypeContents;
            $(
                #[allow(non_upper_case_globals)]
                pub const $name: TypeContents = TypeContents { bits: $bits };
             )+
        }
    }
}

def_type_content_sets! {
    mod TC {
        None                                = 0b0000_0000__0000_0000__0000,

        // Things that are interior to the value (first nibble):
        InteriorUnsized                     = 0b0000_0000__0000_0000__0001,
        InteriorUnsafe                      = 0b0000_0000__0000_0000__0010,
        InteriorParam                       = 0b0000_0000__0000_0000__0100,
        // InteriorAll                         = 0b00000000__00000000__1111,

        // Things that are owned by the value (second and third nibbles):
        OwnsOwned                           = 0b0000_0000__0000_0001__0000,
        OwnsDtor                            = 0b0000_0000__0000_0010__0000,
        OwnsManaged /* see [1] below */     = 0b0000_0000__0000_0100__0000,
        OwnsAll                             = 0b0000_0000__1111_1111__0000,

        // Things that are reachable by the value in any way (fourth nibble):
        ReachesBorrowed                     = 0b0000_0010__0000_0000__0000,
        // ReachesManaged /* see [1] below */  = 0b0000_0100__0000_0000__0000,
        ReachesMutable                      = 0b0000_1000__0000_0000__0000,
        ReachesFfiUnsafe                    = 0b0010_0000__0000_0000__0000,
        ReachesAll                          = 0b0011_1111__0000_0000__0000,

        // Things that mean drop glue is necessary
        NeedsDrop                           = 0b0000_0000__0000_0111__0000,

        // Things that prevent values from being considered sized
        Nonsized                            = 0b0000_0000__0000_0000__0001,

        // Bits to set when a managed value is encountered
        //
        // [1] Do not set the bits TC::OwnsManaged or
        //     TC::ReachesManaged directly, instead reference
        //     TC::Managed to set them both at once.
        Managed                             = 0b0000_0100__0000_0100__0000,

        // All bits
        All                                 = 0b1111_1111__1111_1111__1111
    }
}

impl TypeContents {
    pub fn when(&self, cond: bool) -> TypeContents {
        if cond {*self} else {TC::None}
    }

    pub fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    pub fn owns_managed(&self) -> bool {
        self.intersects(TC::OwnsManaged)
    }

    pub fn owns_owned(&self) -> bool {
        self.intersects(TC::OwnsOwned)
    }

    pub fn is_sized(&self, _: &ctxt) -> bool {
        !self.intersects(TC::Nonsized)
    }

    pub fn interior_param(&self) -> bool {
        self.intersects(TC::InteriorParam)
    }

    pub fn interior_unsafe(&self) -> bool {
        self.intersects(TC::InteriorUnsafe)
    }

    pub fn interior_unsized(&self) -> bool {
        self.intersects(TC::InteriorUnsized)
    }

    pub fn needs_drop(&self, _: &ctxt) -> bool {
        self.intersects(TC::NeedsDrop)
    }

    /// Includes only those bits that still apply when indirected through a `Box` pointer
    pub fn owned_pointer(&self) -> TypeContents {
        TC::OwnsOwned | (
            *self & (TC::OwnsAll | TC::ReachesAll))
    }

    /// Includes only those bits that still apply when indirected through a reference (`&`)
    pub fn reference(&self, bits: TypeContents) -> TypeContents {
        bits | (
            *self & TC::ReachesAll)
    }

    /// Includes only those bits that still apply when indirected through a managed pointer (`@`)
    pub fn managed_pointer(&self) -> TypeContents {
        TC::Managed | (
            *self & TC::ReachesAll)
    }

    /// Includes only those bits that still apply when indirected through an unsafe pointer (`*`)
    pub fn unsafe_pointer(&self) -> TypeContents {
        *self & TC::ReachesAll
    }

    pub fn union<T, F>(v: &[T], mut f: F) -> TypeContents where
        F: FnMut(&T) -> TypeContents,
    {
        v.iter().fold(TC::None, |tc, ty| tc | f(ty))
    }

    pub fn has_dtor(&self) -> bool {
        self.intersects(TC::OwnsDtor)
    }
}

impl ops::BitOr for TypeContents {
    type Output = TypeContents;

    fn bitor(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits | other.bits}
    }
}

impl ops::BitAnd for TypeContents {
    type Output = TypeContents;

    fn bitand(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & other.bits}
    }
}

impl ops::Sub for TypeContents {
    type Output = TypeContents;

    fn sub(self, other: TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & !other.bits}
    }
}

impl fmt::Debug for TypeContents {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeContents({:b})", self.bits)
    }
}

pub fn type_interior_is_unsafe<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    type_contents(cx, ty).interior_unsafe()
}

pub fn type_contents<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> TypeContents {
    return memoized(&cx.tc_cache, ty, |ty| {
        tc_ty(cx, ty, &mut FnvHashMap())
    });

    fn tc_ty<'tcx>(cx: &ctxt<'tcx>,
                   ty: Ty<'tcx>,
                   cache: &mut FnvHashMap<Ty<'tcx>, TypeContents>) -> TypeContents
    {
        // Subtle: Note that we are *not* using cx.tc_cache here but rather a
        // private cache for this walk.  This is needed in the case of cyclic
        // types like:
        //
        //     struct List { next: Box<Option<List>>, ... }
        //
        // When computing the type contents of such a type, we wind up deeply
        // recursing as we go.  So when we encounter the recursive reference
        // to List, we temporarily use TC::None as its contents.  Later we'll
        // patch up the cache with the correct value, once we've computed it
        // (this is basically a co-inductive process, if that helps).  So in
        // the end we'll compute TC::OwnsOwned, in this case.
        //
        // The problem is, as we are doing the computation, we will also
        // compute an *intermediate* contents for, e.g., Option<List> of
        // TC::None.  This is ok during the computation of List itself, but if
        // we stored this intermediate value into cx.tc_cache, then later
        // requests for the contents of Option<List> would also yield TC::None
        // which is incorrect.  This value was computed based on the crutch
        // value for the type contents of list.  The correct value is
        // TC::OwnsOwned.  This manifested as issue #4821.
        match cache.get(&ty) {
            Some(tc) => { return *tc; }
            None => {}
        }
        match cx.tc_cache.borrow().get(&ty) {    // Must check both caches!
            Some(tc) => { return *tc; }
            None => {}
        }
        cache.insert(ty, TC::None);

        let result = match ty.sty {
            // uint and int are ffi-unsafe
            ty_uint(ast::TyUs(_)) | ty_int(ast::TyIs(_)) => {
                TC::ReachesFfiUnsafe
            }

            // Scalar and unique types are sendable, and durable
            ty_infer(ty::FreshIntTy(_)) |
            ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
            ty_bare_fn(..) | ty::ty_char => {
                TC::None
            }

            ty_uniq(typ) => {
                TC::ReachesFfiUnsafe | match typ.sty {
                    ty_str => TC::OwnsOwned,
                    _ => tc_ty(cx, typ, cache).owned_pointer(),
                }
            }

            ty_trait(box TyTrait { ref bounds, .. }) => {
                object_contents(bounds) | TC::ReachesFfiUnsafe | TC::Nonsized
            }

            ty_ptr(ref mt) => {
                tc_ty(cx, mt.ty, cache).unsafe_pointer()
            }

            ty_rptr(r, ref mt) => {
                TC::ReachesFfiUnsafe | match mt.ty.sty {
                    ty_str => borrowed_contents(*r, ast::MutImmutable),
                    ty_vec(..) => tc_ty(cx, mt.ty, cache).reference(borrowed_contents(*r,
                                                                                      mt.mutbl)),
                    _ => tc_ty(cx, mt.ty, cache).reference(borrowed_contents(*r, mt.mutbl)),
                }
            }

            ty_vec(ty, Some(_)) => {
                tc_ty(cx, ty, cache)
            }

            ty_vec(ty, None) => {
                tc_ty(cx, ty, cache) | TC::Nonsized
            }
            ty_str => TC::Nonsized,

            ty_struct(did, substs) => {
                let flds = struct_fields(cx, did, substs);
                let mut res =
                    TypeContents::union(&flds[..],
                                        |f| tc_mt(cx, f.mt, cache));

                if !lookup_repr_hints(cx, did).contains(&attr::ReprExtern) {
                    res = res | TC::ReachesFfiUnsafe;
                }

                if ty::has_dtor(cx, did) {
                    res = res | TC::OwnsDtor;
                }
                apply_lang_items(cx, did, res)
            }

            ty_closure(did, r, substs) => {
                // FIXME(#14449): `borrowed_contents` below assumes `&mut` closure.
                let param_env = ty::empty_parameter_environment(cx);
                let upvars = closure_upvars(&param_env, did, substs).unwrap();
                TypeContents::union(&upvars,
                                    |f| tc_ty(cx, &f.ty, cache))
                    | borrowed_contents(*r, MutMutable)
            }

            ty_tup(ref tys) => {
                TypeContents::union(&tys[..],
                                    |ty| tc_ty(cx, *ty, cache))
            }

            ty_enum(did, substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                let mut res =
                    TypeContents::union(&variants[..], |variant| {
                        TypeContents::union(&variant.args[],
                                            |arg_ty| {
                            tc_ty(cx, *arg_ty, cache)
                        })
                    });

                if ty::has_dtor(cx, did) {
                    res = res | TC::OwnsDtor;
                }

                if variants.len() != 0 {
                    let repr_hints = lookup_repr_hints(cx, did);
                    if repr_hints.len() > 1 {
                        // this is an error later on, but this type isn't safe
                        res = res | TC::ReachesFfiUnsafe;
                    }

                    match repr_hints.get(0) {
                        Some(h) => if !h.is_ffi_safe() {
                            res = res | TC::ReachesFfiUnsafe;
                        },
                        // ReprAny
                        None => {
                            res = res | TC::ReachesFfiUnsafe;

                            // We allow ReprAny enums if they are eligible for
                            // the nullable pointer optimization and the
                            // contained type is an `extern fn`

                            if variants.len() == 2 {
                                let mut data_idx = 0;

                                if variants[0].args.len() == 0 {
                                    data_idx = 1;
                                }

                                if variants[data_idx].args.len() == 1 {
                                    match variants[data_idx].args[0].sty {
                                        ty_bare_fn(..) => { res = res - TC::ReachesFfiUnsafe; }
                                        _ => { }
                                    }
                                }
                            }
                        }
                    }
                }


                apply_lang_items(cx, did, res)
            }

            ty_projection(..) |
            ty_param(_) => {
                TC::All
            }

            ty_open(ty) => {
                let result = tc_ty(cx, ty, cache);
                assert!(!result.is_sized(cx));
                result.unsafe_pointer() | TC::Nonsized
            }

            ty_infer(_) |
            ty_err => {
                cx.sess.bug("asked to compute contents of error type");
            }
        };

        cache.insert(ty, result);
        result
    }

    fn tc_mt<'tcx>(cx: &ctxt<'tcx>,
                   mt: mt<'tcx>,
                   cache: &mut FnvHashMap<Ty<'tcx>, TypeContents>) -> TypeContents
    {
        let mc = TC::ReachesMutable.when(mt.mutbl == MutMutable);
        mc | tc_ty(cx, mt.ty, cache)
    }

    fn apply_lang_items(cx: &ctxt, did: ast::DefId, tc: TypeContents)
                        -> TypeContents {
        if Some(did) == cx.lang_items.managed_bound() {
            tc | TC::Managed
        } else if Some(did) == cx.lang_items.unsafe_cell_type() {
            tc | TC::InteriorUnsafe
        } else {
            tc
        }
    }

    /// Type contents due to containing a reference with the region `region` and borrow kind `bk`
    fn borrowed_contents(region: ty::Region,
                         mutbl: ast::Mutability)
                         -> TypeContents {
        let b = match mutbl {
            ast::MutMutable => TC::ReachesMutable,
            ast::MutImmutable => TC::None,
        };
        b | (TC::ReachesBorrowed).when(region != ty::ReStatic)
    }

    fn object_contents(bounds: &ExistentialBounds) -> TypeContents {
        // These are the type contents of the (opaque) interior. We
        // make no assumptions (other than that it cannot have an
        // in-scope type parameter within, which makes no sense).
        let mut tc = TC::All - TC::InteriorParam;
        for bound in &bounds.builtin_bounds {
            tc = tc - match bound {
                BoundSync | BoundSend | BoundCopy => TC::None,
                BoundSized => TC::Nonsized,
            };
        }
        return tc;
    }
}

fn type_impls_bound<'a,'tcx>(param_env: &ParameterEnvironment<'a,'tcx>,
                             cache: &RefCell<HashMap<Ty<'tcx>,bool>>,
                             ty: Ty<'tcx>,
                             bound: ty::BuiltinBound,
                             span: Span)
                             -> bool
{
    assert!(!ty::type_needs_infer(ty));

    if !type_has_params(ty) && !type_has_self(ty) {
        match cache.borrow().get(&ty) {
            None => {}
            Some(&result) => {
                debug!("type_impls_bound({}, {:?}) = {:?} (cached)",
                       ty.repr(param_env.tcx),
                       bound,
                       result);
                return result
            }
        }
    }

    let infcx = infer::new_infer_ctxt(param_env.tcx);

    let is_impld = traits::type_known_to_meet_builtin_bound(&infcx, param_env, ty, bound, span);

    debug!("type_impls_bound({}, {:?}) = {:?}",
           ty.repr(param_env.tcx),
           bound,
           is_impld);

    if !type_has_params(ty) && !type_has_self(ty) {
        let old_value = cache.borrow_mut().insert(ty, is_impld);
        assert!(old_value.is_none());
    }

    is_impld
}

pub fn type_moves_by_default<'a,'tcx>(param_env: &ParameterEnvironment<'a,'tcx>,
                                      span: Span,
                                      ty: Ty<'tcx>)
                                      -> bool
{
    let tcx = param_env.tcx;
    !type_impls_bound(param_env, &tcx.type_impls_copy_cache, ty, ty::BoundCopy, span)
}

pub fn type_is_sized<'a,'tcx>(param_env: &ParameterEnvironment<'a,'tcx>,
                              span: Span,
                              ty: Ty<'tcx>)
                              -> bool
{
    let tcx = param_env.tcx;
    type_impls_bound(param_env, &tcx.type_impls_sized_cache, ty, ty::BoundSized, span)
}

pub fn is_ffi_safe<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    !type_contents(cx, ty).intersects(TC::ReachesFfiUnsafe)
}

// True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
pub fn is_instantiable<'tcx>(cx: &ctxt<'tcx>, r_ty: Ty<'tcx>) -> bool {
    fn type_requires<'tcx>(cx: &ctxt<'tcx>, seen: &mut Vec<DefId>,
                           r_ty: Ty<'tcx>, ty: Ty<'tcx>) -> bool {
        debug!("type_requires({:?}, {:?})?",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty));

        let r = r_ty == ty || subtypes_require(cx, seen, r_ty, ty);

        debug!("type_requires({:?}, {:?})? {:?}",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty),
               r);
        return r;
    }

    fn subtypes_require<'tcx>(cx: &ctxt<'tcx>, seen: &mut Vec<DefId>,
                              r_ty: Ty<'tcx>, ty: Ty<'tcx>) -> bool {
        debug!("subtypes_require({:?}, {:?})?",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty));

        let r = match ty.sty {
            // fixed length vectors need special treatment compared to
            // normal vectors, since they don't necessarily have the
            // possibility to have length zero.
            ty_vec(_, Some(0)) => false, // don't need no contents
            ty_vec(ty, Some(_)) => type_requires(cx, seen, r_ty, ty),

            ty_bool |
            ty_char |
            ty_int(_) |
            ty_uint(_) |
            ty_float(_) |
            ty_str |
            ty_bare_fn(..) |
            ty_param(_) |
            ty_projection(_) |
            ty_vec(_, None) => {
                false
            }
            ty_uniq(typ) | ty_open(typ) => {
                type_requires(cx, seen, r_ty, typ)
            }
            ty_rptr(_, ref mt) => {
                type_requires(cx, seen, r_ty, mt.ty)
            }

            ty_ptr(..) => {
                false           // unsafe ptrs can always be NULL
            }

            ty_trait(..) => {
                false
            }

            ty_struct(ref did, _) if seen.contains(did) => {
                false
            }

            ty_struct(did, substs) => {
                seen.push(did);
                let fields = struct_fields(cx, did, substs);
                let r = fields.iter().any(|f| type_requires(cx, seen, r_ty, f.mt.ty));
                seen.pop().unwrap();
                r
            }

            ty_err |
            ty_infer(_) |
            ty_closure(..) => {
                // this check is run on type definitions, so we don't expect to see
                // inference by-products or closure types
                cx.sess.bug(&format!("requires check invoked on inapplicable type: {:?}", ty))
            }

            ty_tup(ref ts) => {
                ts.iter().any(|ty| type_requires(cx, seen, r_ty, *ty))
            }

            ty_enum(ref did, _) if seen.contains(did) => {
                false
            }

            ty_enum(did, substs) => {
                seen.push(did);
                let vs = enum_variants(cx, did);
                let r = !vs.is_empty() && vs.iter().all(|variant| {
                    variant.args.iter().any(|aty| {
                        let sty = aty.subst(cx, substs);
                        type_requires(cx, seen, r_ty, sty)
                    })
                });
                seen.pop().unwrap();
                r
            }
        };

        debug!("subtypes_require({:?}, {:?})? {:?}",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty),
               r);

        return r;
    }

    let mut seen = Vec::new();
    !subtypes_require(cx, &mut seen, r_ty, r_ty)
}

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
///
/// The ordering of the cases is significant. They are sorted so that cmp::max
/// will keep the "more erroneous" of two values.
#[derive(Copy, PartialOrd, Ord, Eq, PartialEq, Debug)]
pub enum Representability {
    Representable,
    ContainsRecursive,
    SelfRecursive,
}

/// Check whether a type is representable. This means it cannot contain unboxed
/// structural recursion. This check is needed for structs and enums.
pub fn is_type_representable<'tcx>(cx: &ctxt<'tcx>, sp: Span, ty: Ty<'tcx>)
                                   -> Representability {

    // Iterate until something non-representable is found
    fn find_nonrepresentable<'tcx, It: Iterator<Item=Ty<'tcx>>>(cx: &ctxt<'tcx>, sp: Span,
                                                                seen: &mut Vec<Ty<'tcx>>,
                                                                iter: It)
                                                                -> Representability {
        iter.fold(Representable,
                  |r, ty| cmp::max(r, is_type_structurally_recursive(cx, sp, seen, ty)))
    }

    fn are_inner_types_recursive<'tcx>(cx: &ctxt<'tcx>, sp: Span,
                                       seen: &mut Vec<Ty<'tcx>>, ty: Ty<'tcx>)
                                       -> Representability {
        match ty.sty {
            ty_tup(ref ts) => {
                find_nonrepresentable(cx, sp, seen, ts.iter().cloned())
            }
            // Fixed-length vectors.
            // FIXME(#11924) Behavior undecided for zero-length vectors.
            ty_vec(ty, Some(_)) => {
                is_type_structurally_recursive(cx, sp, seen, ty)
            }
            ty_struct(did, substs) => {
                let fields = struct_fields(cx, did, substs);
                find_nonrepresentable(cx, sp, seen, fields.iter().map(|f| f.mt.ty))
            }
            ty_enum(did, substs) => {
                let vs = enum_variants(cx, did);
                let iter = vs.iter()
                    .flat_map(|variant| { variant.args.iter() })
                    .map(|aty| { aty.subst_spanned(cx, substs, Some(sp)) });

                find_nonrepresentable(cx, sp, seen, iter)
            }
            ty_closure(..) => {
                // this check is run on type definitions, so we don't expect
                // to see closure types
                cx.sess.bug(&format!("requires check invoked on inapplicable type: {:?}", ty))
            }
            _ => Representable,
        }
    }

    fn same_struct_or_enum_def_id(ty: Ty, did: DefId) -> bool {
        match ty.sty {
            ty_struct(ty_did, _) | ty_enum(ty_did, _) => {
                 ty_did == did
            }
            _ => false
        }
    }

    fn same_type<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        match (&a.sty, &b.sty) {
            (&ty_struct(did_a, ref substs_a), &ty_struct(did_b, ref substs_b)) |
            (&ty_enum(did_a, ref substs_a), &ty_enum(did_b, ref substs_b)) => {
                if did_a != did_b {
                    return false;
                }

                let types_a = substs_a.types.get_slice(subst::TypeSpace);
                let types_b = substs_b.types.get_slice(subst::TypeSpace);

                let pairs = types_a.iter().zip(types_b.iter());

                pairs.all(|(&a, &b)| same_type(a, b))
            }
            _ => {
                a == b
            }
        }
    }

    // Does the type `ty` directly (without indirection through a pointer)
    // contain any types on stack `seen`?
    fn is_type_structurally_recursive<'tcx>(cx: &ctxt<'tcx>, sp: Span,
                                            seen: &mut Vec<Ty<'tcx>>,
                                            ty: Ty<'tcx>) -> Representability {
        debug!("is_type_structurally_recursive: {:?}",
               ::util::ppaux::ty_to_string(cx, ty));

        match ty.sty {
            ty_struct(did, _) | ty_enum(did, _) => {
                {
                    // Iterate through stack of previously seen types.
                    let mut iter = seen.iter();

                    // The first item in `seen` is the type we are actually curious about.
                    // We want to return SelfRecursive if this type contains itself.
                    // It is important that we DON'T take generic parameters into account
                    // for this check, so that Bar<T> in this example counts as SelfRecursive:
                    //
                    // struct Foo;
                    // struct Bar<T> { x: Bar<Foo> }

                    match iter.next() {
                        Some(&seen_type) => {
                            if same_struct_or_enum_def_id(seen_type, did) {
                                debug!("SelfRecursive: {:?} contains {:?}",
                                       ::util::ppaux::ty_to_string(cx, seen_type),
                                       ::util::ppaux::ty_to_string(cx, ty));
                                return SelfRecursive;
                            }
                        }
                        None => {}
                    }

                    // We also need to know whether the first item contains other types that
                    // are structurally recursive. If we don't catch this case, we will recurse
                    // infinitely for some inputs.
                    //
                    // It is important that we DO take generic parameters into account here,
                    // so that code like this is considered SelfRecursive, not ContainsRecursive:
                    //
                    // struct Foo { Option<Option<Foo>> }

                    for &seen_type in iter {
                        if same_type(ty, seen_type) {
                            debug!("ContainsRecursive: {:?} contains {:?}",
                                   ::util::ppaux::ty_to_string(cx, seen_type),
                                   ::util::ppaux::ty_to_string(cx, ty));
                            return ContainsRecursive;
                        }
                    }
                }

                // For structs and enums, track all previously seen types by pushing them
                // onto the 'seen' stack.
                seen.push(ty);
                let out = are_inner_types_recursive(cx, sp, seen, ty);
                seen.pop();
                out
            }
            _ => {
                // No need to push in other cases.
                are_inner_types_recursive(cx, sp, seen, ty)
            }
        }
    }

    debug!("is_type_representable: {:?}",
           ::util::ppaux::ty_to_string(cx, ty));

    // To avoid a stack overflow when checking an enum variant or struct that
    // contains a different, structurally recursive type, maintain a stack
    // of seen types and check recursion for each of them (issues #3008, #3779).
    let mut seen: Vec<Ty> = Vec::new();
    let r = is_type_structurally_recursive(cx, sp, &mut seen, ty);
    debug!("is_type_representable: {:?} is {:?}",
           ::util::ppaux::ty_to_string(cx, ty), r);
    r
}

pub fn type_is_trait(ty: Ty) -> bool {
    type_trait_info(ty).is_some()
}

pub fn type_trait_info<'tcx>(ty: Ty<'tcx>) -> Option<&'tcx TyTrait<'tcx>> {
    match ty.sty {
        ty_uniq(ty) | ty_rptr(_, mt { ty, ..}) | ty_ptr(mt { ty, ..}) => match ty.sty {
            ty_trait(ref t) => Some(&**t),
            _ => None
        },
        ty_trait(ref t) => Some(&**t),
        _ => None
    }
}

pub fn type_is_integral(ty: Ty) -> bool {
    match ty.sty {
      ty_infer(IntVar(_)) | ty_int(_) | ty_uint(_) => true,
      _ => false
    }
}

pub fn type_is_fresh(ty: Ty) -> bool {
    match ty.sty {
      ty_infer(FreshTy(_)) => true,
      ty_infer(FreshIntTy(_)) => true,
      _ => false
    }
}

pub fn type_is_uint(ty: Ty) -> bool {
    match ty.sty {
      ty_infer(IntVar(_)) | ty_uint(ast::TyUs(_)) => true,
      _ => false
    }
}

pub fn type_is_char(ty: Ty) -> bool {
    match ty.sty {
        ty_char => true,
        _ => false
    }
}

pub fn type_is_bare_fn(ty: Ty) -> bool {
    match ty.sty {
        ty_bare_fn(..) => true,
        _ => false
    }
}

pub fn type_is_bare_fn_item(ty: Ty) -> bool {
    match ty.sty {
        ty_bare_fn(Some(_), _) => true,
        _ => false
    }
}

pub fn type_is_fp(ty: Ty) -> bool {
    match ty.sty {
      ty_infer(FloatVar(_)) | ty_float(_) => true,
      _ => false
    }
}

pub fn type_is_numeric(ty: Ty) -> bool {
    return type_is_integral(ty) || type_is_fp(ty);
}

pub fn type_is_signed(ty: Ty) -> bool {
    match ty.sty {
      ty_int(_) => true,
      _ => false
    }
}

pub fn type_is_machine(ty: Ty) -> bool {
    match ty.sty {
        ty_int(ast::TyIs(_)) | ty_uint(ast::TyUs(_)) => false,
        ty_int(..) | ty_uint(..) | ty_float(..) => true,
        _ => false
    }
}

// Whether a type is enum like, that is an enum type with only nullary
// constructors
pub fn type_is_c_like_enum(cx: &ctxt, ty: Ty) -> bool {
    match ty.sty {
        ty_enum(did, _) => {
            let variants = enum_variants(cx, did);
            if variants.len() == 0 {
                false
            } else {
                variants.iter().all(|v| v.args.len() == 0)
            }
        }
        _ => false
    }
}

// Returns the type and mutability of *ty.
//
// The parameter `explicit` indicates if this is an *explicit* dereference.
// Some types---notably unsafe ptrs---can only be dereferenced explicitly.
pub fn deref<'tcx>(ty: Ty<'tcx>, explicit: bool) -> Option<mt<'tcx>> {
    match ty.sty {
        ty_uniq(ty) => {
            Some(mt {
                ty: ty,
                mutbl: ast::MutImmutable,
            })
        },
        ty_rptr(_, mt) => Some(mt),
        ty_ptr(mt) if explicit => Some(mt),
        _ => None
    }
}

pub fn close_type<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty_open(ty) => mk_rptr(cx, cx.mk_region(ReStatic), mt {ty: ty, mutbl:ast::MutImmutable}),
        _ => cx.sess.bug(&format!("Trying to close a non-open type {}",
                                 ty_to_string(cx, ty))[])
    }
}

pub fn type_content<'tcx>(ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty_uniq(ty) => ty,
        ty_rptr(_, mt) |ty_ptr(mt) => mt.ty,
        _ => ty
    }
}

// Extract the unsized type in an open type (or just return ty if it is not open).
pub fn unopen_type<'tcx>(ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty_open(ty) => ty,
        _ => ty
    }
}

// Returns the type of ty[i]
pub fn index<'tcx>(ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.sty {
        ty_vec(ty, _) => Some(ty),
        _ => None
    }
}

// Returns the type of elements contained within an 'array-like' type.
// This is exactly the same as the above, except it supports strings,
// which can't actually be indexed.
pub fn array_element_ty<'tcx>(tcx: &ctxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.sty {
        ty_vec(ty, _) => Some(ty),
        ty_str => Some(tcx.types.u8),
        _ => None
    }
}

/// Returns the type of element at index `i` in tuple or tuple-like type `t`.
/// For an enum `t`, `variant` is None only if `t` is a univariant enum.
pub fn positional_element_ty<'tcx>(cx: &ctxt<'tcx>,
                                   ty: Ty<'tcx>,
                                   i: uint,
                                   variant: Option<ast::DefId>) -> Option<Ty<'tcx>> {

    match (&ty.sty, variant) {
        (&ty_tup(ref v), None) => v.get(i).cloned(),


        (&ty_struct(def_id, substs), None) => lookup_struct_fields(cx, def_id)
            .get(i)
            .map(|&t|lookup_item_type(cx, t.id).ty.subst(cx, substs)),

        (&ty_enum(def_id, substs), Some(variant_def_id)) => {
            let variant_info = enum_variant_with_id(cx, def_id, variant_def_id);
            variant_info.args.get(i).map(|t|t.subst(cx, substs))
        }

        (&ty_enum(def_id, substs), None) => {
            assert!(enum_is_univariant(cx, def_id));
            let enum_variants = enum_variants(cx, def_id);
            let variant_info = &(*enum_variants)[0];
            variant_info.args.get(i).map(|t|t.subst(cx, substs))
        }

        _ => None
    }
}

/// Returns the type of element at field `n` in struct or struct-like type `t`.
/// For an enum `t`, `variant` must be some def id.
pub fn named_element_ty<'tcx>(cx: &ctxt<'tcx>,
                              ty: Ty<'tcx>,
                              n: ast::Name,
                              variant: Option<ast::DefId>) -> Option<Ty<'tcx>> {

    match (&ty.sty, variant) {
        (&ty_struct(def_id, substs), None) => {
            let r = lookup_struct_fields(cx, def_id);
            r.iter().find(|f| f.name == n)
                .map(|&f| lookup_field_type(cx, def_id, f.id, substs))
        }
        (&ty_enum(def_id, substs), Some(variant_def_id)) => {
            let variant_info = enum_variant_with_id(cx, def_id, variant_def_id);
            variant_info.arg_names.as_ref()
                .expect("must have struct enum variant if accessing a named fields")
                .iter().zip(variant_info.args.iter())
                .find(|&(ident, _)| ident.name == n)
                .map(|(_ident, arg_t)| arg_t.subst(cx, substs))
        }
        _ => None
    }
}

pub fn node_id_to_trait_ref<'tcx>(cx: &ctxt<'tcx>, id: ast::NodeId)
                                  -> Rc<ty::TraitRef<'tcx>> {
    match cx.trait_refs.borrow().get(&id) {
        Some(ty) => ty.clone(),
        None => cx.sess.bug(
            &format!("node_id_to_trait_ref: no trait ref for node `{}`",
                    cx.map.node_to_string(id))[])
    }
}

pub fn node_id_to_type<'tcx>(cx: &ctxt<'tcx>, id: ast::NodeId) -> Ty<'tcx> {
    match node_id_to_type_opt(cx, id) {
       Some(ty) => ty,
       None => cx.sess.bug(
           &format!("node_id_to_type: no type for node `{}`",
                   cx.map.node_to_string(id))[])
    }
}

pub fn node_id_to_type_opt<'tcx>(cx: &ctxt<'tcx>, id: ast::NodeId) -> Option<Ty<'tcx>> {
    match cx.node_types.borrow().get(&id) {
       Some(&ty) => Some(ty),
       None => None
    }
}

pub fn node_id_item_substs<'tcx>(cx: &ctxt<'tcx>, id: ast::NodeId) -> ItemSubsts<'tcx> {
    match cx.item_substs.borrow().get(&id) {
      None => ItemSubsts::empty(),
      Some(ts) => ts.clone(),
    }
}

pub fn fn_is_variadic(fty: Ty) -> bool {
    match fty.sty {
        ty_bare_fn(_, ref f) => f.sig.0.variadic,
        ref s => {
            panic!("fn_is_variadic() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_fn_sig<'tcx>(fty: Ty<'tcx>) -> &'tcx PolyFnSig<'tcx> {
    match fty.sty {
        ty_bare_fn(_, ref f) => &f.sig,
        ref s => {
            panic!("ty_fn_sig() called on non-fn type: {:?}", s)
        }
    }
}

/// Returns the ABI of the given function.
pub fn ty_fn_abi(fty: Ty) -> abi::Abi {
    match fty.sty {
        ty_bare_fn(_, ref f) => f.abi,
        _ => panic!("ty_fn_abi() called on non-fn type"),
    }
}

// Type accessors for substructures of types
pub fn ty_fn_args<'tcx>(fty: Ty<'tcx>) -> ty::Binder<Vec<Ty<'tcx>>> {
    ty_fn_sig(fty).inputs()
}

pub fn ty_fn_ret<'tcx>(fty: Ty<'tcx>) -> Binder<FnOutput<'tcx>> {
    match fty.sty {
        ty_bare_fn(_, ref f) => f.sig.output(),
        ref s => {
            panic!("ty_fn_ret() called on non-fn type: {:?}", s)
        }
    }
}

pub fn is_fn_ty(fty: Ty) -> bool {
    match fty.sty {
        ty_bare_fn(..) => true,
        _ => false
    }
}

pub fn ty_region(tcx: &ctxt,
                 span: Span,
                 ty: Ty) -> Region {
    match ty.sty {
        ty_rptr(r, _) => *r,
        ref s => {
            tcx.sess.span_bug(
                span,
                &format!("ty_region() invoked on an inappropriate ty: {:?}",
                        s)[]);
        }
    }
}

pub fn free_region_from_def(outlives_extent: region::DestructionScopeData,
                            def: &RegionParameterDef)
    -> ty::Region
{
    let ret =
        ty::ReFree(ty::FreeRegion { scope: outlives_extent,
                                    bound_region: ty::BrNamed(def.def_id,
                                                              def.name) });
    debug!("free_region_from_def returns {:?}", ret);
    ret
}

// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
pub fn pat_ty<'tcx>(cx: &ctxt<'tcx>, pat: &ast::Pat) -> Ty<'tcx> {
    return node_id_to_type(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
// some cases, we insert `AutoAdjustment` annotations such as auto-deref or
// auto-ref.  The type returned by this function does not consider such
// adjustments.  See `expr_ty_adjusted()` instead.
//
// NB (2): This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(ty) -> T with T = int".
pub fn expr_ty<'tcx>(cx: &ctxt<'tcx>, expr: &ast::Expr) -> Ty<'tcx> {
    return node_id_to_type(cx, expr.id);
}

pub fn expr_ty_opt<'tcx>(cx: &ctxt<'tcx>, expr: &ast::Expr) -> Option<Ty<'tcx>> {
    return node_id_to_type_opt(cx, expr.id);
}

/// Returns the type of `expr`, considering any `AutoAdjustment`
/// entry recorded for that expression.
///
/// It would almost certainly be better to store the adjusted ty in with
/// the `AutoAdjustment`, but I opted not to do this because it would
/// require serializing and deserializing the type and, although that's not
/// hard to do, I just hate that code so much I didn't want to touch it
/// unless it was to fix it properly, which seemed a distraction from the
/// task at hand! -nmatsakis
pub fn expr_ty_adjusted<'tcx>(cx: &ctxt<'tcx>, expr: &ast::Expr) -> Ty<'tcx> {
    adjust_ty(cx, expr.span, expr.id, expr_ty(cx, expr),
              cx.adjustments.borrow().get(&expr.id),
              |method_call| cx.method_map.borrow().get(&method_call).map(|method| method.ty))
}

pub fn expr_span(cx: &ctxt, id: NodeId) -> Span {
    match cx.map.find(id) {
        Some(ast_map::NodeExpr(e)) => {
            e.span
        }
        Some(f) => {
            cx.sess.bug(&format!("Node id {} is not an expr: {:?}",
                                id,
                                f)[]);
        }
        None => {
            cx.sess.bug(&format!("Node id {} is not present \
                                in the node map", id)[]);
        }
    }
}

pub fn local_var_name_str(cx: &ctxt, id: NodeId) -> InternedString {
    match cx.map.find(id) {
        Some(ast_map::NodeLocal(pat)) => {
            match pat.node {
                ast::PatIdent(_, ref path1, _) => {
                    token::get_ident(path1.node)
                }
                _ => {
                    cx.sess.bug(
                        &format!("Variable id {} maps to {:?}, not local",
                                id,
                                pat)[]);
                }
            }
        }
        r => {
            cx.sess.bug(&format!("Variable id {} maps to {:?}, not local",
                                id,
                                r)[]);
        }
    }
}

/// See `expr_ty_adjusted`
pub fn adjust_ty<'tcx, F>(cx: &ctxt<'tcx>,
                          span: Span,
                          expr_id: ast::NodeId,
                          unadjusted_ty: Ty<'tcx>,
                          adjustment: Option<&AutoAdjustment<'tcx>>,
                          mut method_type: F)
                          -> Ty<'tcx> where
    F: FnMut(MethodCall) -> Option<Ty<'tcx>>,
{
    if let ty_err = unadjusted_ty.sty {
        return unadjusted_ty;
    }

    return match adjustment {
        Some(adjustment) => {
            match *adjustment {
               AdjustReifyFnPointer(_) => {
                    match unadjusted_ty.sty {
                        ty::ty_bare_fn(Some(_), b) => {
                            ty::mk_bare_fn(cx, None, b)
                        }
                        ref b => {
                            cx.sess.bug(
                                &format!("AdjustReifyFnPointer adjustment on non-fn-item: \
                                         {:?}",
                                        b)[]);
                        }
                    }
                }

                AdjustDerefRef(ref adj) => {
                    let mut adjusted_ty = unadjusted_ty;

                    if !ty::type_is_error(adjusted_ty) {
                        for i in 0..adj.autoderefs {
                            let method_call = MethodCall::autoderef(expr_id, i);
                            match method_type(method_call) {
                                Some(method_ty) => {
                                    // overloaded deref operators have all late-bound
                                    // regions fully instantiated and coverge
                                    let fn_ret =
                                        ty::no_late_bound_regions(cx,
                                                                  &ty_fn_ret(method_ty)).unwrap();
                                    adjusted_ty = fn_ret.unwrap();
                                }
                                None => {}
                            }
                            match deref(adjusted_ty, true) {
                                Some(mt) => { adjusted_ty = mt.ty; }
                                None => {
                                    cx.sess.span_bug(
                                        span,
                                        &format!("the {}th autoderef failed: \
                                                {}",
                                                i,
                                                ty_to_string(cx, adjusted_ty))
                                        []);
                                }
                            }
                        }
                    }

                    adjust_ty_for_autoref(cx, span, adjusted_ty, adj.autoref.as_ref())
                }
            }
        }
        None => unadjusted_ty
    };
}

pub fn adjust_ty_for_autoref<'tcx>(cx: &ctxt<'tcx>,
                                   span: Span,
                                   ty: Ty<'tcx>,
                                   autoref: Option<&AutoRef<'tcx>>)
                                   -> Ty<'tcx>
{
    match autoref {
        None => ty,

        Some(&AutoPtr(r, m, ref a)) => {
            let adjusted_ty = match a {
                &Some(box ref a) => adjust_ty_for_autoref(cx, span, ty, Some(a)),
                &None => ty
            };
            mk_rptr(cx, cx.mk_region(r), mt {
                ty: adjusted_ty,
                mutbl: m
            })
        }

        Some(&AutoUnsafe(m, ref a)) => {
            let adjusted_ty = match a {
                &Some(box ref a) => adjust_ty_for_autoref(cx, span, ty, Some(a)),
                &None => ty
            };
            mk_ptr(cx, mt {ty: adjusted_ty, mutbl: m})
        }

        Some(&AutoUnsize(ref k)) => unsize_ty(cx, ty, k, span),

        Some(&AutoUnsizeUniq(ref k)) => ty::mk_uniq(cx, unsize_ty(cx, ty, k, span)),
    }
}

// Take a sized type and a sizing adjustment and produce an unsized version of
// the type.
pub fn unsize_ty<'tcx>(cx: &ctxt<'tcx>,
                       ty: Ty<'tcx>,
                       kind: &UnsizeKind<'tcx>,
                       span: Span)
                       -> Ty<'tcx> {
    match kind {
        &UnsizeLength(len) => match ty.sty {
            ty_vec(ty, Some(n)) => {
                assert!(len == n);
                mk_vec(cx, ty, None)
            }
            _ => cx.sess.span_bug(span,
                                  &format!("UnsizeLength with bad sty: {:?}",
                                          ty_to_string(cx, ty))[])
        },
        &UnsizeStruct(box ref k, tp_index) => match ty.sty {
            ty_struct(did, substs) => {
                let ty_substs = substs.types.get_slice(subst::TypeSpace);
                let new_ty = unsize_ty(cx, ty_substs[tp_index], k, span);
                let mut unsized_substs = substs.clone();
                unsized_substs.types.get_mut_slice(subst::TypeSpace)[tp_index] = new_ty;
                mk_struct(cx, did, cx.mk_substs(unsized_substs))
            }
            _ => cx.sess.span_bug(span,
                                  &format!("UnsizeStruct with bad sty: {:?}",
                                          ty_to_string(cx, ty))[])
        },
        &UnsizeVtable(TyTrait { ref principal, ref bounds }, _) => {
            mk_trait(cx, principal.clone(), bounds.clone())
        }
    }
}

pub fn resolve_expr(tcx: &ctxt, expr: &ast::Expr) -> def::Def {
    match tcx.def_map.borrow().get(&expr.id) {
        Some(&def) => def,
        None => {
            tcx.sess.span_bug(expr.span, &format!(
                "no def-map entry for expr {}", expr.id)[]);
        }
    }
}

pub fn expr_is_lval(tcx: &ctxt, e: &ast::Expr) -> bool {
    match expr_kind(tcx, e) {
        LvalueExpr => true,
        RvalueDpsExpr | RvalueDatumExpr | RvalueStmtExpr => false
    }
}

/// We categorize expressions into three kinds.  The distinction between
/// lvalue/rvalue is fundamental to the language.  The distinction between the
/// two kinds of rvalues is an artifact of trans which reflects how we will
/// generate code for that kind of expression.  See trans/expr.rs for more
/// information.
#[derive(Copy)]
pub enum ExprKind {
    LvalueExpr,
    RvalueDpsExpr,
    RvalueDatumExpr,
    RvalueStmtExpr
}

pub fn expr_kind(tcx: &ctxt, expr: &ast::Expr) -> ExprKind {
    if tcx.method_map.borrow().contains_key(&MethodCall::expr(expr.id)) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS, but there are a few exceptions:
        return match expr.node {
            // `a += b` has a unit result.
            ast::ExprAssignOp(..) => RvalueStmtExpr,

            // the deref method invoked for `*a` always yields an `&T`
            ast::ExprUnary(ast::UnDeref, _) => LvalueExpr,

            // the index method invoked for `a[i]` always yields an `&T`
            ast::ExprIndex(..) => LvalueExpr,

            // in the general case, result could be any type, use DPS
            _ => RvalueDpsExpr
        };
    }

    match expr.node {
        ast::ExprPath(_) | ast::ExprQPath(_) => {
            match resolve_expr(tcx, expr) {
                def::DefVariant(tid, vid, _) => {
                    let variant_info = enum_variant_with_id(tcx, tid, vid);
                    if variant_info.args.len() > 0 {
                        // N-ary variant.
                        RvalueDatumExpr
                    } else {
                        // Nullary variant.
                        RvalueDpsExpr
                    }
                }

                def::DefStruct(_) => {
                    match tcx.node_types.borrow().get(&expr.id) {
                        Some(ty) => match ty.sty {
                            ty_bare_fn(..) => RvalueDatumExpr,
                            _ => RvalueDpsExpr
                        },
                        // See ExprCast below for why types might be missing.
                        None => RvalueDatumExpr
                     }
                }

                // Special case: A unit like struct's constructor must be called without () at the
                // end (like `UnitStruct`) which means this is an ExprPath to a DefFn. But in case
                // of unit structs this is should not be interpreted as function pointer but as
                // call to the constructor.
                def::DefFn(_, true) => RvalueDpsExpr,

                // Fn pointers are just scalar values.
                def::DefFn(..) | def::DefStaticMethod(..) | def::DefMethod(..) => RvalueDatumExpr,

                // Note: there is actually a good case to be made that
                // DefArg's, particularly those of immediate type, ought to
                // considered rvalues.
                def::DefStatic(..) |
                def::DefUpvar(..) |
                def::DefLocal(..) => LvalueExpr,

                def::DefConst(..) => RvalueDatumExpr,

                def => {
                    tcx.sess.span_bug(
                        expr.span,
                        &format!("uncategorized def for expr {}: {:?}",
                                expr.id,
                                def)[]);
                }
            }
        }

        ast::ExprUnary(ast::UnDeref, _) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprIndex(..) => {
            LvalueExpr
        }

        ast::ExprCall(..) |
        ast::ExprMethodCall(..) |
        ast::ExprStruct(..) |
        ast::ExprRange(..) |
        ast::ExprTup(..) |
        ast::ExprIf(..) |
        ast::ExprMatch(..) |
        ast::ExprClosure(..) |
        ast::ExprBlock(..) |
        ast::ExprRepeat(..) |
        ast::ExprVec(..) => {
            RvalueDpsExpr
        }

        ast::ExprIfLet(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprIfLet");
        }
        ast::ExprWhileLet(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprWhileLet");
        }

        ast::ExprForLoop(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprForLoop");
        }

        ast::ExprLit(ref lit) if lit_is_str(&**lit) => {
            RvalueDpsExpr
        }

        ast::ExprCast(..) => {
            match tcx.node_types.borrow().get(&expr.id) {
                Some(&ty) => {
                    if type_is_trait(ty) {
                        RvalueDpsExpr
                    } else {
                        RvalueDatumExpr
                    }
                }
                None => {
                    // Technically, it should not happen that the expr is not
                    // present within the table.  However, it DOES happen
                    // during type check, because the final types from the
                    // expressions are not yet recorded in the tcx.  At that
                    // time, though, we are only interested in knowing lvalue
                    // vs rvalue.  It would be better to base this decision on
                    // the AST type in cast node---but (at the time of this
                    // writing) it's not easy to distinguish casts to traits
                    // from other casts based on the AST.  This should be
                    // easier in the future, when casts to traits
                    // would like @Foo, Box<Foo>, or &Foo.
                    RvalueDatumExpr
                }
            }
        }

        ast::ExprBreak(..) |
        ast::ExprAgain(..) |
        ast::ExprRet(..) |
        ast::ExprWhile(..) |
        ast::ExprLoop(..) |
        ast::ExprAssign(..) |
        ast::ExprInlineAsm(..) |
        ast::ExprAssignOp(..) => {
            RvalueStmtExpr
        }

        ast::ExprLit(_) | // Note: LitStr is carved out above
        ast::ExprUnary(..) |
        ast::ExprBox(None, _) |
        ast::ExprAddrOf(..) |
        ast::ExprBinary(..) => {
            RvalueDatumExpr
        }

        ast::ExprBox(Some(ref place), _) => {
            // Special case `Box<T>` for now:
            let definition = match tcx.def_map.borrow().get(&place.id) {
                Some(&def) => def,
                None => panic!("no def for place"),
            };
            let def_id = definition.def_id();
            if tcx.lang_items.exchange_heap() == Some(def_id) {
                RvalueDatumExpr
            } else {
                RvalueDpsExpr
            }
        }

        ast::ExprParen(ref e) => expr_kind(tcx, &**e),

        ast::ExprMac(..) => {
            tcx.sess.span_bug(
                expr.span,
                "macro expression remains after expansion");
        }
    }
}

pub fn stmt_node_id(s: &ast::Stmt) -> ast::NodeId {
    match s.node {
      ast::StmtDecl(_, id) | StmtExpr(_, id) | StmtSemi(_, id) => {
        return id;
      }
      ast::StmtMac(..) => panic!("unexpanded macro in trans")
    }
}

pub fn field_idx_strict(tcx: &ctxt, name: ast::Name, fields: &[field])
                     -> uint {
    let mut i = 0;
    for f in fields { if f.name == name { return i; } i += 1; }
    tcx.sess.bug(&format!(
        "no field named `{}` found in the list of fields `{:?}`",
        token::get_name(name),
        fields.iter()
              .map(|f| token::get_name(f.name).to_string())
              .collect::<Vec<String>>())[]);
}

pub fn impl_or_trait_item_idx(id: ast::Name, trait_items: &[ImplOrTraitItem])
                              -> Option<uint> {
    trait_items.iter().position(|m| m.name() == id)
}

pub fn ty_sort_string<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> String {
    match ty.sty {
        ty_bool | ty_char | ty_int(_) |
        ty_uint(_) | ty_float(_) | ty_str => {
            ::util::ppaux::ty_to_string(cx, ty)
        }
        ty_tup(ref tys) if tys.is_empty() => ::util::ppaux::ty_to_string(cx, ty),

        ty_enum(id, _) => format!("enum `{}`", item_path_str(cx, id)),
        ty_uniq(_) => "box".to_string(),
        ty_vec(_, Some(n)) => format!("array of {} elements", n),
        ty_vec(_, None) => "slice".to_string(),
        ty_ptr(_) => "*-ptr".to_string(),
        ty_rptr(_, _) => "&-ptr".to_string(),
        ty_bare_fn(Some(_), _) => format!("fn item"),
        ty_bare_fn(None, _) => "fn pointer".to_string(),
        ty_trait(ref inner) => {
            format!("trait {}", item_path_str(cx, inner.principal_def_id()))
        }
        ty_struct(id, _) => {
            format!("struct `{}`", item_path_str(cx, id))
        }
        ty_closure(..) => "closure".to_string(),
        ty_tup(_) => "tuple".to_string(),
        ty_infer(TyVar(_)) => "inferred type".to_string(),
        ty_infer(IntVar(_)) => "integral variable".to_string(),
        ty_infer(FloatVar(_)) => "floating-point variable".to_string(),
        ty_infer(FreshTy(_)) => "skolemized type".to_string(),
        ty_infer(FreshIntTy(_)) => "skolemized integral type".to_string(),
        ty_projection(_) => "associated type".to_string(),
        ty_param(ref p) => {
            if p.space == subst::SelfSpace {
                "Self".to_string()
            } else {
                "type parameter".to_string()
            }
        }
        ty_err => "type error".to_string(),
        ty_open(_) => "opened DST".to_string(),
    }
}

impl<'tcx> Repr<'tcx> for ty::type_err<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        ty::type_err_to_str(tcx, self)
    }
}

/// Explains the source of a type err in a short, human readable way. This is meant to be placed
/// in parentheses after some larger message. You should also invoke `note_and_explain_type_err()`
/// afterwards to present additional details, particularly when it comes to lifetime-related
/// errors.
pub fn type_err_to_str<'tcx>(cx: &ctxt<'tcx>, err: &type_err<'tcx>) -> String {
    match *err {
        terr_cyclic_ty => "cyclic type of infinite size".to_string(),
        terr_mismatch => "types differ".to_string(),
        terr_unsafety_mismatch(values) => {
            format!("expected {} fn, found {} fn",
                    values.expected,
                    values.found)
        }
        terr_abi_mismatch(values) => {
            format!("expected {} fn, found {} fn",
                    values.expected,
                    values.found)
        }
        terr_mutability => "values differ in mutability".to_string(),
        terr_box_mutability => {
            "boxed values differ in mutability".to_string()
        }
        terr_vec_mutability => "vectors differ in mutability".to_string(),
        terr_ptr_mutability => "pointers differ in mutability".to_string(),
        terr_ref_mutability => "references differ in mutability".to_string(),
        terr_ty_param_size(values) => {
            format!("expected a type with {} type params, \
                     found one with {} type params",
                    values.expected,
                    values.found)
        }
        terr_fixed_array_size(values) => {
            format!("expected an array with a fixed size of {} elements, \
                     found one with {} elements",
                    values.expected,
                    values.found)
        }
        terr_tuple_size(values) => {
            format!("expected a tuple with {} elements, \
                     found one with {} elements",
                    values.expected,
                    values.found)
        }
        terr_arg_count => {
            "incorrect number of function parameters".to_string()
        }
        terr_regions_does_not_outlive(..) => {
            "lifetime mismatch".to_string()
        }
        terr_regions_not_same(..) => {
            "lifetimes are not the same".to_string()
        }
        terr_regions_no_overlap(..) => {
            "lifetimes do not intersect".to_string()
        }
        terr_regions_insufficiently_polymorphic(br, _) => {
            format!("expected bound lifetime parameter {}, \
                     found concrete lifetime",
                    bound_region_ptr_to_string(cx, br))
        }
        terr_regions_overly_polymorphic(br, _) => {
            format!("expected concrete lifetime, \
                     found bound lifetime parameter {}",
                    bound_region_ptr_to_string(cx, br))
        }
        terr_sorts(values) => {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            let expected_str = ty_sort_string(cx, values.expected);
            let found_str = ty_sort_string(cx, values.found);
            if expected_str == found_str {
                format!("expected {}, found a different {}", expected_str, found_str)
            } else {
                format!("expected {}, found {}", expected_str, found_str)
            }
        }
        terr_traits(values) => {
            format!("expected trait `{}`, found trait `{}`",
                    item_path_str(cx, values.expected),
                    item_path_str(cx, values.found))
        }
        terr_builtin_bounds(values) => {
            if values.expected.is_empty() {
                format!("expected no bounds, found `{}`",
                        values.found.user_string(cx))
            } else if values.found.is_empty() {
                format!("expected bounds `{}`, found no bounds",
                        values.expected.user_string(cx))
            } else {
                format!("expected bounds `{}`, found bounds `{}`",
                        values.expected.user_string(cx),
                        values.found.user_string(cx))
            }
        }
        terr_integer_as_char => {
            "expected an integral type, found `char`".to_string()
        }
        terr_int_mismatch(ref values) => {
            format!("expected `{:?}`, found `{:?}`",
                    values.expected,
                    values.found)
        }
        terr_float_mismatch(ref values) => {
            format!("expected `{:?}`, found `{:?}`",
                    values.expected,
                    values.found)
        }
        terr_variadic_mismatch(ref values) => {
            format!("expected {} fn, found {} function",
                    if values.expected { "variadic" } else { "non-variadic" },
                    if values.found { "variadic" } else { "non-variadic" })
        }
        terr_convergence_mismatch(ref values) => {
            format!("expected {} fn, found {} function",
                    if values.expected { "converging" } else { "diverging" },
                    if values.found { "converging" } else { "diverging" })
        }
        terr_projection_name_mismatched(ref values) => {
            format!("expected {}, found {}",
                    token::get_name(values.expected),
                    token::get_name(values.found))
        }
        terr_projection_bounds_length(ref values) => {
            format!("expected {} associated type bindings, found {}",
                    values.expected,
                    values.found)
        }
    }
}

pub fn note_and_explain_type_err(cx: &ctxt, err: &type_err) {
    match *err {
        terr_regions_does_not_outlive(subregion, superregion) => {
            note_and_explain_region(cx, "", subregion, "...");
            note_and_explain_region(cx, "...does not necessarily outlive ",
                                    superregion, "");
        }
        terr_regions_not_same(region1, region2) => {
            note_and_explain_region(cx, "", region1, "...");
            note_and_explain_region(cx, "...is not the same lifetime as ",
                                    region2, "");
        }
        terr_regions_no_overlap(region1, region2) => {
            note_and_explain_region(cx, "", region1, "...");
            note_and_explain_region(cx, "...does not overlap ",
                                    region2, "");
        }
        terr_regions_insufficiently_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    "concrete lifetime that was found is ",
                                    conc_region, "");
        }
        terr_regions_overly_polymorphic(_, ty::ReInfer(ty::ReVar(_))) => {
            // don't bother to print out the message below for
            // inference variables, it's not very illuminating.
        }
        terr_regions_overly_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    "expected concrete lifetime is ",
                                    conc_region, "");
        }
        _ => {}
    }
}

pub fn provided_source(cx: &ctxt, id: ast::DefId) -> Option<ast::DefId> {
    cx.provided_method_sources.borrow().get(&id).cloned()
}

pub fn provided_trait_methods<'tcx>(cx: &ctxt<'tcx>, id: ast::DefId)
                                    -> Vec<Rc<Method<'tcx>>> {
    if is_local(id) {
        match cx.map.find(id.node) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ItemTrait(_, _, _, ref ms) => {
                        let (_, p) =
                            ast_util::split_trait_methods(&ms[..]);
                        p.iter()
                         .map(|m| {
                            match impl_or_trait_item(
                                    cx,
                                    ast_util::local_def(m.id)) {
                                MethodTraitItem(m) => m,
                                TypeTraitItem(_) => {
                                    cx.sess.bug("provided_trait_methods(): \
                                                 split_trait_methods() put \
                                                 associated types in the \
                                                 provided method bucket?!")
                                }
                            }
                         }).collect()
                    }
                    _ => {
                        cx.sess.bug(&format!("provided_trait_methods: `{:?}` is \
                                             not a trait",
                                            id)[])
                    }
                }
            }
            _ => {
                cx.sess.bug(&format!("provided_trait_methods: `{:?}` is not a \
                                     trait",
                                    id)[])
            }
        }
    } else {
        csearch::get_provided_trait_methods(cx, id)
    }
}

/// Helper for looking things up in the various maps that are populated during
/// typeck::collect (e.g., `cx.impl_or_trait_items`, `cx.tcache`, etc).  All of
/// these share the pattern that if the id is local, it should have been loaded
/// into the map by the `typeck::collect` phase.  If the def-id is external,
/// then we have to go consult the crate loading code (and cache the result for
/// the future).
fn lookup_locally_or_in_crate_store<V, F>(descr: &str,
                                          def_id: ast::DefId,
                                          map: &mut DefIdMap<V>,
                                          load_external: F) -> V where
    V: Clone,
    F: FnOnce() -> V,
{
    match map.get(&def_id).cloned() {
        Some(v) => { return v; }
        None => { }
    }

    if def_id.krate == ast::LOCAL_CRATE {
        panic!("No def'n found for {:?} in tcx.{}", def_id, descr);
    }
    let v = load_external();
    map.insert(def_id, v.clone());
    v
}

pub fn trait_item<'tcx>(cx: &ctxt<'tcx>, trait_did: ast::DefId, idx: uint)
                        -> ImplOrTraitItem<'tcx> {
    let method_def_id = (*ty::trait_item_def_ids(cx, trait_did))[idx].def_id();
    impl_or_trait_item(cx, method_def_id)
}

pub fn trait_items<'tcx>(cx: &ctxt<'tcx>, trait_did: ast::DefId)
                         -> Rc<Vec<ImplOrTraitItem<'tcx>>> {
    let mut trait_items = cx.trait_items_cache.borrow_mut();
    match trait_items.get(&trait_did).cloned() {
        Some(trait_items) => trait_items,
        None => {
            let def_ids = ty::trait_item_def_ids(cx, trait_did);
            let items: Rc<Vec<ImplOrTraitItem>> =
                Rc::new(def_ids.iter()
                               .map(|d| impl_or_trait_item(cx, d.def_id()))
                               .collect());
            trait_items.insert(trait_did, items.clone());
            items
        }
    }
}

pub fn trait_impl_polarity<'tcx>(cx: &ctxt<'tcx>, id: ast::DefId)
                            -> Option<ast::ImplPolarity> {
     if id.krate == ast::LOCAL_CRATE {
         match cx.map.find(id.node) {
             Some(ast_map::NodeItem(item)) => {
                 match item.node {
                     ast::ItemImpl(_, polarity, _, _, _, _) => Some(polarity),
                     _ => None
                 }
             }
             _ => None
         }
     } else {
         csearch::get_impl_polarity(cx, id)
     }
}

pub fn impl_or_trait_item<'tcx>(cx: &ctxt<'tcx>, id: ast::DefId)
                                -> ImplOrTraitItem<'tcx> {
    lookup_locally_or_in_crate_store("impl_or_trait_items",
                                     id,
                                     &mut *cx.impl_or_trait_items
                                             .borrow_mut(),
                                     || {
        csearch::get_impl_or_trait_item(cx, id)
    })
}

/// Returns true if the given ID refers to an associated type and false if it
/// refers to anything else.
pub fn is_associated_type(cx: &ctxt, id: ast::DefId) -> bool {
    memoized(&cx.associated_types, id, |id: ast::DefId| {
        if id.krate == ast::LOCAL_CRATE {
            match cx.impl_or_trait_items.borrow().get(&id) {
                Some(ref item) => {
                    match **item {
                        TypeTraitItem(_) => true,
                        MethodTraitItem(_) => false,
                    }
                }
                None => false,
            }
        } else {
            csearch::is_associated_type(&cx.sess.cstore, id)
        }
    })
}

/// Returns the parameter index that the given associated type corresponds to.
pub fn associated_type_parameter_index(cx: &ctxt,
                                       trait_def: &TraitDef,
                                       associated_type_id: ast::DefId)
                                       -> uint {
    for type_parameter_def in trait_def.generics.types.iter() {
        if type_parameter_def.def_id == associated_type_id {
            return type_parameter_def.index as uint
        }
    }
    cx.sess.bug("couldn't find associated type parameter index")
}

pub fn trait_item_def_ids(cx: &ctxt, id: ast::DefId)
                          -> Rc<Vec<ImplOrTraitItemId>> {
    lookup_locally_or_in_crate_store("trait_item_def_ids",
                                     id,
                                     &mut *cx.trait_item_def_ids.borrow_mut(),
                                     || {
        Rc::new(csearch::get_trait_item_def_ids(&cx.sess.cstore, id))
    })
}

pub fn impl_trait_ref<'tcx>(cx: &ctxt<'tcx>, id: ast::DefId)
                            -> Option<Rc<TraitRef<'tcx>>> {
    memoized(&cx.impl_trait_cache, id, |id: ast::DefId| {
        if id.krate == ast::LOCAL_CRATE {
            debug!("(impl_trait_ref) searching for trait impl {:?}", id);
            match cx.map.find(id.node) {
                Some(ast_map::NodeItem(item)) => {
                    match item.node {
                        ast::ItemImpl(_, _, _, ref opt_trait, _, _) => {
                            match opt_trait {
                                &Some(ref t) => {
                                    let trait_ref = ty::node_id_to_trait_ref(cx, t.ref_id);
                                    Some(trait_ref)
                                }
                                &None => None
                            }
                        }
                        _ => None
                    }
                }
                _ => None
            }
        } else {
            csearch::get_impl_trait(cx, id)
        }
    })
}

pub fn trait_ref_to_def_id(tcx: &ctxt, tr: &ast::TraitRef) -> ast::DefId {
    let def = *tcx.def_map.borrow()
                     .get(&tr.ref_id)
                     .expect("no def-map entry for trait");
    def.def_id()
}

pub fn try_add_builtin_trait(
    tcx: &ctxt,
    trait_def_id: ast::DefId,
    builtin_bounds: &mut EnumSet<BuiltinBound>)
    -> bool
{
    //! Checks whether `trait_ref` refers to one of the builtin
    //! traits, like `Send`, and adds the corresponding
    //! bound to the set `builtin_bounds` if so. Returns true if `trait_ref`
    //! is a builtin trait.

    match tcx.lang_items.to_builtin_kind(trait_def_id) {
        Some(bound) => { builtin_bounds.insert(bound); true }
        None => false
    }
}

pub fn ty_to_def_id(ty: Ty) -> Option<ast::DefId> {
    match ty.sty {
        ty_trait(ref tt) =>
            Some(tt.principal_def_id()),
        ty_struct(id, _) |
        ty_enum(id, _) |
        ty_closure(id, _, _) =>
            Some(id),
        _ =>
            None
    }
}

// Enum information
#[derive(Clone)]
pub struct VariantInfo<'tcx> {
    pub args: Vec<Ty<'tcx>>,
    pub arg_names: Option<Vec<ast::Ident>>,
    pub ctor_ty: Option<Ty<'tcx>>,
    pub name: ast::Name,
    pub id: ast::DefId,
    pub disr_val: Disr,
    pub vis: Visibility
}

impl<'tcx> VariantInfo<'tcx> {

    /// Creates a new VariantInfo from the corresponding ast representation.
    ///
    /// Does not do any caching of the value in the type context.
    pub fn from_ast_variant(cx: &ctxt<'tcx>,
                            ast_variant: &ast::Variant,
                            discriminant: Disr) -> VariantInfo<'tcx> {
        let ctor_ty = node_id_to_type(cx, ast_variant.node.id);

        match ast_variant.node.kind {
            ast::TupleVariantKind(ref args) => {
                let arg_tys = if args.len() > 0 {
                    // the regions in the argument types come from the
                    // enum def'n, and hence will all be early bound
                    ty::no_late_bound_regions(cx, &ty_fn_args(ctor_ty)).unwrap()
                } else {
                    Vec::new()
                };

                return VariantInfo {
                    args: arg_tys,
                    arg_names: None,
                    ctor_ty: Some(ctor_ty),
                    name: ast_variant.node.name.name,
                    id: ast_util::local_def(ast_variant.node.id),
                    disr_val: discriminant,
                    vis: ast_variant.node.vis
                };
            },
            ast::StructVariantKind(ref struct_def) => {
                let fields: &[StructField] = &struct_def.fields[];

                assert!(fields.len() > 0);

                let arg_tys = struct_def.fields.iter()
                    .map(|field| node_id_to_type(cx, field.node.id)).collect();
                let arg_names = fields.iter().map(|field| {
                    match field.node.kind {
                        NamedField(ident, _) => ident,
                        UnnamedField(..) => cx.sess.bug(
                            "enum_variants: all fields in struct must have a name")
                    }
                }).collect();

                return VariantInfo {
                    args: arg_tys,
                    arg_names: Some(arg_names),
                    ctor_ty: None,
                    name: ast_variant.node.name.name,
                    id: ast_util::local_def(ast_variant.node.id),
                    disr_val: discriminant,
                    vis: ast_variant.node.vis
                };
            }
        }
    }
}

pub fn substd_enum_variants<'tcx>(cx: &ctxt<'tcx>,
                                  id: ast::DefId,
                                  substs: &Substs<'tcx>)
                                  -> Vec<Rc<VariantInfo<'tcx>>> {
    enum_variants(cx, id).iter().map(|variant_info| {
        let substd_args = variant_info.args.iter()
            .map(|aty| aty.subst(cx, substs)).collect::<Vec<_>>();

        let substd_ctor_ty = variant_info.ctor_ty.subst(cx, substs);

        Rc::new(VariantInfo {
            args: substd_args,
            ctor_ty: substd_ctor_ty,
            ..(**variant_info).clone()
        })
    }).collect()
}

pub fn item_path_str(cx: &ctxt, id: ast::DefId) -> String {
    with_path(cx, id, |path| ast_map::path_to_string(path)).to_string()
}

#[derive(Copy)]
pub enum DtorKind {
    NoDtor,
    TraitDtor(DefId, bool)
}

impl DtorKind {
    pub fn is_present(&self) -> bool {
        match *self {
            TraitDtor(..) => true,
            _ => false
        }
    }

    pub fn has_drop_flag(&self) -> bool {
        match self {
            &NoDtor => false,
            &TraitDtor(_, flag) => flag
        }
    }
}

/* If struct_id names a struct with a dtor, return Some(the dtor's id).
   Otherwise return none. */
pub fn ty_dtor(cx: &ctxt, struct_id: DefId) -> DtorKind {
    match cx.destructor_for_type.borrow().get(&struct_id) {
        Some(&method_def_id) => {
            let flag = !has_attr(cx, struct_id, "unsafe_no_drop_flag");

            TraitDtor(method_def_id, flag)
        }
        None => NoDtor,
    }
}

pub fn has_dtor(cx: &ctxt, struct_id: DefId) -> bool {
    cx.destructor_for_type.borrow().contains_key(&struct_id)
}

pub fn with_path<T, F>(cx: &ctxt, id: ast::DefId, f: F) -> T where
    F: FnOnce(ast_map::PathElems) -> T,
{
    if id.krate == ast::LOCAL_CRATE {
        cx.map.with_path(id.node, f)
    } else {
        f(csearch::get_item_path(cx, id).iter().cloned().chain(None))
    }
}

pub fn enum_is_univariant(cx: &ctxt, id: ast::DefId) -> bool {
    enum_variants(cx, id).len() == 1
}

pub fn type_is_empty(cx: &ctxt, ty: Ty) -> bool {
    match ty.sty {
       ty_enum(did, _) => (*enum_variants(cx, did)).is_empty(),
       _ => false
     }
}

pub fn enum_variants<'tcx>(cx: &ctxt<'tcx>, id: ast::DefId)
                           -> Rc<Vec<Rc<VariantInfo<'tcx>>>> {
    memoized(&cx.enum_var_cache, id, |id: ast::DefId| {
        if ast::LOCAL_CRATE != id.krate {
            Rc::new(csearch::get_enum_variants(cx, id))
        } else {
            /*
              Although both this code and check_enum_variants in typeck/check
              call eval_const_expr, it should never get called twice for the same
              expr, since check_enum_variants also updates the enum_var_cache
             */
            match cx.map.get(id.node) {
                ast_map::NodeItem(ref item) => {
                    match item.node {
                        ast::ItemEnum(ref enum_definition, _) => {
                            let mut last_discriminant: Option<Disr> = None;
                            Rc::new(enum_definition.variants.iter().map(|variant| {

                                let mut discriminant = match last_discriminant {
                                    Some(val) => val + 1,
                                    None => INITIAL_DISCRIMINANT_VALUE
                                };

                                if let Some(ref e) = variant.node.disr_expr {
                                    // Preserve all values, and prefer signed.
                                    let ty = Some(cx.types.i64);
                                    match const_eval::eval_const_expr_partial(cx, &**e, ty) {
                                        Ok(const_eval::const_int(val)) => {
                                            discriminant = val as Disr;
                                        }
                                        Ok(const_eval::const_uint(val)) => {
                                            discriminant = val as Disr;
                                        }
                                        Ok(_) => {
                                            span_err!(cx.sess, e.span, E0304,
                                                      "expected signed integer constant");
                                        }
                                        Err(err) => {
                                            span_err!(cx.sess, e.span, E0305,
                                                      "expected constant: {}", err);
                                        }
                                    }
                                };

                                last_discriminant = Some(discriminant);
                                Rc::new(VariantInfo::from_ast_variant(cx, &**variant,
                                                                      discriminant))
                            }).collect())
                        }
                        _ => {
                            cx.sess.bug("enum_variants: id not bound to an enum")
                        }
                    }
                }
                _ => cx.sess.bug("enum_variants: id not bound to an enum")
            }
        }
    })
}

// Returns information about the enum variant with the given ID:
pub fn enum_variant_with_id<'tcx>(cx: &ctxt<'tcx>,
                                  enum_id: ast::DefId,
                                  variant_id: ast::DefId)
                                  -> Rc<VariantInfo<'tcx>> {
    enum_variants(cx, enum_id).iter()
                              .find(|variant| variant.id == variant_id)
                              .expect("enum_variant_with_id(): no variant exists with that ID")
                              .clone()
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
pub fn lookup_item_type<'tcx>(cx: &ctxt<'tcx>,
                              did: ast::DefId)
                              -> TypeScheme<'tcx> {
    lookup_locally_or_in_crate_store(
        "tcache", did, &mut *cx.tcache.borrow_mut(),
        || csearch::get_type(cx, did))
}

/// Given the did of a trait, returns its canonical trait ref.
pub fn lookup_trait_def<'tcx>(cx: &ctxt<'tcx>, did: ast::DefId)
                              -> Rc<TraitDef<'tcx>> {
    memoized(&cx.trait_defs, did, |did: DefId| {
        assert!(did.krate != ast::LOCAL_CRATE);
        Rc::new(csearch::get_trait_def(cx, did))
    })
}

/// Given the did of a trait, returns its full set of predicates.
pub fn lookup_predicates<'tcx>(cx: &ctxt<'tcx>, did: ast::DefId)
                                -> GenericPredicates<'tcx>
{
    memoized(&cx.predicates, did, |did: DefId| {
        assert!(did.krate != ast::LOCAL_CRATE);
        csearch::get_predicates(cx, did)
    })
}

/// Given a reference to a trait, returns the "superbounds" declared
/// on the trait, with appropriate substitutions applied. Basically,
/// this applies a filter to the where clauses on the trait, returning
/// those that have the form:
///
///     Self : SuperTrait<...>
///     Self : 'region
pub fn predicates_for_trait_ref<'tcx>(tcx: &ctxt<'tcx>,
                                      trait_ref: &PolyTraitRef<'tcx>)
                                      -> Vec<ty::Predicate<'tcx>>
{
    let trait_def = lookup_trait_def(tcx, trait_ref.def_id());

    debug!("bounds_for_trait_ref(trait_def={:?}, trait_ref={:?})",
           trait_def.repr(tcx), trait_ref.repr(tcx));

    // The interaction between HRTB and supertraits is not entirely
    // obvious. Let me walk you (and myself) through an example.
    //
    // Let's start with an easy case. Consider two traits:
    //
    //     trait Foo<'a> : Bar<'a,'a> { }
    //     trait Bar<'b,'c> { }
    //
    // Now, if we have a trait reference `for<'x> T : Foo<'x>`, then
    // we can deduce that `for<'x> T : Bar<'x,'x>`. Basically, if we
    // knew that `Foo<'x>` (for any 'x) then we also know that
    // `Bar<'x,'x>` (for any 'x). This more-or-less falls out from
    // normal substitution.
    //
    // In terms of why this is sound, the idea is that whenever there
    // is an impl of `T:Foo<'a>`, it must show that `T:Bar<'a,'a>`
    // holds.  So if there is an impl of `T:Foo<'a>` that applies to
    // all `'a`, then we must know that `T:Bar<'a,'a>` holds for all
    // `'a`.
    //
    // Another example to be careful of is this:
    //
    //     trait Foo1<'a> : for<'b> Bar1<'a,'b> { }
    //     trait Bar1<'b,'c> { }
    //
    // Here, if we have `for<'x> T : Foo1<'x>`, then what do we know?
    // The answer is that we know `for<'x,'b> T : Bar1<'x,'b>`. The
    // reason is similar to the previous example: any impl of
    // `T:Foo1<'x>` must show that `for<'b> T : Bar1<'x, 'b>`.  So
    // basically we would want to collapse the bound lifetimes from
    // the input (`trait_ref`) and the supertraits.
    //
    // To achieve this in practice is fairly straightforward. Let's
    // consider the more complicated scenario:
    //
    // - We start out with `for<'x> T : Foo1<'x>`. In this case, `'x`
    //   has a De Bruijn index of 1. We want to produce `for<'x,'b> T : Bar1<'x,'b>`,
    //   where both `'x` and `'b` would have a DB index of 1.
    //   The substitution from the input trait-ref is therefore going to be
    //   `'a => 'x` (where `'x` has a DB index of 1).
    // - The super-trait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
    //   early-bound parameter and `'b' is a late-bound parameter with a
    //   DB index of 1.
    // - If we replace `'a` with `'x` from the input, it too will have
    //   a DB index of 1, and thus we'll have `for<'x,'b> Bar1<'x,'b>`
    //   just as we wanted.
    //
    // There is only one catch. If we just apply the substitution `'a
    // => 'x` to `for<'b> Bar1<'a,'b>`, the substitution code will
    // adjust the DB index because we substituting into a binder (it
    // tries to be so smart...) resulting in `for<'x> for<'b>
    // Bar1<'x,'b>` (we have no syntax for this, so use your
    // imagination). Basically the 'x will have DB index of 2 and 'b
    // will have DB index of 1. Not quite what we want. So we apply
    // the substitution to the *contents* of the trait reference,
    // rather than the trait reference itself (put another way, the
    // substitution code expects equal binding levels in the values
    // from the substitution and the value being substituted into, and
    // this trick achieves that).

    // Carefully avoid the binder introduced by each trait-ref by
    // substituting over the substs, not the trait-refs themselves,
    // thus achieving the "collapse" described in the big comment
    // above.
    let trait_bounds: Vec<_> =
        trait_def.bounds.trait_bounds
        .iter()
        .map(|poly_trait_ref| ty::Binder(poly_trait_ref.0.subst(tcx, trait_ref.substs())))
        .collect();

    let projection_bounds: Vec<_> =
        trait_def.bounds.projection_bounds
        .iter()
        .map(|poly_proj| ty::Binder(poly_proj.0.subst(tcx, trait_ref.substs())))
        .collect();

    debug!("bounds_for_trait_ref: trait_bounds={} projection_bounds={}",
           trait_bounds.repr(tcx),
           projection_bounds.repr(tcx));

    // The region bounds and builtin bounds do not currently introduce
    // binders so we can just substitute in a straightforward way here.
    let region_bounds =
        trait_def.bounds.region_bounds.subst(tcx, trait_ref.substs());
    let builtin_bounds =
        trait_def.bounds.builtin_bounds.subst(tcx, trait_ref.substs());

    let bounds = ty::ParamBounds {
        trait_bounds: trait_bounds,
        region_bounds: region_bounds,
        builtin_bounds: builtin_bounds,
        projection_bounds: projection_bounds,
    };

    predicates(tcx, trait_ref.self_ty(), &bounds)
}

pub fn predicates<'tcx>(
    tcx: &ctxt<'tcx>,
    param_ty: Ty<'tcx>,
    bounds: &ParamBounds<'tcx>)
    -> Vec<Predicate<'tcx>>
{
    let mut vec = Vec::new();

    for builtin_bound in &bounds.builtin_bounds {
        match traits::trait_ref_for_builtin_bound(tcx, builtin_bound, param_ty) {
            Ok(trait_ref) => { vec.push(trait_ref.as_predicate()); }
            Err(ErrorReported) => { }
        }
    }

    for &region_bound in &bounds.region_bounds {
        // account for the binder being introduced below; no need to shift `param_ty`
        // because, at present at least, it can only refer to early-bound regions
        let region_bound = ty_fold::shift_region(region_bound, 1);
        vec.push(ty::Binder(ty::OutlivesPredicate(param_ty, region_bound)).as_predicate());
    }

    for bound_trait_ref in &bounds.trait_bounds {
        vec.push(bound_trait_ref.as_predicate());
    }

    for projection in &bounds.projection_bounds {
        vec.push(projection.as_predicate());
    }

    vec
}

/// Get the attributes of a definition.
pub fn get_attrs<'tcx>(tcx: &'tcx ctxt, did: DefId)
                       -> CowVec<'tcx, ast::Attribute> {
    if is_local(did) {
        let item = tcx.map.expect_item(did.node);
        Cow::Borrowed(&item.attrs[])
    } else {
        Cow::Owned(csearch::get_item_attrs(&tcx.sess.cstore, did))
    }
}

/// Determine whether an item is annotated with an attribute
pub fn has_attr(tcx: &ctxt, did: DefId, attr: &str) -> bool {
    get_attrs(tcx, did).iter().any(|item| item.check_name(attr))
}

/// Determine whether an item is annotated with `#[repr(packed)]`
pub fn lookup_packed(tcx: &ctxt, did: DefId) -> bool {
    lookup_repr_hints(tcx, did).contains(&attr::ReprPacked)
}

/// Determine whether an item is annotated with `#[simd]`
pub fn lookup_simd(tcx: &ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "simd")
}

/// Obtain the representation annotation for a struct definition.
pub fn lookup_repr_hints(tcx: &ctxt, did: DefId) -> Rc<Vec<attr::ReprAttr>> {
    memoized(&tcx.repr_hint_cache, did, |did: DefId| {
        Rc::new(if did.krate == LOCAL_CRATE {
            get_attrs(tcx, did).iter().flat_map(|meta| {
                attr::find_repr_attrs(tcx.sess.diagnostic(), meta).into_iter()
            }).collect()
        } else {
            csearch::get_repr_attrs(&tcx.sess.cstore, did)
        })
    })
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the struct is generic
pub fn lookup_field_type<'tcx>(tcx: &ctxt<'tcx>,
                               struct_id: DefId,
                               id: DefId,
                               substs: &Substs<'tcx>)
                               -> Ty<'tcx> {
    let ty = if id.krate == ast::LOCAL_CRATE {
        node_id_to_type(tcx, id.node)
    } else {
        let mut tcache = tcx.tcache.borrow_mut();
        let pty = tcache.entry(id).get().unwrap_or_else(
            |vacant_entry| vacant_entry.insert(csearch::get_field_type(tcx, struct_id, id)));
        pty.ty
    };
    ty.subst(tcx, substs)
}

// Look up the list of field names and IDs for a given struct.
// Panics if the id is not bound to a struct.
pub fn lookup_struct_fields(cx: &ctxt, did: ast::DefId) -> Vec<field_ty> {
    if did.krate == ast::LOCAL_CRATE {
        let struct_fields = cx.struct_fields.borrow();
        match struct_fields.get(&did) {
            Some(fields) => (**fields).clone(),
            _ => {
                cx.sess.bug(
                    &format!("ID not mapped to struct fields: {}",
                            cx.map.node_to_string(did.node))[]);
            }
        }
    } else {
        csearch::get_struct_fields(&cx.sess.cstore, did)
    }
}

pub fn is_tuple_struct(cx: &ctxt, did: ast::DefId) -> bool {
    let fields = lookup_struct_fields(cx, did);
    !fields.is_empty() && fields.iter().all(|f| f.name == token::special_names::unnamed_field)
}

// Returns a list of fields corresponding to the struct's items. trans uses
// this. Takes a list of substs with which to instantiate field types.
pub fn struct_fields<'tcx>(cx: &ctxt<'tcx>, did: ast::DefId, substs: &Substs<'tcx>)
                           -> Vec<field<'tcx>> {
    lookup_struct_fields(cx, did).iter().map(|f| {
       field {
            name: f.name,
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: MutImmutable
            }
        }
    }).collect()
}

// Returns a list of fields corresponding to the tuple's items. trans uses
// this.
pub fn tup_fields<'tcx>(v: &[Ty<'tcx>]) -> Vec<field<'tcx>> {
    v.iter().enumerate().map(|(i, &f)| {
       field {
            name: token::intern(&i.to_string()[]),
            mt: mt {
                ty: f,
                mutbl: MutImmutable
            }
        }
    }).collect()
}

#[derive(Copy, Clone)]
pub struct ClosureUpvar<'tcx> {
    pub def: def::Def,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

// Returns a list of `ClosureUpvar`s for each upvar.
pub fn closure_upvars<'tcx>(typer: &mc::Typer<'tcx>,
                            closure_id: ast::DefId,
                            substs: &Substs<'tcx>)
                            -> Option<Vec<ClosureUpvar<'tcx>>>
{
    // Presently an unboxed closure type cannot "escape" out of a
    // function, so we will only encounter ones that originated in the
    // local crate or were inlined into it along with some function.
    // This may change if abstract return types of some sort are
    // implemented.
    assert!(closure_id.krate == ast::LOCAL_CRATE);
    let tcx = typer.tcx();
    match tcx.freevars.borrow().get(&closure_id.node) {
        None => Some(vec![]),
        Some(ref freevars) => {
            freevars.iter()
                    .map(|freevar| {
                        let freevar_def_id = freevar.def.def_id();
                        let freevar_ty = match typer.node_ty(freevar_def_id.node) {
                            Ok(t) => { t }
                            Err(()) => { return None; }
                        };
                        let freevar_ty = freevar_ty.subst(tcx, substs);

                        let upvar_id = ty::UpvarId {
                            var_id: freevar_def_id.node,
                            closure_expr_id: closure_id.node
                        };

                        typer.upvar_capture(upvar_id).map(|capture| {
                            let freevar_ref_ty = match capture {
                                UpvarCapture::ByValue => {
                                    freevar_ty
                                }
                                UpvarCapture::ByRef(borrow) => {
                                    mk_rptr(tcx,
                                            tcx.mk_region(borrow.region),
                                            ty::mt {
                                                ty: freevar_ty,
                                                mutbl: borrow.kind.to_mutbl_lossy(),
                                            })
                                }
                            };

                            ClosureUpvar {
                                def: freevar.def,
                                span: freevar.span,
                                ty: freevar_ref_ty,
                            }
                        })
                    })
                    .collect()
        }
    }
}

pub fn is_binopable<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>, op: ast::BinOp) -> bool {
    #![allow(non_upper_case_globals)]
    static tycat_other: int = 0;
    static tycat_bool: int = 1;
    static tycat_char: int = 2;
    static tycat_int: int = 3;
    static tycat_float: int = 4;
    static tycat_raw_ptr: int = 6;

    static opcat_add: int = 0;
    static opcat_sub: int = 1;
    static opcat_mult: int = 2;
    static opcat_shift: int = 3;
    static opcat_rel: int = 4;
    static opcat_eq: int = 5;
    static opcat_bit: int = 6;
    static opcat_logic: int = 7;
    static opcat_mod: int = 8;

    fn opcat(op: ast::BinOp) -> int {
        match op.node {
          ast::BiAdd => opcat_add,
          ast::BiSub => opcat_sub,
          ast::BiMul => opcat_mult,
          ast::BiDiv => opcat_mult,
          ast::BiRem => opcat_mod,
          ast::BiAnd => opcat_logic,
          ast::BiOr => opcat_logic,
          ast::BiBitXor => opcat_bit,
          ast::BiBitAnd => opcat_bit,
          ast::BiBitOr => opcat_bit,
          ast::BiShl => opcat_shift,
          ast::BiShr => opcat_shift,
          ast::BiEq => opcat_eq,
          ast::BiNe => opcat_eq,
          ast::BiLt => opcat_rel,
          ast::BiLe => opcat_rel,
          ast::BiGe => opcat_rel,
          ast::BiGt => opcat_rel
        }
    }

    fn tycat<'tcx>(cx: &ctxt<'tcx>, ty: Ty<'tcx>) -> int {
        if type_is_simd(cx, ty) {
            return tycat(cx, simd_type(cx, ty))
        }
        match ty.sty {
          ty_char => tycat_char,
          ty_bool => tycat_bool,
          ty_int(_) | ty_uint(_) | ty_infer(IntVar(_)) => tycat_int,
          ty_float(_) | ty_infer(FloatVar(_)) => tycat_float,
          ty_ptr(_) => tycat_raw_ptr,
          _ => tycat_other
        }
    }

    static t: bool = true;
    static f: bool = false;

    let tbl = [
    //           +, -, *, shift, rel, ==, bit, logic, mod
    /*other*/   [f, f, f, f,     f,   f,  f,   f,     f],
    /*bool*/    [f, f, f, f,     t,   t,  t,   t,     f],
    /*char*/    [f, f, f, f,     t,   t,  f,   f,     f],
    /*int*/     [t, t, t, t,     t,   t,  t,   f,     t],
    /*float*/   [t, t, t, f,     t,   t,  f,   f,     f],
    /*bot*/     [t, t, t, t,     t,   t,  t,   t,     t],
    /*raw ptr*/ [f, f, f, f,     t,   t,  f,   f,     f]];

    return tbl[tycat(cx, ty) as uint ][opcat(op) as uint];
}

// Returns the repeat count for a repeating vector expression.
pub fn eval_repeat_count(tcx: &ctxt, count_expr: &ast::Expr) -> uint {
    match const_eval::eval_const_expr_partial(tcx, count_expr, Some(tcx.types.uint)) {
        Ok(val) => {
            let found = match val {
                const_eval::const_uint(count) => return count as uint,
                const_eval::const_int(count) if count >= 0 => return count as uint,
                const_eval::const_int(_) =>
                    "negative integer",
                const_eval::const_float(_) =>
                    "float",
                const_eval::const_str(_) =>
                    "string",
                const_eval::const_bool(_) =>
                    "boolean",
                const_eval::const_binary(_) =>
                    "binary array"
            };
            span_err!(tcx.sess, count_expr.span, E0306,
                "expected positive integer for repeat count, found {}",
                found);
        }
        Err(_) => {
            let found = match count_expr.node {
                ast::ExprPath(ast::Path {
                    global: false,
                    ref segments,
                    ..
                }) if segments.len() == 1 =>
                    "variable",
                _ =>
                    "non-constant expression"
            };
            span_err!(tcx.sess, count_expr.span, E0307,
                "expected constant integer for repeat count, found {}",
                found);
        }
    }
    0
}

// Iterate over a type parameter's bounded traits and any supertraits
// of those traits, ignoring kinds.
// Here, the supertraits are the transitive closure of the supertrait
// relation on the supertraits from each bounded trait's constraint
// list.
pub fn each_bound_trait_and_supertraits<'tcx, F>(tcx: &ctxt<'tcx>,
                                                 bounds: &[PolyTraitRef<'tcx>],
                                                 mut f: F)
                                                 -> bool where
    F: FnMut(PolyTraitRef<'tcx>) -> bool,
{
    for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
        if !f(bound_trait_ref) {
            return false;
        }
    }
    return true;
}

/// Given a set of predicates that apply to an object type, returns
/// the region bounds that the (erased) `Self` type must
/// outlive. Precisely *because* the `Self` type is erased, the
/// parameter `erased_self_ty` must be supplied to indicate what type
/// has been used to represent `Self` in the predicates
/// themselves. This should really be a unique type; `FreshTy(0)` is a
/// popular choice.
///
/// Requires that trait definitions have been processed so that we can
/// elaborate predicates and walk supertraits.
pub fn required_region_bounds<'tcx>(tcx: &ctxt<'tcx>,
                                    erased_self_ty: Ty<'tcx>,
                                    predicates: Vec<ty::Predicate<'tcx>>)
                                    -> Vec<ty::Region>
{
    debug!("required_region_bounds(erased_self_ty={:?}, predicates={:?})",
           erased_self_ty.repr(tcx),
           predicates.repr(tcx));

    assert!(!erased_self_ty.has_escaping_regions());

    traits::elaborate_predicates(tcx, predicates)
        .filter_map(|predicate| {
            match predicate {
                ty::Predicate::Projection(..) |
                ty::Predicate::Trait(..) |
                ty::Predicate::Equate(..) |
                ty::Predicate::RegionOutlives(..) => {
                    None
                }
                ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(t, r))) => {
                    // Search for a bound of the form `erased_self_ty
                    // : 'a`, but be wary of something like `for<'a>
                    // erased_self_ty : 'a` (we interpret a
                    // higher-ranked bound like that as 'static,
                    // though at present the code in `fulfill.rs`
                    // considers such bounds to be unsatisfiable, so
                    // it's kind of a moot point since you could never
                    // construct such an object, but this seems
                    // correct even if that code changes).
                    if t == erased_self_ty && !r.has_escaping_regions() {
                        if r.has_escaping_regions() {
                            Some(ty::ReStatic)
                        } else {
                            Some(r)
                        }
                    } else {
                        None
                    }
                }
            }
        })
        .collect()
}

pub fn get_tydesc_ty<'tcx>(tcx: &ctxt<'tcx>) -> Result<Ty<'tcx>, String> {
    tcx.lang_items.require(TyDescStructLangItem).map(|tydesc_lang_item| {
        tcx.intrinsic_defs.borrow().get(&tydesc_lang_item).cloned()
            .expect("Failed to resolve TyDesc")
    })
}

pub fn item_variances(tcx: &ctxt, item_id: ast::DefId) -> Rc<ItemVariances> {
    lookup_locally_or_in_crate_store(
        "item_variance_map", item_id, &mut *tcx.item_variance_map.borrow_mut(),
        || Rc::new(csearch::get_item_variances(&tcx.sess.cstore, item_id)))
}

/// Records a trait-to-implementation mapping.
pub fn record_trait_implementation(tcx: &ctxt,
                                   trait_def_id: DefId,
                                   impl_def_id: DefId) {

    match tcx.trait_impls.borrow().get(&trait_def_id) {
        Some(impls_for_trait) => {
            impls_for_trait.borrow_mut().push(impl_def_id);
            return;
        }
        None => {}
    }

    tcx.trait_impls.borrow_mut().insert(trait_def_id, Rc::new(RefCell::new(vec!(impl_def_id))));
}

/// Populates the type context with all the implementations for the given type
/// if necessary.
pub fn populate_implementations_for_type_if_necessary(tcx: &ctxt,
                                                      type_id: ast::DefId) {
    if type_id.krate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_types.borrow().contains(&type_id) {
        return
    }

    debug!("populate_implementations_for_type_if_necessary: searching for {:?}", type_id);

    let mut inherent_impls = Vec::new();
    csearch::each_implementation_for_type(&tcx.sess.cstore, type_id,
            |impl_def_id| {
        let impl_items = csearch::get_impl_items(&tcx.sess.cstore, impl_def_id);

        // Record the trait->implementation mappings, if applicable.
        let associated_traits = csearch::get_impl_trait(tcx, impl_def_id);
        if let Some(ref trait_ref) = associated_traits {
            record_trait_implementation(tcx, trait_ref.def_id, impl_def_id);
        }

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for impl_item_def_id in &impl_items {
            let method_def_id = impl_item_def_id.def_id();
            match impl_or_trait_item(tcx, method_def_id) {
                MethodTraitItem(method) => {
                    if let Some(source) = method.provided_source {
                        tcx.provided_method_sources
                           .borrow_mut()
                           .insert(method_def_id, source);
                    }
                }
                TypeTraitItem(_) => {}
            }
        }

        // Store the implementation info.
        tcx.impl_items.borrow_mut().insert(impl_def_id, impl_items);

        // If this is an inherent implementation, record it.
        if associated_traits.is_none() {
            inherent_impls.push(impl_def_id);
        }
    });

    tcx.inherent_impls.borrow_mut().insert(type_id, Rc::new(inherent_impls));
    tcx.populated_external_types.borrow_mut().insert(type_id);
}

/// Populates the type context with all the implementations for the given
/// trait if necessary.
pub fn populate_implementations_for_trait_if_necessary(
        tcx: &ctxt,
        trait_id: ast::DefId) {
    if trait_id.krate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_traits.borrow().contains(&trait_id) {
        return
    }

    csearch::each_implementation_for_trait(&tcx.sess.cstore, trait_id,
            |implementation_def_id| {
        let impl_items = csearch::get_impl_items(&tcx.sess.cstore, implementation_def_id);

        // Record the trait->implementation mapping.
        record_trait_implementation(tcx, trait_id, implementation_def_id);

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for impl_item_def_id in &impl_items {
            let method_def_id = impl_item_def_id.def_id();
            match impl_or_trait_item(tcx, method_def_id) {
                MethodTraitItem(method) => {
                    if let Some(source) = method.provided_source {
                        tcx.provided_method_sources
                           .borrow_mut()
                           .insert(method_def_id, source);
                    }
                }
                TypeTraitItem(_) => {}
            }
        }

        // Store the implementation info.
        tcx.impl_items.borrow_mut().insert(implementation_def_id, impl_items);
    });

    tcx.populated_external_traits.borrow_mut().insert(trait_id);
}

/// Given the def_id of an impl, return the def_id of the trait it implements.
/// If it implements no trait, return `None`.
pub fn trait_id_of_impl(tcx: &ctxt,
                        def_id: ast::DefId)
                        -> Option<ast::DefId> {
    ty::impl_trait_ref(tcx, def_id).map(|tr| tr.def_id)
}

/// If the given def ID describes a method belonging to an impl, return the
/// ID of the impl that the method belongs to. Otherwise, return `None`.
pub fn impl_of_method(tcx: &ctxt, def_id: ast::DefId)
                       -> Option<ast::DefId> {
    if def_id.krate != LOCAL_CRATE {
        return match csearch::get_impl_or_trait_item(tcx,
                                                     def_id).container() {
            TraitContainer(_) => None,
            ImplContainer(def_id) => Some(def_id),
        };
    }
    match tcx.impl_or_trait_items.borrow().get(&def_id).cloned() {
        Some(trait_item) => {
            match trait_item.container() {
                TraitContainer(_) => None,
                ImplContainer(def_id) => Some(def_id),
            }
        }
        None => None
    }
}

/// If the given def ID describes an item belonging to a trait (either a
/// default method or an implementation of a trait method), return the ID of
/// the trait that the method belongs to. Otherwise, return `None`.
pub fn trait_of_item(tcx: &ctxt, def_id: ast::DefId) -> Option<ast::DefId> {
    if def_id.krate != LOCAL_CRATE {
        return csearch::get_trait_of_item(&tcx.sess.cstore, def_id, tcx);
    }
    match tcx.impl_or_trait_items.borrow().get(&def_id).cloned() {
        Some(impl_or_trait_item) => {
            match impl_or_trait_item.container() {
                TraitContainer(def_id) => Some(def_id),
                ImplContainer(def_id) => trait_id_of_impl(tcx, def_id),
            }
        }
        None => None
    }
}

/// If the given def ID describes an item belonging to a trait, (either a
/// default method or an implementation of a trait method), return the ID of
/// the method inside trait definition (this means that if the given def ID
/// is already that of the original trait method, then the return value is
/// the same).
/// Otherwise, return `None`.
pub fn trait_item_of_item(tcx: &ctxt, def_id: ast::DefId)
                          -> Option<ImplOrTraitItemId> {
    let impl_item = match tcx.impl_or_trait_items.borrow().get(&def_id) {
        Some(m) => m.clone(),
        None => return None,
    };
    let name = impl_item.name();
    match trait_of_item(tcx, def_id) {
        Some(trait_did) => {
            let trait_items = ty::trait_items(tcx, trait_did);
            trait_items.iter()
                .position(|m| m.name() == name)
                .map(|idx| ty::trait_item(tcx, trait_did, idx).id())
        }
        None => None
    }
}

/// Creates a hash of the type `Ty` which will be the same no matter what crate
/// context it's calculated within. This is used by the `type_id` intrinsic.
pub fn hash_crate_independent<'tcx>(tcx: &ctxt<'tcx>, ty: Ty<'tcx>, svh: &Svh) -> u64 {
    let mut state = SipHasher::new();
    helper(tcx, ty, svh, &mut state);
    return state.finish();

    fn helper<'tcx>(tcx: &ctxt<'tcx>, ty: Ty<'tcx>, svh: &Svh,
                    state: &mut SipHasher) {
        macro_rules! byte { ($b:expr) => { ($b as u8).hash(state) } }
        macro_rules! hash { ($e:expr) => { $e.hash(state) }  }

        let region = |state: &mut SipHasher, r: Region| {
            match r {
                ReStatic => {}
                ReLateBound(db, BrAnon(i)) => {
                    db.hash(state);
                    i.hash(state);
                }
                ReEmpty |
                ReEarlyBound(..) |
                ReLateBound(..) |
                ReFree(..) |
                ReScope(..) |
                ReInfer(..) => {
                    tcx.sess.bug("unexpected region found when hashing a type")
                }
            }
        };
        let did = |state: &mut SipHasher, did: DefId| {
            let h = if ast_util::is_local(did) {
                svh.clone()
            } else {
                tcx.sess.cstore.get_crate_hash(did.krate)
            };
            h.as_str().hash(state);
            did.node.hash(state);
        };
        let mt = |state: &mut SipHasher, mt: mt| {
            mt.mutbl.hash(state);
        };
        let fn_sig = |state: &mut SipHasher, sig: &Binder<FnSig<'tcx>>| {
            let sig = anonymize_late_bound_regions(tcx, sig).0;
            for a in &sig.inputs { helper(tcx, *a, svh, state); }
            if let ty::FnConverging(output) = sig.output {
                helper(tcx, output, svh, state);
            }
        };
        maybe_walk_ty(ty, |ty| {
            match ty.sty {
                ty_bool => byte!(2),
                ty_char => byte!(3),
                ty_int(i) => {
                    byte!(4);
                    hash!(i);
                }
                ty_uint(u) => {
                    byte!(5);
                    hash!(u);
                }
                ty_float(f) => {
                    byte!(6);
                    hash!(f);
                }
                ty_str => {
                    byte!(7);
                }
                ty_enum(d, _) => {
                    byte!(8);
                    did(state, d);
                }
                ty_uniq(_) => {
                    byte!(9);
                }
                ty_vec(_, Some(n)) => {
                    byte!(10);
                    n.hash(state);
                }
                ty_vec(_, None) => {
                    byte!(11);
                }
                ty_ptr(m) => {
                    byte!(12);
                    mt(state, m);
                }
                ty_rptr(r, m) => {
                    byte!(13);
                    region(state, *r);
                    mt(state, m);
                }
                ty_bare_fn(opt_def_id, ref b) => {
                    byte!(14);
                    hash!(opt_def_id);
                    hash!(b.unsafety);
                    hash!(b.abi);
                    fn_sig(state, &b.sig);
                    return false;
                }
                ty_trait(ref data) => {
                    byte!(17);
                    did(state, data.principal_def_id());
                    hash!(data.bounds);

                    let principal = anonymize_late_bound_regions(tcx, &data.principal).0;
                    for subty in principal.substs.types.iter() {
                        helper(tcx, *subty, svh, state);
                    }

                    return false;
                }
                ty_struct(d, _) => {
                    byte!(18);
                    did(state, d);
                }
                ty_tup(ref inner) => {
                    byte!(19);
                    hash!(inner.len());
                }
                ty_param(p) => {
                    byte!(20);
                    hash!(p.space);
                    hash!(p.idx);
                    hash!(token::get_name(p.name));
                }
                ty_open(_) => byte!(22),
                ty_infer(_) => unreachable!(),
                ty_err => byte!(23),
                ty_closure(d, r, _) => {
                    byte!(24);
                    did(state, d);
                    region(state, *r);
                }
                ty_projection(ref data) => {
                    byte!(25);
                    did(state, data.trait_ref.def_id);
                    hash!(token::get_name(data.item_name));
                }
            }
            true
        });
    }
}

impl Variance {
    pub fn to_string(self) -> &'static str {
        match self {
            Covariant => "+",
            Contravariant => "-",
            Invariant => "o",
            Bivariant => "*",
        }
    }
}

/// Construct a parameter environment suitable for static contexts or other contexts where there
/// are no free type/lifetime parameters in scope.
pub fn empty_parameter_environment<'a,'tcx>(cx: &'a ctxt<'tcx>) -> ParameterEnvironment<'a,'tcx> {
    ty::ParameterEnvironment { tcx: cx,
                               free_substs: Substs::empty(),
                               caller_bounds: Vec::new(),
                               implicit_region_bound: ty::ReEmpty,
                               selection_cache: traits::SelectionCache::new(), }
}

/// Constructs and returns a substitution that can be applied to move from
/// the "outer" view of a type or method to the "inner" view.
/// In general, this means converting from bound parameters to
/// free parameters. Since we currently represent bound/free type
/// parameters in the same way, this only has an effect on regions.
pub fn construct_free_substs<'a,'tcx>(
    tcx: &'a ctxt<'tcx>,
    generics: &Generics<'tcx>,
    free_id: ast::NodeId)
    -> Substs<'tcx>
{
    // map T => T
    let mut types = VecPerParamSpace::empty();
    push_types_from_defs(tcx, &mut types, generics.types.as_slice());

    let free_id_outlive = region::DestructionScopeData::new(free_id);

    // map bound 'a => free 'a
    let mut regions = VecPerParamSpace::empty();
    push_region_params(&mut regions, free_id_outlive, generics.regions.as_slice());

    return Substs {
        types: types,
        regions: subst::NonerasedRegions(regions)
    };

    fn push_region_params(regions: &mut VecPerParamSpace<ty::Region>,
                          all_outlive_extent: region::DestructionScopeData,
                          region_params: &[RegionParameterDef])
    {
        for r in region_params {
            regions.push(r.space, ty::free_region_from_def(all_outlive_extent, r));
        }
    }

    fn push_types_from_defs<'tcx>(tcx: &ty::ctxt<'tcx>,
                                  types: &mut VecPerParamSpace<Ty<'tcx>>,
                                  defs: &[TypeParameterDef<'tcx>]) {
        for def in defs {
            debug!("construct_parameter_environment(): push_types_from_defs: def={:?}",
                   def.repr(tcx));
            let ty = ty::mk_param_from_def(tcx, def);
            types.push(def.space, ty);
       }
    }
}

/// See `ParameterEnvironment` struct def'n for details
pub fn construct_parameter_environment<'a,'tcx>(
    tcx: &'a ctxt<'tcx>,
    span: Span,
    generics: &ty::Generics<'tcx>,
    generic_predicates: &ty::GenericPredicates<'tcx>,
    free_id: ast::NodeId)
    -> ParameterEnvironment<'a, 'tcx>
{
    //
    // Construct the free substs.
    //

    let free_substs = construct_free_substs(tcx, generics, free_id);
    let free_id_outlive = region::DestructionScopeData::new(free_id);

    //
    // Compute the bounds on Self and the type parameters.
    //

    let bounds = generic_predicates.instantiate(tcx, &free_substs);
    let bounds = liberate_late_bound_regions(tcx, free_id_outlive, &ty::Binder(bounds));
    let predicates = bounds.predicates.into_vec();

    //
    // Compute region bounds. For now, these relations are stored in a
    // global table on the tcx, so just enter them there. I'm not
    // crazy about this scheme, but it's convenient, at least.
    //

    record_region_bounds(tcx, &*predicates);

    debug!("construct_parameter_environment: free_id={:?} free_subst={:?} predicates={:?}",
           free_id,
           free_substs.repr(tcx),
           predicates.repr(tcx));

    //
    // Finally, we have to normalize the bounds in the environment, in
    // case they contain any associated type projections. This process
    // can yield errors if the put in illegal associated types, like
    // `<i32 as Foo>::Bar` where `i32` does not implement `Foo`. We
    // report these errors right here; this doesn't actually feel
    // right to me, because constructing the environment feels like a
    // kind of a "idempotent" action, but I'm not sure where would be
    // a better place. In practice, we construct environments for
    // every fn once during type checking, and we'll abort if there
    // are any errors at that point, so after type checking you can be
    // sure that this will succeed without errors anyway.
    //

    let unnormalized_env = ty::ParameterEnvironment {
        tcx: tcx,
        free_substs: free_substs,
        implicit_region_bound: ty::ReScope(free_id_outlive.to_code_extent()),
        caller_bounds: predicates,
        selection_cache: traits::SelectionCache::new(),
    };

    let cause = traits::ObligationCause::misc(span, free_id);
    return traits::normalize_param_env_or_error(unnormalized_env, cause);

    fn record_region_bounds<'tcx>(tcx: &ty::ctxt<'tcx>, predicates: &[ty::Predicate<'tcx>]) {
        debug!("record_region_bounds(predicates={:?})", predicates.repr(tcx));

        for predicate in predicates {
            match *predicate {
                Predicate::Projection(..) |
                Predicate::Trait(..) |
                Predicate::Equate(..) |
                Predicate::TypeOutlives(..) => {
                    // No region bounds here
                }
                Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r_a, r_b))) => {
                    match (r_a, r_b) {
                        (ty::ReFree(fr_a), ty::ReFree(fr_b)) => {
                            // Record that `'a:'b`. Or, put another way, `'b <= 'a`.
                            tcx.region_maps.relate_free_regions(fr_b, fr_a);
                        }
                        _ => {
                            // All named regions are instantiated with free regions.
                            tcx.sess.bug(
                                &format!("record_region_bounds: non free region: {} / {}",
                                         r_a.repr(tcx),
                                         r_b.repr(tcx)));
                        }
                    }
                }
            }
        }
    }
}

impl BorrowKind {
    pub fn from_mutbl(m: ast::Mutability) -> BorrowKind {
        match m {
            ast::MutMutable => MutBorrow,
            ast::MutImmutable => ImmBorrow,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> ast::Mutability {
        match self {
            MutBorrow => ast::MutMutable,
            ImmBorrow => ast::MutImmutable,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            UniqueImmBorrow => ast::MutMutable,
        }
    }

    pub fn to_user_str(&self) -> &'static str {
        match *self {
            MutBorrow => "mutable",
            ImmBorrow => "immutable",
            UniqueImmBorrow => "uniquely immutable",
        }
    }
}

impl<'tcx> ctxt<'tcx> {
    pub fn is_method_call(&self, expr_id: ast::NodeId) -> bool {
        self.method_map.borrow().contains_key(&MethodCall::expr(expr_id))
    }

    pub fn upvar_capture(&self, upvar_id: ty::UpvarId) -> Option<ty::UpvarCapture> {
        Some(self.upvar_capture_map.borrow()[upvar_id].clone())
    }
}

impl<'a,'tcx> mc::Typer<'tcx> for ParameterEnvironment<'a,'tcx> {
    fn node_ty(&self, id: ast::NodeId) -> mc::McResult<Ty<'tcx>> {
        Ok(ty::node_id_to_type(self.tcx, id))
    }

    fn expr_ty_adjusted(&self, expr: &ast::Expr) -> mc::McResult<Ty<'tcx>> {
        Ok(ty::expr_ty_adjusted(self.tcx, expr))
    }

    fn node_method_ty(&self, method_call: ty::MethodCall) -> Option<Ty<'tcx>> {
        self.tcx.method_map.borrow().get(&method_call).map(|method| method.ty)
    }

    fn node_method_origin(&self, method_call: ty::MethodCall)
                          -> Option<ty::MethodOrigin<'tcx>>
    {
        self.tcx.method_map.borrow().get(&method_call).map(|method| method.origin.clone())
    }

    fn adjustments(&self) -> &RefCell<NodeMap<ty::AutoAdjustment<'tcx>>> {
        &self.tcx.adjustments
    }

    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.tcx.is_method_call(id)
    }

    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<region::CodeExtent> {
        self.tcx.region_maps.temporary_scope(rvalue_id)
    }

    fn upvar_capture(&self, upvar_id: ty::UpvarId) -> Option<ty::UpvarCapture> {
        self.tcx.upvar_capture(upvar_id)
    }

    fn type_moves_by_default(&self, span: Span, ty: Ty<'tcx>) -> bool {
        type_moves_by_default(self, span, ty)
    }
}

impl<'a,'tcx> ClosureTyper<'tcx> for ty::ParameterEnvironment<'a,'tcx> {
    fn param_env<'b>(&'b self) -> &'b ty::ParameterEnvironment<'b,'tcx> {
        self
    }

    fn closure_kind(&self,
                    def_id: ast::DefId)
                    -> Option<ty::ClosureKind>
    {
        Some(self.tcx.closure_kind(def_id))
    }

    fn closure_type(&self,
                    def_id: ast::DefId,
                    substs: &subst::Substs<'tcx>)
                    -> ty::ClosureTy<'tcx>
    {
        self.tcx.closure_type(def_id, substs)
    }

    fn closure_upvars(&self,
                      def_id: ast::DefId,
                      substs: &Substs<'tcx>)
                      -> Option<Vec<ClosureUpvar<'tcx>>>
    {
        closure_upvars(self, def_id, substs)
    }
}


/// The category of explicit self.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum ExplicitSelfCategory {
    StaticExplicitSelfCategory,
    ByValueExplicitSelfCategory,
    ByReferenceExplicitSelfCategory(Region, ast::Mutability),
    ByBoxExplicitSelfCategory,
}

/// Pushes all the lifetimes in the given type onto the given list. A
/// "lifetime in a type" is a lifetime specified by a reference or a lifetime
/// in a list of type substitutions. This does *not* traverse into nominal
/// types, nor does it resolve fictitious types.
pub fn accumulate_lifetimes_in_type(accumulator: &mut Vec<ty::Region>,
                                    ty: Ty) {
    walk_ty(ty, |ty| {
        match ty.sty {
            ty_rptr(region, _) => {
                accumulator.push(*region)
            }
            ty_trait(ref t) => {
                accumulator.push_all(t.principal.0.substs.regions().as_slice());
            }
            ty_enum(_, substs) |
            ty_struct(_, substs) => {
                accum_substs(accumulator, substs);
            }
            ty_closure(_, region, substs) => {
                accumulator.push(*region);
                accum_substs(accumulator, substs);
            }
            ty_bool |
            ty_char |
            ty_int(_) |
            ty_uint(_) |
            ty_float(_) |
            ty_uniq(_) |
            ty_str |
            ty_vec(_, _) |
            ty_ptr(_) |
            ty_bare_fn(..) |
            ty_tup(_) |
            ty_projection(_) |
            ty_param(_) |
            ty_infer(_) |
            ty_open(_) |
            ty_err => {
            }
        }
    });

    fn accum_substs(accumulator: &mut Vec<Region>, substs: &Substs) {
        match substs.regions {
            subst::ErasedRegions => {}
            subst::NonerasedRegions(ref regions) => {
                for region in regions.iter() {
                    accumulator.push(*region)
                }
            }
        }
    }
}

/// A free variable referred to in a function.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Freevar {
    /// The variable being accessed free.
    pub def: def::Def,

    // First span where it is accessed (there can be multiple).
    pub span: Span
}

pub type FreevarMap = NodeMap<Vec<Freevar>>;

pub type CaptureModeMap = NodeMap<ast::CaptureClause>;

// Trait method resolution
pub type TraitMap = NodeMap<Vec<DefId>>;

// Map from the NodeId of a glob import to a list of items which are actually
// imported.
pub type GlobMap = HashMap<NodeId, HashSet<Name>>;

pub fn with_freevars<T, F>(tcx: &ty::ctxt, fid: ast::NodeId, f: F) -> T where
    F: FnOnce(&[Freevar]) -> T,
{
    match tcx.freevars.borrow().get(&fid) {
        None => f(&[]),
        Some(d) => f(&d[..])
    }
}

impl<'tcx> AutoAdjustment<'tcx> {
    pub fn is_identity(&self) -> bool {
        match *self {
            AdjustReifyFnPointer(..) => false,
            AdjustDerefRef(ref r) => r.is_identity(),
        }
    }
}

impl<'tcx> AutoDerefRef<'tcx> {
    pub fn is_identity(&self) -> bool {
        self.autoderefs == 0 && self.autoref.is_none()
    }
}

/// Replace any late-bound regions bound in `value` with free variants attached to scope-id
/// `scope_id`.
pub fn liberate_late_bound_regions<'tcx, T>(
    tcx: &ty::ctxt<'tcx>,
    all_outlive_scope: region::DestructionScopeData,
    value: &Binder<T>)
    -> T
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    replace_late_bound_regions(
        tcx, value,
        |br| ty::ReFree(ty::FreeRegion{scope: all_outlive_scope, bound_region: br})).0
}

pub fn count_late_bound_regions<'tcx, T>(
    tcx: &ty::ctxt<'tcx>,
    value: &Binder<T>)
    -> uint
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    let (_, skol_map) = replace_late_bound_regions(tcx, value, |_| ty::ReStatic);
    skol_map.len()
}

pub fn binds_late_bound_regions<'tcx, T>(
    tcx: &ty::ctxt<'tcx>,
    value: &Binder<T>)
    -> bool
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    count_late_bound_regions(tcx, value) > 0
}

pub fn no_late_bound_regions<'tcx, T>(
    tcx: &ty::ctxt<'tcx>,
    value: &Binder<T>)
    -> Option<T>
    where T : TypeFoldable<'tcx> + Repr<'tcx> + Clone
{
    if binds_late_bound_regions(tcx, value) {
        None
    } else {
        Some(value.0.clone())
    }
}

/// Replace any late-bound regions bound in `value` with `'static`. Useful in trans but also
/// method lookup and a few other places where precise region relationships are not required.
pub fn erase_late_bound_regions<'tcx, T>(
    tcx: &ty::ctxt<'tcx>,
    value: &Binder<T>)
    -> T
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    replace_late_bound_regions(tcx, value, |_| ty::ReStatic).0
}

/// Rewrite any late-bound regions so that they are anonymous.  Region numbers are
/// assigned starting at 1 and increasing monotonically in the order traversed
/// by the fold operation.
///
/// The chief purpose of this function is to canonicalize regions so that two
/// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
/// structurally identical.  For example, `for<'a, 'b> fn(&'a int, &'b int)` and
/// `for<'a, 'b> fn(&'b int, &'a int)` will become identical after anonymization.
pub fn anonymize_late_bound_regions<'tcx, T>(
    tcx: &ctxt<'tcx>,
    sig: &Binder<T>)
    -> Binder<T>
    where T : TypeFoldable<'tcx> + Repr<'tcx>,
{
    let mut counter = 0;
    ty::Binder(replace_late_bound_regions(tcx, sig, |_| {
        counter += 1;
        ReLateBound(ty::DebruijnIndex::new(1), BrAnon(counter))
    }).0)
}

/// Replaces the late-bound-regions in `value` that are bound by `value`.
pub fn replace_late_bound_regions<'tcx, T, F>(
    tcx: &ty::ctxt<'tcx>,
    binder: &Binder<T>,
    mut mapf: F)
    -> (T, FnvHashMap<ty::BoundRegion,ty::Region>)
    where T : TypeFoldable<'tcx> + Repr<'tcx>,
          F : FnMut(BoundRegion) -> ty::Region,
{
    debug!("replace_late_bound_regions({})", binder.repr(tcx));

    let mut map = FnvHashMap();

    // Note: fold the field `0`, not the binder, so that late-bound
    // regions bound by `binder` are considered free.
    let value = ty_fold::fold_regions(tcx, &binder.0, |region, current_depth| {
        debug!("region={}", region.repr(tcx));
        match region {
            ty::ReLateBound(debruijn, br) if debruijn.depth == current_depth => {
                let region =
                    * map.entry(br).get().unwrap_or_else(
                        |vacant_entry| vacant_entry.insert(mapf(br)));

                if let ty::ReLateBound(debruijn1, br) = region {
                    // If the callback returns a late-bound region,
                    // that region should always use depth 1. Then we
                    // adjust it to the correct depth.
                    assert_eq!(debruijn1.depth, 1);
                    ty::ReLateBound(debruijn, br)
                } else {
                    region
                }
            }
            _ => {
                region
            }
        }
    });

    debug!("resulting map: {:?} value: {:?}", map, value.repr(tcx));
    (value, map)
}

impl DebruijnIndex {
    pub fn new(depth: u32) -> DebruijnIndex {
        assert!(depth > 0);
        DebruijnIndex { depth: depth }
    }

    pub fn shifted(&self, amount: u32) -> DebruijnIndex {
        DebruijnIndex { depth: self.depth + amount }
    }
}

impl<'tcx> Repr<'tcx> for AutoAdjustment<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            AdjustReifyFnPointer(def_id) => {
                format!("AdjustReifyFnPointer({})", def_id.repr(tcx))
            }
            AdjustDerefRef(ref data) => {
                data.repr(tcx)
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for UnsizeKind<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            UnsizeLength(n) => format!("UnsizeLength({})", n),
            UnsizeStruct(ref k, n) => format!("UnsizeStruct({},{})", k.repr(tcx), n),
            UnsizeVtable(ref a, ref b) => format!("UnsizeVtable({},{})", a.repr(tcx), b.repr(tcx)),
        }
    }
}

impl<'tcx> Repr<'tcx> for AutoDerefRef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("AutoDerefRef({}, {})", self.autoderefs, self.autoref.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for AutoRef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            AutoPtr(a, b, ref c) => {
                format!("AutoPtr({},{:?},{})", a.repr(tcx), b, c.repr(tcx))
            }
            AutoUnsize(ref a) => {
                format!("AutoUnsize({})", a.repr(tcx))
            }
            AutoUnsizeUniq(ref a) => {
                format!("AutoUnsizeUniq({})", a.repr(tcx))
            }
            AutoUnsafe(ref a, ref b) => {
                format!("AutoUnsafe({:?},{})", a, b.repr(tcx))
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for TyTrait<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TyTrait({},{})",
                self.principal.repr(tcx),
                self.bounds.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Predicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            Predicate::Trait(ref a) => a.repr(tcx),
            Predicate::Equate(ref pair) => pair.repr(tcx),
            Predicate::RegionOutlives(ref pair) => pair.repr(tcx),
            Predicate::TypeOutlives(ref pair) => pair.repr(tcx),
            Predicate::Projection(ref pair) => pair.repr(tcx),
        }
    }
}

impl<'tcx> Repr<'tcx> for vtable_origin<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            vtable_static(def_id, ref tys, ref vtable_res) => {
                format!("vtable_static({:?}:{}, {}, {})",
                        def_id,
                        ty::item_path_str(tcx, def_id),
                        tys.repr(tcx),
                        vtable_res.repr(tcx))
            }

            vtable_param(x, y) => {
                format!("vtable_param({:?}, {})", x, y)
            }

            vtable_closure(def_id) => {
                format!("vtable_closure({:?})", def_id)
            }

            vtable_error => {
                format!("vtable_error")
            }
        }
    }
}

pub fn make_substs_for_receiver_types<'tcx>(tcx: &ty::ctxt<'tcx>,
                                            trait_ref: &ty::TraitRef<'tcx>,
                                            method: &ty::Method<'tcx>)
                                            -> subst::Substs<'tcx>
{
    /*!
     * Substitutes the values for the receiver's type parameters
     * that are found in method, leaving the method's type parameters
     * intact.
     */

    let meth_tps: Vec<Ty> =
        method.generics.types.get_slice(subst::FnSpace)
              .iter()
              .map(|def| ty::mk_param_from_def(tcx, def))
              .collect();
    let meth_regions: Vec<ty::Region> =
        method.generics.regions.get_slice(subst::FnSpace)
              .iter()
              .map(|def| ty::ReEarlyBound(def.def_id.node, def.space,
                                          def.index, def.name))
              .collect();
    trait_ref.substs.clone().with_method(meth_tps, meth_regions)
}

#[derive(Copy)]
pub enum CopyImplementationError {
    FieldDoesNotImplementCopy(ast::Name),
    VariantDoesNotImplementCopy(ast::Name),
    TypeIsStructural,
    TypeHasDestructor,
}

pub fn can_type_implement_copy<'a,'tcx>(param_env: &ParameterEnvironment<'a, 'tcx>,
                                        span: Span,
                                        self_type: Ty<'tcx>)
                                        -> Result<(),CopyImplementationError>
{
    let tcx = param_env.tcx;

    let did = match self_type.sty {
        ty::ty_struct(struct_did, substs) => {
            let fields = ty::struct_fields(tcx, struct_did, substs);
            for field in &fields {
                if type_moves_by_default(param_env, span, field.mt.ty) {
                    return Err(FieldDoesNotImplementCopy(field.name))
                }
            }
            struct_did
        }
        ty::ty_enum(enum_did, substs) => {
            let enum_variants = ty::enum_variants(tcx, enum_did);
            for variant in &*enum_variants {
                for variant_arg_type in &variant.args {
                    let substd_arg_type =
                        variant_arg_type.subst(tcx, substs);
                    if type_moves_by_default(param_env, span, substd_arg_type) {
                        return Err(VariantDoesNotImplementCopy(variant.name))
                    }
                }
            }
            enum_did
        }
        _ => return Err(TypeIsStructural),
    };

    if ty::has_dtor(tcx, did) {
        return Err(TypeHasDestructor)
    }

    Ok(())
}

// FIXME(#20298) -- all of these types basically walk various
// structures to test whether types/regions are reachable with various
// properties. It should be possible to express them in terms of one
// common "walker" trait or something.

pub trait RegionEscape {
    fn has_escaping_regions(&self) -> bool {
        self.has_regions_escaping_depth(0)
    }

    fn has_regions_escaping_depth(&self, depth: u32) -> bool;
}

impl<'tcx> RegionEscape for Ty<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        ty::type_escapes_depth(*self, depth)
    }
}

impl<'tcx> RegionEscape for Substs<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.types.has_regions_escaping_depth(depth) ||
            self.regions.has_regions_escaping_depth(depth)
    }
}

impl<'tcx,T:RegionEscape> RegionEscape for VecPerParamSpace<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.iter_enumerated().any(|(space, _, t)| {
            if space == subst::FnSpace {
                t.has_regions_escaping_depth(depth+1)
            } else {
                t.has_regions_escaping_depth(depth)
            }
        })
    }
}

impl<'tcx> RegionEscape for TypeScheme<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.ty.has_regions_escaping_depth(depth)
    }
}

impl RegionEscape for Region {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.escapes_depth(depth)
    }
}

impl<'tcx> RegionEscape for GenericPredicates<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.predicates.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for Predicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            Predicate::Trait(ref data) => data.has_regions_escaping_depth(depth),
            Predicate::Equate(ref data) => data.has_regions_escaping_depth(depth),
            Predicate::RegionOutlives(ref data) => data.has_regions_escaping_depth(depth),
            Predicate::TypeOutlives(ref data) => data.has_regions_escaping_depth(depth),
            Predicate::Projection(ref data) => data.has_regions_escaping_depth(depth),
        }
    }
}

impl<'tcx> RegionEscape for TraitRef<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.substs.types.iter().any(|t| t.has_regions_escaping_depth(depth)) ||
            self.substs.regions.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for subst::RegionSubsts {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            subst::ErasedRegions => false,
            subst::NonerasedRegions(ref r) => {
                r.iter().any(|t| t.has_regions_escaping_depth(depth))
            }
        }
    }
}

impl<'tcx,T:RegionEscape> RegionEscape for Binder<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth + 1)
    }
}

impl<'tcx> RegionEscape for EquatePredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth) || self.1.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for TraitPredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.trait_ref.has_regions_escaping_depth(depth)
    }
}

impl<T:RegionEscape,U:RegionEscape> RegionEscape for OutlivesPredicate<T,U> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth) || self.1.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ProjectionPredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.projection_ty.has_regions_escaping_depth(depth) ||
            self.ty.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ProjectionTy<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.trait_ref.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> Repr<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ProjectionPredicate({}, {})",
                self.projection_ty.repr(tcx),
                self.ty.repr(tcx))
    }
}

pub trait HasProjectionTypes {
    fn has_projection_types(&self) -> bool;
}

impl<'tcx,T:HasProjectionTypes> HasProjectionTypes for Vec<T> {
    fn has_projection_types(&self) -> bool {
        self.iter().any(|p| p.has_projection_types())
    }
}

impl<'tcx,T:HasProjectionTypes> HasProjectionTypes for VecPerParamSpace<T> {
    fn has_projection_types(&self) -> bool {
        self.iter().any(|p| p.has_projection_types())
    }
}

impl<'tcx> HasProjectionTypes for ClosureTy<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.sig.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for ClosureUpvar<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.ty.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for ty::InstantiatedPredicates<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.predicates.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for Predicate<'tcx> {
    fn has_projection_types(&self) -> bool {
        match *self {
            Predicate::Trait(ref data) => data.has_projection_types(),
            Predicate::Equate(ref data) => data.has_projection_types(),
            Predicate::RegionOutlives(ref data) => data.has_projection_types(),
            Predicate::TypeOutlives(ref data) => data.has_projection_types(),
            Predicate::Projection(ref data) => data.has_projection_types(),
        }
    }
}

impl<'tcx> HasProjectionTypes for TraitPredicate<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.trait_ref.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for EquatePredicate<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.0.has_projection_types() || self.1.has_projection_types()
    }
}

impl HasProjectionTypes for Region {
    fn has_projection_types(&self) -> bool {
        false
    }
}

impl<T:HasProjectionTypes,U:HasProjectionTypes> HasProjectionTypes for OutlivesPredicate<T,U> {
    fn has_projection_types(&self) -> bool {
        self.0.has_projection_types() || self.1.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for ProjectionPredicate<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.projection_ty.has_projection_types() || self.ty.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for ProjectionTy<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.trait_ref.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for Ty<'tcx> {
    fn has_projection_types(&self) -> bool {
        ty::type_has_projection(*self)
    }
}

impl<'tcx> HasProjectionTypes for TraitRef<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.substs.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for subst::Substs<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.types.iter().any(|t| t.has_projection_types())
    }
}

impl<'tcx,T> HasProjectionTypes for Option<T>
    where T : HasProjectionTypes
{
    fn has_projection_types(&self) -> bool {
        self.iter().any(|t| t.has_projection_types())
    }
}

impl<'tcx,T> HasProjectionTypes for Rc<T>
    where T : HasProjectionTypes
{
    fn has_projection_types(&self) -> bool {
        (**self).has_projection_types()
    }
}

impl<'tcx,T> HasProjectionTypes for Box<T>
    where T : HasProjectionTypes
{
    fn has_projection_types(&self) -> bool {
        (**self).has_projection_types()
    }
}

impl<T> HasProjectionTypes for Binder<T>
    where T : HasProjectionTypes
{
    fn has_projection_types(&self) -> bool {
        self.0.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for FnOutput<'tcx> {
    fn has_projection_types(&self) -> bool {
        match *self {
            FnConverging(t) => t.has_projection_types(),
            FnDiverging => false,
        }
    }
}

impl<'tcx> HasProjectionTypes for FnSig<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.inputs.iter().any(|t| t.has_projection_types()) ||
            self.output.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for field<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.mt.ty.has_projection_types()
    }
}

impl<'tcx> HasProjectionTypes for BareFnTy<'tcx> {
    fn has_projection_types(&self) -> bool {
        self.sig.has_projection_types()
    }
}

pub trait ReferencesError {
    fn references_error(&self) -> bool;
}

impl<T:ReferencesError> ReferencesError for Binder<T> {
    fn references_error(&self) -> bool {
        self.0.references_error()
    }
}

impl<T:ReferencesError> ReferencesError for Rc<T> {
    fn references_error(&self) -> bool {
        (&**self).references_error()
    }
}

impl<'tcx> ReferencesError for TraitPredicate<'tcx> {
    fn references_error(&self) -> bool {
        self.trait_ref.references_error()
    }
}

impl<'tcx> ReferencesError for ProjectionPredicate<'tcx> {
    fn references_error(&self) -> bool {
        self.projection_ty.trait_ref.references_error() || self.ty.references_error()
    }
}

impl<'tcx> ReferencesError for TraitRef<'tcx> {
    fn references_error(&self) -> bool {
        self.input_types().iter().any(|t| t.references_error())
    }
}

impl<'tcx> ReferencesError for Ty<'tcx> {
    fn references_error(&self) -> bool {
        type_is_error(*self)
    }
}

impl<'tcx> ReferencesError for Predicate<'tcx> {
    fn references_error(&self) -> bool {
        match *self {
            Predicate::Trait(ref data) => data.references_error(),
            Predicate::Equate(ref data) => data.references_error(),
            Predicate::RegionOutlives(ref data) => data.references_error(),
            Predicate::TypeOutlives(ref data) => data.references_error(),
            Predicate::Projection(ref data) => data.references_error(),
        }
    }
}

impl<A,B> ReferencesError for OutlivesPredicate<A,B>
    where A : ReferencesError, B : ReferencesError
{
    fn references_error(&self) -> bool {
        self.0.references_error() || self.1.references_error()
    }
}

impl<'tcx> ReferencesError for EquatePredicate<'tcx>
{
    fn references_error(&self) -> bool {
        self.0.references_error() || self.1.references_error()
    }
}

impl ReferencesError for Region
{
    fn references_error(&self) -> bool {
        false
    }
}

impl<'tcx> Repr<'tcx> for ClosureTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ClosureTy({},{},{})",
                self.unsafety,
                self.sig.repr(tcx),
                self.abi)
    }
}

impl<'tcx> Repr<'tcx> for ClosureUpvar<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ClosureUpvar({},{})",
                self.def.repr(tcx),
                self.ty.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for field<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("field({},{})",
                self.name.repr(tcx),
                self.mt.repr(tcx))
    }
}

impl<'a, 'tcx> Repr<'tcx> for ParameterEnvironment<'a, 'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ParameterEnvironment(\
            free_substs={}, \
            implicit_region_bound={}, \
            caller_bounds={})",
            self.free_substs.repr(tcx),
            self.implicit_region_bound.repr(tcx),
            self.caller_bounds.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ObjectLifetimeDefault {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            ObjectLifetimeDefault::Ambiguous => format!("Ambiguous"),
            ObjectLifetimeDefault::Specific(ref r) => r.repr(tcx),
        }
    }
}
