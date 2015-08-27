// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: (@jroesch) @eddyb should remove this when he renames ctxt
#![allow(non_camel_case_types)]

pub use self::InferTy::*;
pub use self::ImplOrTraitItemId::*;
pub use self::ClosureKind::*;
pub use self::Variance::*;
pub use self::AutoAdjustment::*;
pub use self::Representability::*;
pub use self::AutoRef::*;
pub use self::DtorKind::*;
pub use self::ExplicitSelfCategory::*;
pub use self::FnOutput::*;
pub use self::Region::*;
pub use self::ImplOrTraitItemContainer::*;
pub use self::BorrowKind::*;
pub use self::ImplOrTraitItem::*;
pub use self::BoundRegion::*;
pub use self::TypeVariants::*;
pub use self::IntVarValue::*;
pub use self::CopyImplementationError::*;

pub use self::BuiltinBound::Send as BoundSend;
pub use self::BuiltinBound::Sized as BoundSized;
pub use self::BuiltinBound::Copy as BoundCopy;
pub use self::BuiltinBound::Sync as BoundSync;

use ast_map::{self, LinkedPath};
use back::svh::Svh;
use session::Session;
use lint;
use metadata::csearch;
use middle;
use middle::cast;
use middle::check_const;
use middle::const_eval::{self, ConstVal, ErrKind};
use middle::const_eval::EvalHint::UncheckedExprHint;
use middle::def::{self, DefMap, ExportMap};
use middle::def_id::{DefId, LOCAL_CRATE};
use middle::fast_reject;
use middle::free_region::FreeRegionMap;
use middle::lang_items::{FnTraitLangItem, FnMutTraitLangItem, FnOnceTraitLangItem};
use middle::region;
use middle::resolve_lifetime;
use middle::infer;
use middle::infer::type_variable;
use middle::pat_util;
use middle::region::RegionMaps;
use middle::stability;
use middle::subst::{self, ParamSpace, Subst, Substs, VecPerParamSpace};
use middle::traits;
use middle::ty;
use middle::ty_fold::{self, TypeFoldable, TypeFolder};
use middle::ty_walk::{self, TypeWalker};
use util::common::{memoized, ErrorReported};
use util::nodemap::{NodeMap, NodeSet, DefIdMap, DefIdSet};
use util::nodemap::FnvHashMap;
use util::num::ToPrimitive;

use arena::TypedArena;
use std::borrow::{Borrow, Cow};
use std::cell::{Cell, RefCell, Ref};
use std::cmp;
use std::fmt;
use std::hash::{Hash, SipHasher, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops;
use std::rc::Rc;
use std::slice;
use std::vec::IntoIter;
use collections::enum_set::{self, EnumSet, CLike};
use core::nonzero::NonZero;
use std::collections::{HashMap, HashSet};
use rustc_data_structures::ivar;
use syntax::abi;
use syntax::ast::{CrateNum, ItemImpl, ItemTrait};
use syntax::ast::{MutImmutable, MutMutable, Name, NodeId, Visibility};
use syntax::attr::{self, AttrMetaMethods, SignedInt, UnsignedInt};
use syntax::codemap::Span;
use syntax::parse::token::{InternedString, special_idents};
use syntax::ast;

pub type Disr = u64;

pub const INITIAL_DISCRIMINANT_VALUE: Disr = 0;

// Data types

/// The complete set of all analyses described in this module. This is
/// produced by the driver and fed to trans and later passes.
pub struct CrateAnalysis {
    pub export_map: ExportMap,
    pub exported_items: middle::privacy::ExportedItems,
    pub public_items: middle::privacy::PublicItems,
    pub reachable: NodeSet,
    pub name: String,
    pub glob_map: Option<GlobMap>,
}

#[derive(Copy, Clone)]
pub enum DtorKind {
    NoDtor,
    TraitDtor(bool)
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
            &TraitDtor(flag) => flag
        }
    }
}

pub trait IntTypeExt {
    fn to_ty<'tcx>(&self, cx: &ctxt<'tcx>) -> Ty<'tcx>;
    fn i64_to_disr(&self, val: i64) -> Option<Disr>;
    fn u64_to_disr(&self, val: u64) -> Option<Disr>;
    fn disr_incr(&self, val: Disr) -> Option<Disr>;
    fn disr_string(&self, val: Disr) -> String;
    fn disr_wrap_incr(&self, val: Option<Disr>) -> Disr;
}

impl IntTypeExt for attr::IntType {
    fn to_ty<'tcx>(&self, cx: &ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            SignedInt(ast::TyI8)      => cx.types.i8,
            SignedInt(ast::TyI16)     => cx.types.i16,
            SignedInt(ast::TyI32)     => cx.types.i32,
            SignedInt(ast::TyI64)     => cx.types.i64,
            SignedInt(ast::TyIs)   => cx.types.isize,
            UnsignedInt(ast::TyU8)    => cx.types.u8,
            UnsignedInt(ast::TyU16)   => cx.types.u16,
            UnsignedInt(ast::TyU32)   => cx.types.u32,
            UnsignedInt(ast::TyU64)   => cx.types.u64,
            UnsignedInt(ast::TyUs) => cx.types.usize,
        }
    }

    fn i64_to_disr(&self, val: i64) -> Option<Disr> {
        match *self {
            SignedInt(ast::TyI8)    => val.to_i8()  .map(|v| v as Disr),
            SignedInt(ast::TyI16)   => val.to_i16() .map(|v| v as Disr),
            SignedInt(ast::TyI32)   => val.to_i32() .map(|v| v as Disr),
            SignedInt(ast::TyI64)   => val.to_i64() .map(|v| v as Disr),
            UnsignedInt(ast::TyU8)  => val.to_u8()  .map(|v| v as Disr),
            UnsignedInt(ast::TyU16) => val.to_u16() .map(|v| v as Disr),
            UnsignedInt(ast::TyU32) => val.to_u32() .map(|v| v as Disr),
            UnsignedInt(ast::TyU64) => val.to_u64() .map(|v| v as Disr),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn u64_to_disr(&self, val: u64) -> Option<Disr> {
        match *self {
            SignedInt(ast::TyI8)    => val.to_i8()  .map(|v| v as Disr),
            SignedInt(ast::TyI16)   => val.to_i16() .map(|v| v as Disr),
            SignedInt(ast::TyI32)   => val.to_i32() .map(|v| v as Disr),
            SignedInt(ast::TyI64)   => val.to_i64() .map(|v| v as Disr),
            UnsignedInt(ast::TyU8)  => val.to_u8()  .map(|v| v as Disr),
            UnsignedInt(ast::TyU16) => val.to_u16() .map(|v| v as Disr),
            UnsignedInt(ast::TyU32) => val.to_u32() .map(|v| v as Disr),
            UnsignedInt(ast::TyU64) => val.to_u64() .map(|v| v as Disr),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn disr_incr(&self, val: Disr) -> Option<Disr> {
        macro_rules! add1 {
            ($e:expr) => { $e.and_then(|v|v.checked_add(1)).map(|v| v as Disr) }
        }
        match *self {
            // SignedInt repr means we *want* to reinterpret the bits
            // treating the highest bit of Disr as a sign-bit, so
            // cast to i64 before range-checking.
            SignedInt(ast::TyI8)    => add1!((val as i64).to_i8()),
            SignedInt(ast::TyI16)   => add1!((val as i64).to_i16()),
            SignedInt(ast::TyI32)   => add1!((val as i64).to_i32()),
            SignedInt(ast::TyI64)   => add1!(Some(val as i64)),

            UnsignedInt(ast::TyU8)  => add1!(val.to_u8()),
            UnsignedInt(ast::TyU16) => add1!(val.to_u16()),
            UnsignedInt(ast::TyU32) => add1!(val.to_u32()),
            UnsignedInt(ast::TyU64) => add1!(Some(val)),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    // This returns a String because (1.) it is only used for
    // rendering an error message and (2.) a string can represent the
    // full range from `i64::MIN` through `u64::MAX`.
    fn disr_string(&self, val: Disr) -> String {
        match *self {
            SignedInt(ast::TyI8)    => format!("{}", val as i8 ),
            SignedInt(ast::TyI16)   => format!("{}", val as i16),
            SignedInt(ast::TyI32)   => format!("{}", val as i32),
            SignedInt(ast::TyI64)   => format!("{}", val as i64),
            UnsignedInt(ast::TyU8)  => format!("{}", val as u8 ),
            UnsignedInt(ast::TyU16) => format!("{}", val as u16),
            UnsignedInt(ast::TyU32) => format!("{}", val as u32),
            UnsignedInt(ast::TyU64) => format!("{}", val as u64),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn disr_wrap_incr(&self, val: Option<Disr>) -> Disr {
        macro_rules! add1 {
            ($e:expr) => { ($e).wrapping_add(1) as Disr }
        }
        let val = val.unwrap_or(ty::INITIAL_DISCRIMINANT_VALUE);
        match *self {
            SignedInt(ast::TyI8)    => add1!(val as i8 ),
            SignedInt(ast::TyI16)   => add1!(val as i16),
            SignedInt(ast::TyI32)   => add1!(val as i32),
            SignedInt(ast::TyI64)   => add1!(val as i64),
            UnsignedInt(ast::TyU8)  => add1!(val as u8 ),
            UnsignedInt(ast::TyU16) => add1!(val as u16),
            UnsignedInt(ast::TyU32) => add1!(val as u32),
            UnsignedInt(ast::TyU64) => add1!(val as u64),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImplOrTraitItemContainer {
    TraitContainer(DefId),
    ImplContainer(DefId),
}

impl ImplOrTraitItemContainer {
    pub fn id(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[derive(Clone)]
pub enum ImplOrTraitItem<'tcx> {
    ConstTraitItem(Rc<AssociatedConst<'tcx>>),
    MethodTraitItem(Rc<Method<'tcx>>),
    TypeTraitItem(Rc<AssociatedType<'tcx>>),
}

impl<'tcx> ImplOrTraitItem<'tcx> {
    fn id(&self) -> ImplOrTraitItemId {
        match *self {
            ConstTraitItem(ref associated_const) => {
                ConstTraitItemId(associated_const.def_id)
            }
            MethodTraitItem(ref method) => MethodTraitItemId(method.def_id),
            TypeTraitItem(ref associated_type) => {
                TypeTraitItemId(associated_type.def_id)
            }
        }
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            ConstTraitItem(ref associated_const) => associated_const.def_id,
            MethodTraitItem(ref method) => method.def_id,
            TypeTraitItem(ref associated_type) => associated_type.def_id,
        }
    }

    pub fn name(&self) -> ast::Name {
        match *self {
            ConstTraitItem(ref associated_const) => associated_const.name,
            MethodTraitItem(ref method) => method.name,
            TypeTraitItem(ref associated_type) => associated_type.name,
        }
    }

    pub fn vis(&self) -> ast::Visibility {
        match *self {
            ConstTraitItem(ref associated_const) => associated_const.vis,
            MethodTraitItem(ref method) => method.vis,
            TypeTraitItem(ref associated_type) => associated_type.vis,
        }
    }

    pub fn container(&self) -> ImplOrTraitItemContainer {
        match *self {
            ConstTraitItem(ref associated_const) => associated_const.container,
            MethodTraitItem(ref method) => method.container,
            TypeTraitItem(ref associated_type) => associated_type.container,
        }
    }

    pub fn as_opt_method(&self) -> Option<Rc<Method<'tcx>>> {
        match *self {
            MethodTraitItem(ref m) => Some((*m).clone()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ImplOrTraitItemId {
    ConstTraitItemId(DefId),
    MethodTraitItemId(DefId),
    TypeTraitItemId(DefId),
}

impl ImplOrTraitItemId {
    pub fn def_id(&self) -> DefId {
        match *self {
            ConstTraitItemId(def_id) => def_id,
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
    pub def_id: DefId,
    pub container: ImplOrTraitItemContainer,

    // If this method is provided, we need to know where it came from
    pub provided_source: Option<DefId>
}

impl<'tcx> Method<'tcx> {
    pub fn new(name: ast::Name,
               generics: ty::Generics<'tcx>,
               predicates: GenericPredicates<'tcx>,
               fty: BareFnTy<'tcx>,
               explicit_self: ExplicitSelfCategory,
               vis: ast::Visibility,
               def_id: DefId,
               container: ImplOrTraitItemContainer,
               provided_source: Option<DefId>)
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

    pub fn container_id(&self) -> DefId {
        match self.container {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AssociatedConst<'tcx> {
    pub name: ast::Name,
    pub ty: Ty<'tcx>,
    pub vis: ast::Visibility,
    pub def_id: DefId,
    pub container: ImplOrTraitItemContainer,
    pub default: Option<DefId>,
}

#[derive(Clone, Copy, Debug)]
pub struct AssociatedType<'tcx> {
    pub name: ast::Name,
    pub ty: Option<Ty<'tcx>>,
    pub vis: ast::Visibility,
    pub def_id: DefId,
    pub container: ImplOrTraitItemContainer,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TypeAndMut<'tcx> {
    pub ty: Ty<'tcx>,
    pub mutbl: ast::Mutability,
}

#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable)]
pub struct ItemVariances {
    pub types: VecPerParamSpace<Variance>,
    pub regions: VecPerParamSpace<Variance>,
}

#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable, Copy)]
pub enum Variance {
    Covariant,      // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,      // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant,  // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,      // T<A> <: T<B>            -- e.g., unused type parameter
}

impl fmt::Debug for Variance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            Covariant => "+",
            Contravariant => "-",
            Invariant => "o",
            Bivariant => "*",
        })
    }
}

#[derive(Copy, Clone)]
pub enum AutoAdjustment<'tcx> {
    AdjustReifyFnPointer,   // go from a fn-item type to a fn-pointer type
    AdjustUnsafeFnPointer,  // go from a safe fn pointer to an unsafe fn pointer
    AdjustDerefRef(AutoDerefRef<'tcx>),
}

/// Represents coercing a pointer to a different kind of pointer - where 'kind'
/// here means either or both of raw vs borrowed vs unique and fat vs thin.
///
/// We transform pointers by following the following steps in order:
/// 1. Deref the pointer `self.autoderefs` times (may be 0).
/// 2. If `autoref` is `Some(_)`, then take the address and produce either a
///    `&` or `*` pointer.
/// 3. If `unsize` is `Some(_)`, then apply the unsize transformation,
///    which will do things like convert thin pointers to fat
///    pointers, or convert structs containing thin pointers to
///    structs containing fat pointers, or convert between fat
///    pointers.  We don't store the details of how the transform is
///    done (in fact, we don't know that, because it might depend on
///    the precise type parameters). We just store the target
///    type. Trans figures out what has to be done at monomorphization
///    time based on the precise source/target type at hand.
///
/// To make that more concrete, here are some common scenarios:
///
/// 1. The simplest cases are where the pointer is not adjusted fat vs thin.
/// Here the pointer will be dereferenced N times (where a dereference can
/// happen to to raw or borrowed pointers or any smart pointer which implements
/// Deref, including Box<_>). The number of dereferences is given by
/// `autoderefs`.  It can then be auto-referenced zero or one times, indicated
/// by `autoref`, to either a raw or borrowed pointer. In these cases unsize is
/// None.
///
/// 2. A thin-to-fat coercon involves unsizing the underlying data. We start
/// with a thin pointer, deref a number of times, unsize the underlying data,
/// then autoref. The 'unsize' phase may change a fixed length array to a
/// dynamically sized one, a concrete object to a trait object, or statically
/// sized struct to a dyncamically sized one. E.g., &[i32; 4] -> &[i32] is
/// represented by:
///
/// ```
/// AutoDerefRef {
///     autoderefs: 1,          // &[i32; 4] -> [i32; 4]
///     autoref: Some(AutoPtr), // [i32] -> &[i32]
///     unsize: Some([i32]),    // [i32; 4] -> [i32]
/// }
/// ```
///
/// Note that for a struct, the 'deep' unsizing of the struct is not recorded.
/// E.g., `struct Foo<T> { x: T }` we can coerce &Foo<[i32; 4]> to &Foo<[i32]>
/// The autoderef and -ref are the same as in the above example, but the type
/// stored in `unsize` is `Foo<[i32]>`, we don't store any further detail about
/// the underlying conversions from `[i32; 4]` to `[i32]`.
///
/// 3. Coercing a `Box<T>` to `Box<Trait>` is an interesting special case.  In
/// that case, we have the pointer we need coming in, so there are no
/// autoderefs, and no autoref. Instead we just do the `Unsize` transformation.
/// At some point, of course, `Box` should move out of the compiler, in which
/// case this is analogous to transformating a struct. E.g., Box<[i32; 4]> ->
/// Box<[i32]> is represented by:
///
/// ```
/// AutoDerefRef {
///     autoderefs: 0,
///     autoref: None,
///     unsize: Some(Box<[i32]>),
/// }
/// ```
#[derive(Copy, Clone)]
pub struct AutoDerefRef<'tcx> {
    /// Step 1. Apply a number of dereferences, producing an lvalue.
    pub autoderefs: usize,

    /// Step 2. Optionally produce a pointer/reference from the value.
    pub autoref: Option<AutoRef<'tcx>>,

    /// Step 3. Unsize a pointer/reference value, e.g. `&[T; n]` to
    /// `&[T]`. The stored type is the target pointer type. Note that
    /// the source could be a thin or fat pointer.
    pub unsize: Option<Ty<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AutoRef<'tcx> {
    /// Convert from T to &T.
    AutoPtr(&'tcx Region, ast::Mutability),

    /// Convert from T to *T.
    /// Value to thin pointer.
    AutoUnsafe(ast::Mutability),
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug)]
pub enum CustomCoerceUnsized {
    /// Records the index of the field being coerced.
    Struct(usize)
}

#[derive(Clone, Copy, Debug)]
pub struct MethodCallee<'tcx> {
    /// Impl method ID, for inherent methods, or trait method ID, otherwise.
    pub def_id: DefId,
    pub ty: Ty<'tcx>,
    pub substs: &'tcx subst::Substs<'tcx>
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
    pub autoderef: u32
}

impl MethodCall {
    pub fn expr(id: ast::NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            autoderef: 0
        }
    }

    pub fn autoderef(expr_id: ast::NodeId, autoderef: u32) -> MethodCall {
        MethodCall {
            expr_id: expr_id,
            autoderef: 1 + autoderef
        }
    }
}

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
pub type MethodMap<'tcx> = FnvHashMap<MethodCall, MethodCallee<'tcx>>;

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: CrateNum,
    pub pos: usize,
    pub len: usize
}

/// A restriction that certain types must be the same size. The use of
/// `transmute` gives rise to these restrictions. These generally
/// cannot be checked until trans; therefore, each call to `transmute`
/// will push one or more such restriction into the
/// `transmute_restrictions` vector during `intrinsicck`. They are
/// then checked during `trans` by the fn `check_intrinsics`.
#[derive(Copy, Clone)]
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
    // internings
    type_: TypedArena<TyS<'tcx>>,
    substs: TypedArena<Substs<'tcx>>,
    bare_fn: TypedArena<BareFnTy<'tcx>>,
    region: TypedArena<Region>,
    stability: TypedArena<attr::Stability>,

    // references
    trait_defs: TypedArena<TraitDef<'tcx>>,
    adt_defs: TypedArena<AdtDefData<'tcx, 'tcx>>,
}

impl<'tcx> CtxtArenas<'tcx> {
    pub fn new() -> CtxtArenas<'tcx> {
        CtxtArenas {
            type_: TypedArena::new(),
            substs: TypedArena::new(),
            bare_fn: TypedArena::new(),
            region: TypedArena::new(),
            stability: TypedArena::new(),

            trait_defs: TypedArena::new(),
            adt_defs: TypedArena::new()
        }
    }
}

pub struct CommonTypes<'tcx> {
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub isize: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub err: Ty<'tcx>,
}

pub struct Tables<'tcx> {
    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    pub node_types: NodeMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    pub item_substs: NodeMap<ItemSubsts<'tcx>>,

    pub adjustments: NodeMap<ty::AutoAdjustment<'tcx>>,

    pub method_map: MethodMap<'tcx>,

    /// Borrows
    pub upvar_capture_map: UpvarCaptureMap,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_tys: DefIdMap<ClosureTy<'tcx>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kinds: DefIdMap<ClosureKind>,
}

impl<'tcx> Tables<'tcx> {
    pub fn empty() -> Tables<'tcx> {
        Tables {
            node_types: FnvHashMap(),
            item_substs: NodeMap(),
            adjustments: NodeMap(),
            method_map: FnvHashMap(),
            upvar_capture_map: FnvHashMap(),
            closure_tys: DefIdMap(),
            closure_kinds: DefIdMap(),
        }
    }
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
    stability_interner: RefCell<FnvHashMap<&'tcx attr::Stability, &'tcx attr::Stability>>,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    pub sess: Session,
    pub def_map: DefMap,

    pub named_region_map: resolve_lifetime::NamedRegionMap,

    pub region_maps: RegionMaps,

    // For each fn declared in the local crate, type check stores the
    // free-region relationships that were deduced from its where
    // clauses and parameter types. These are then read-again by
    // borrowck. (They are not used during trans, and hence are not
    // serialized or needed for cross-crate fns.)
    free_region_maps: RefCell<NodeMap<FreeRegionMap>>,
    // FIXME: jroesch make this a refcell

    pub tables: RefCell<Tables<'tcx>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub impl_or_trait_items: RefCell<DefIdMap<ImplOrTraitItem<'tcx>>>,

    /// Maps from a trait def-id to a list of the def-ids of its trait items
    pub trait_item_def_ids: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItemId>>>>,

    /// A cache for the trait_items() routine
    pub trait_items_cache: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItem<'tcx>>>>>,

    pub impl_trait_refs: RefCell<DefIdMap<Option<TraitRef<'tcx>>>>,
    pub trait_defs: RefCell<DefIdMap<&'tcx TraitDef<'tcx>>>,
    pub adt_defs: RefCell<DefIdMap<AdtDefMaster<'tcx>>>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated predicates.
    pub predicates: RefCell<DefIdMap<GenericPredicates<'tcx>>>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    pub super_predicates: RefCell<DefIdMap<GenericPredicates<'tcx>>>,

    pub map: ast_map::Map<'tcx>,
    pub freevars: RefCell<FreevarMap>,
    pub tcache: RefCell<DefIdMap<TypeScheme<'tcx>>>,
    pub rcache: RefCell<FnvHashMap<CReaderCacheKey, Ty<'tcx>>>,
    pub tc_cache: RefCell<FnvHashMap<Ty<'tcx>, TypeContents>>,
    pub ast_ty_to_ty_cache: RefCell<NodeMap<Ty<'tcx>>>,
    pub ty_param_defs: RefCell<NodeMap<TypeParameterDef<'tcx>>>,
    pub normalized_cache: RefCell<FnvHashMap<Ty<'tcx>, Ty<'tcx>>>,
    pub lang_items: middle::lang_items::LanguageItems,
    /// A mapping of fake provided method def_ids to the default implementation
    pub provided_method_sources: RefCell<DefIdMap<DefId>>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub item_variance_map: RefCell<DefIdMap<Rc<ItemVariances>>>,

    /// True if the variance has been computed yet; false otherwise.
    pub variance_computed: Cell<bool>,

    /// A method will be in this list if and only if it is a destructor.
    pub destructors: RefCell<DefIdSet>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: RefCell<DefIdMap<Rc<Vec<DefId>>>>,

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
    /// The set of external primitive types whose implementations have been read.
    /// FIXME(arielb1): why is this separate from populated_external_types?
    pub populated_external_primitive_impls: RefCell<DefIdSet>,

    /// These caches are used by const_eval when decoding external constants.
    pub extern_const_statics: RefCell<DefIdMap<ast::NodeId>>,
    pub extern_const_variants: RefCell<DefIdMap<ast::NodeId>>,
    pub extern_const_fns: RefCell<DefIdMap<ast::NodeId>>,

    pub node_lint_levels: RefCell<FnvHashMap<(ast::NodeId, lint::LintId),
                                              lint::LevelSource>>,

    /// The types that must be asserted to be the same size for `transmute`
    /// to be valid. We gather up these restrictions in the intrinsicck pass
    /// and check them in trans.
    pub transmute_restrictions: RefCell<Vec<TransmuteRestriction<'tcx>>>,

    /// Maps any item's def-id to its stability index.
    pub stability: RefCell<stability::Index<'tcx>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// A set of predicates that have been fulfilled *somewhere*.
    /// This is used to avoid duplicate work. Predicates are only
    /// added to this set when they mention only "global" names
    /// (i.e., no type or lifetime parameters).
    pub fulfilled_predicates: RefCell<traits::FulfilledPredicates<'tcx>>,

    /// Caches the representation hints for struct definitions.
    pub repr_hint_cache: RefCell<DefIdMap<Rc<Vec<attr::ReprAttr>>>>,

    /// Maps Expr NodeId's to their constant qualification.
    pub const_qualif_map: RefCell<NodeMap<check_const::ConstQualif>>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    pub custom_coerce_unsized_kinds: RefCell<DefIdMap<CustomCoerceUnsized>>,

    /// Maps a cast expression to its kind. This is keyed on the
    /// *from* expression of the cast, not the cast itself.
    pub cast_kinds: RefCell<NodeMap<cast::CastKind>>,

    /// Maps Fn items to a collection of fragment infos.
    ///
    /// The main goal is to identify data (each of which may be moved
    /// or assigned) whose subparts are not moved nor assigned
    /// (i.e. their state is *unfragmented*) and corresponding ast
    /// nodes where the path to that data is moved or assigned.
    ///
    /// In the long term, unfragmented values will have their
    /// destructor entirely driven by a single stack-local drop-flag,
    /// and their parents, the collections of the unfragmented values
    /// (or more simply, "fragmented values"), are mapped to the
    /// corresponding collections of stack-local drop-flags.
    ///
    /// (However, in the short term that is not the case; e.g. some
    /// unfragmented paths still need to be zeroed, namely when they
    /// reference parent data from an outer scope that was not
    /// entirely moved, and therefore that needs to be zeroed so that
    /// we do not get double-drop when we hit the end of the parent
    /// scope.)
    ///
    /// Also: currently the table solely holds keys for node-ids of
    /// unfragmented values (see `FragmentInfo` enum definition), but
    /// longer-term we will need to also store mappings from
    /// fragmented data to the set of unfragmented pieces that
    /// constitute it.
    pub fragment_infos: RefCell<DefIdMap<Vec<FragmentInfo>>>,
}

/// Describes the fragment-state associated with a NodeId.
///
/// Currently only unfragmented paths have entries in the table,
/// but longer-term this enum is expected to expand to also
/// include data for fragmented paths.
#[derive(Copy, Clone, Debug)]
pub enum FragmentInfo {
    Moved { var: NodeId, move_expr: NodeId },
    Assigned { var: NodeId, assign_expr: NodeId, assignee_id: NodeId },
}

impl<'tcx> ctxt<'tcx> {
    pub fn node_types(&self) -> Ref<NodeMap<Ty<'tcx>>> {
        fn projection<'a, 'tcx>(tables: &'a Tables<'tcx>) ->  &'a NodeMap<Ty<'tcx>> {
            &tables.node_types
        }

        Ref::map(self.tables.borrow(), projection)
    }

    pub fn node_type_insert(&self, id: NodeId, ty: Ty<'tcx>) {
        self.tables.borrow_mut().node_types.insert(id, ty);
    }

    pub fn intern_trait_def(&self, def: TraitDef<'tcx>) -> &'tcx TraitDef<'tcx> {
        let did = def.trait_ref.def_id;
        let interned = self.arenas.trait_defs.alloc(def);
        self.trait_defs.borrow_mut().insert(did, interned);
        interned
    }

    pub fn intern_adt_def(&self,
                          did: DefId,
                          kind: AdtKind,
                          variants: Vec<VariantDefData<'tcx, 'tcx>>)
                          -> AdtDefMaster<'tcx> {
        let def = AdtDefData::new(self, did, kind, variants);
        let interned = self.arenas.adt_defs.alloc(def);
        // this will need a transmute when reverse-variance is removed
        self.adt_defs.borrow_mut().insert(did, interned);
        interned
    }

    pub fn intern_stability(&self, stab: attr::Stability) -> &'tcx attr::Stability {
        if let Some(st) = self.stability_interner.borrow().get(&stab) {
            return st;
        }

        let interned = self.arenas.stability.alloc(stab);
        self.stability_interner.borrow_mut().insert(interned, interned);
        interned
    }

    pub fn store_free_region_map(&self, id: NodeId, map: FreeRegionMap) {
        self.free_region_maps.borrow_mut()
                             .insert(id, map);
    }

    pub fn free_region_map(&self, id: NodeId) -> FreeRegionMap {
        self.free_region_maps.borrow()[&id].clone()
    }

    pub fn lift<T: ?Sized + Lift<'tcx>>(&self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }
}

/// A trait implemented for all X<'a> types which can be safely and
/// efficiently converted to X<'tcx> as long as they are part of the
/// provided ty::ctxt<'tcx>.
/// This can be done, for example, for Ty<'tcx> or &'tcx Substs<'tcx>
/// by looking them up in their respective interners.
/// None is returned if the value or one of the components is not part
/// of the provided context.
/// For Ty, None can be returned if either the type interner doesn't
/// contain the TypeVariants key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g. `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx> {
    type Lifted;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted>;
}

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for [T] {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted> {
        let mut result = Vec::with_capacity(self.len());
        for x in self {
            if let Some(value) = tcx.lift(x) {
                result.push(value);
            } else {
                return None;
            }
        }
        Some(result)
    }
}

impl<'tcx> Lift<'tcx> for Region {
    type Lifted = Self;
    fn lift_to_tcx(&self, _: &ctxt<'tcx>) -> Option<Region> {
        Some(*self)
    }
}

impl<'a, 'tcx> Lift<'tcx> for Ty<'a> {
    type Lifted = Ty<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Ty<'tcx>> {
        if let Some(&ty) = tcx.interner.borrow().get(&self.sty) {
            if *self as *const _ == ty as *const _ {
                return Some(ty);
            }
        }
        None
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Substs<'a> {
    type Lifted = &'tcx Substs<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<&'tcx Substs<'tcx>> {
        if let Some(&substs) = tcx.substs_interner.borrow().get(*self) {
            if *self as *const _ == substs as *const _ {
                return Some(substs);
            }
        }
        None
    }
}

impl<'a, 'tcx> Lift<'tcx> for TraitRef<'a> {
    type Lifted = TraitRef<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<TraitRef<'tcx>> {
        tcx.lift(&self.substs).map(|substs| TraitRef {
            def_id: self.def_id,
            substs: substs
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for TraitPredicate<'a> {
    type Lifted = TraitPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| TraitPredicate {
            trait_ref: trait_ref
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for EquatePredicate<'a> {
    type Lifted = EquatePredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<EquatePredicate<'tcx>> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| EquatePredicate(a, b))
    }
}

impl<'tcx, A: Copy+Lift<'tcx>, B: Copy+Lift<'tcx>> Lift<'tcx> for OutlivesPredicate<A, B> {
    type Lifted = OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ProjectionPredicate<'a> {
    type Lifted = ProjectionPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<ProjectionPredicate<'tcx>> {
        tcx.lift(&(self.projection_ty.trait_ref, self.ty)).map(|(trait_ref, ty)| {
            ProjectionPredicate {
                projection_ty: ProjectionTy {
                    trait_ref: trait_ref,
                    item_name: self.projection_ty.item_name
                },
                ty: ty
            }
        })
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Binder<T> {
    type Lifted = Binder<T::Lifted>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).map(|x| Binder(x))
    }
}

pub mod tls {
    use middle::ty;
    use session::Session;

    use std::fmt;
    use syntax::codemap;

    /// Marker type used for the scoped TLS slot.
    /// The type context cannot be used directly because the scoped TLS
    /// in libstd doesn't allow types generic over lifetimes.
    struct ThreadLocalTyCx;

    scoped_thread_local!(static TLS_TCX: ThreadLocalTyCx);

    fn span_debug(span: codemap::Span, f: &mut fmt::Formatter) -> fmt::Result {
        with(|tcx| {
            write!(f, "{}", tcx.sess.codemap().span_to_string(span))
        })
    }

    pub fn enter<'tcx, F: FnOnce(&ty::ctxt<'tcx>) -> R, R>(tcx: ty::ctxt<'tcx>, f: F)
                                                           -> (Session, R) {
        let result = codemap::SPAN_DEBUG.with(|span_dbg| {
            let original_span_debug = span_dbg.get();
            span_dbg.set(span_debug);
            let tls_ptr = &tcx as *const _ as *const ThreadLocalTyCx;
            let result = TLS_TCX.set(unsafe { &*tls_ptr }, || f(&tcx));
            span_dbg.set(original_span_debug);
            result
        });
        (tcx.sess, result)
    }

    pub fn with<F: FnOnce(&ty::ctxt) -> R, R>(f: F) -> R {
        TLS_TCX.with(|tcx| f(unsafe { &*(tcx as *const _ as *const ty::ctxt) }))
    }

    pub fn with_opt<F: FnOnce(Option<&ty::ctxt>) -> R, R>(f: F) -> R {
        if TLS_TCX.is_set() {
            with(|v| f(Some(v)))
        } else {
            f(None)
        }
    }
}

// Flags that we track on types. These flags are propagated upwards
// through the type during type construction, so that we can quickly
// check whether the type has various kinds of types in it without
// recursing over the type itself.
bitflags! {
    flags TypeFlags: u32 {
        const HAS_PARAMS         = 1 << 0,
        const HAS_SELF           = 1 << 1,
        const HAS_TY_INFER       = 1 << 2,
        const HAS_RE_INFER       = 1 << 3,
        const HAS_RE_EARLY_BOUND = 1 << 4,
        const HAS_FREE_REGIONS   = 1 << 5,
        const HAS_TY_ERR         = 1 << 6,
        const HAS_PROJECTION     = 1 << 7,
        const HAS_TY_CLOSURE     = 1 << 8,

        // true if there are "names" of types and regions and so forth
        // that are local to a particular fn
        const HAS_LOCAL_NAMES   = 1 << 9,

        const NEEDS_SUBST        = TypeFlags::HAS_PARAMS.bits |
                                   TypeFlags::HAS_SELF.bits |
                                   TypeFlags::HAS_RE_EARLY_BOUND.bits,

        // Flags representing the nominal content of a type,
        // computed by FlagsComputation. If you add a new nominal
        // flag, it should be added here too.
        const NOMINAL_FLAGS     = TypeFlags::HAS_PARAMS.bits |
                                  TypeFlags::HAS_SELF.bits |
                                  TypeFlags::HAS_TY_INFER.bits |
                                  TypeFlags::HAS_RE_INFER.bits |
                                  TypeFlags::HAS_RE_EARLY_BOUND.bits |
                                  TypeFlags::HAS_FREE_REGIONS.bits |
                                  TypeFlags::HAS_TY_ERR.bits |
                                  TypeFlags::HAS_PROJECTION.bits |
                                  TypeFlags::HAS_TY_CLOSURE.bits |
                                  TypeFlags::HAS_LOCAL_NAMES.bits,

        // Caches for type_is_sized, type_moves_by_default
        const SIZEDNESS_CACHED  = 1 << 16,
        const IS_SIZED          = 1 << 17,
        const MOVENESS_CACHED   = 1 << 18,
        const MOVES_BY_DEFAULT  = 1 << 19,
    }
}

macro_rules! sty_debug_print {
    ($ctxt: expr, $($variant: ident),*) => {{
        // curious inner module to allow variant names to be used as
        // variable names.
        #[allow(non_snake_case)]
        mod inner {
            use middle::ty;
            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                region_infer: usize,
                ty_infer: usize,
                both_infer: usize,
            }

            pub fn go(tcx: &ty::ctxt) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*


                for (_, t) in tcx.interner.borrow().iter() {
                    let variant = match t.sty {
                        ty::TyBool | ty::TyChar | ty::TyInt(..) | ty::TyUint(..) |
                            ty::TyFloat(..) | ty::TyStr => continue,
                        ty::TyError => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let region = t.flags.get().intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.get().intersects(ty::TypeFlags::HAS_TY_INFER);

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
            TyEnum, TyBox, TyArray, TySlice, TyRawPtr, TyRef, TyBareFn, TyTrait,
            TyStruct, TyClosure, TyTuple, TyParam, TyInfer, TyProjection);

        println!("Substs interner: #{}", self.substs_interner.borrow().len());
        println!("BareFnTy interner: #{}", self.bare_fn_interner.borrow().len());
        println!("Region interner: #{}", self.region_interner.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
    }
}

pub struct TyS<'tcx> {
    pub sty: TypeVariants<'tcx>,
    pub flags: Cell<TypeFlags>,

    // the maximal depth of any bound regions appearing in this type.
    region_depth: u32,
}

impl fmt::Debug for TypeFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.bits)
    }
}

impl<'tcx> PartialEq for TyS<'tcx> {
    #[inline]
    fn eq(&self, other: &TyS<'tcx>) -> bool {
        // (self as *const _) == (other as *const _)
        (self as *const TyS<'tcx>) == (other as *const TyS<'tcx>)
    }
}
impl<'tcx> Eq for TyS<'tcx> {}

impl<'tcx> Hash for TyS<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const TyS).hash(s)
    }
}

pub type Ty<'tcx> = &'tcx TyS<'tcx>;

/// An IVar that contains a Ty. 'lt is a (reverse-variant) upper bound
/// on the lifetime of the IVar. This is required because of variance
/// problems: the IVar needs to be variant with respect to 'tcx (so
/// it can be referred to from Ty) but can only be modified if its
/// lifetime is exactly 'tcx.
///
/// Safety invariants:
///     (A) self.0, if fulfilled, is a valid Ty<'tcx>
///     (B) no aliases to this value with a 'tcx longer than this
///         value's 'lt exist
///
/// NonZero is used rather than Unique because Unique isn't Copy.
pub struct TyIVar<'tcx, 'lt: 'tcx>(ivar::Ivar<NonZero<*const TyS<'static>>>,
                                   PhantomData<fn(TyS<'lt>)->TyS<'tcx>>);

impl<'tcx, 'lt> TyIVar<'tcx, 'lt> {
    #[inline]
    pub fn new() -> Self {
        // Invariant (A) satisfied because the IVar is unfulfilled
        // Invariant (B) because 'lt : 'tcx
        TyIVar(ivar::Ivar::new(), PhantomData)
    }

    #[inline]
    pub fn get(&self) -> Option<Ty<'tcx>> {
        match self.0.get() {
            None => None,
            // valid because of invariant (A)
            Some(v) => Some(unsafe { &*(*v as *const TyS<'tcx>) })
        }
    }
    #[inline]
    pub fn unwrap(&self) -> Ty<'tcx> {
        self.get().unwrap()
    }

    pub fn fulfill(&self, value: Ty<'lt>) {
        // Invariant (A) is fulfilled, because by (B), every alias
        // of this has a 'tcx longer than 'lt.
        let value: *const TyS<'lt> = value;
        // FIXME(27214): unneeded [as *const ()]
        let value = value as *const () as *const TyS<'static>;
        self.0.fulfill(unsafe { NonZero::new(value) })
    }
}

impl<'tcx, 'lt> fmt::Debug for TyIVar<'tcx, 'lt> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(val) => write!(f, "TyIVar({:?})", val),
            None => f.write_str("TyIVar(<unfulfilled>)")
        }
    }
}

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

impl<'tcx> Hash for InternedTy<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.ty.sty.hash(s)
    }
}

impl<'tcx> Borrow<TypeVariants<'tcx>> for InternedTy<'tcx> {
    fn borrow<'a>(&'a self) -> &'a TypeVariants<'tcx> {
        &self.ty.sty
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BareFnTy<'tcx> {
    pub unsafety: ast::Unsafety,
    pub abi: abi::Abi,
    pub sig: PolyFnSig<'tcx>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
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

    pub fn unwrap_or(self, def: Ty<'tcx>) -> Ty<'tcx> {
        match self {
            ty::FnConverging(t) => t,
            ty::FnDiverging => def
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
        self.map_bound_ref(|fn_sig| fn_sig.inputs.clone())
    }
    pub fn input(&self, index: usize) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs[index])
    }
    pub fn output(&self) -> ty::Binder<FnOutput<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.output.clone())
    }
    pub fn variadic(&self) -> bool {
        self.skip_binder().variadic
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamTy {
    pub space: subst::ParamSpace,
    pub idx: u32,
    pub name: ast::Name,
}

/// A [De Bruijn index][dbi] is a standard means of representing
/// regions (and perhaps later types) in a higher-ranked setting. In
/// particular, imagine a type like this:
///
///     for<'a> fn(for<'b> fn(&'b isize, &'a isize), &'a char)
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
/// Let's start with the reference type `&'b isize` that is the first
/// argument to the inner function. This region `'b` is assigned a De
/// Bruijn index of 1, meaning "the innermost binder" (in this case, a
/// fn). The region `'a` that appears in the second argument type (`&'a
/// isize`) would then be assigned a De Bruijn index of 2, meaning "the
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

/// Representation of regions.
///
/// Unlike types, most region variants are "fictitious", not concrete,
/// regions. Among these, `ReStatic`, `ReEmpty` and `ReScope` are the only
/// ones representing concrete regions.
///
/// ## Bound Regions
///
/// These are regions that are stored behind a binder and must be substituted
/// with some concrete region before being used. There are 2 kind of
/// bound regions: early-bound, which are bound in a TypeScheme/TraitDef,
/// and are substituted by a Substs,  and late-bound, which are part of
/// higher-ranked types (e.g. `for<'a> fn(&'a ())`) and are substituted by
/// the likes of `liberate_late_bound_regions`. The distinction exists
/// because higher-ranked lifetimes aren't supported in all places. See [1][2].
///
/// Unlike TyParam-s, bound regions are not supposed to exist "in the wild"
/// outside their binder, e.g. in types passed to type inference, and
/// should first be substituted (by skolemized regions, free regions,
/// or region variables).
///
/// ## Skolemized and Free Regions
///
/// One often wants to work with bound regions without knowing their precise
/// identity. For example, when checking a function, the lifetime of a borrow
/// can end up being assigned to some region parameter. In these cases,
/// it must be ensured that bounds on the region can't be accidentally
/// assumed without being checked.
///
/// The process of doing that is called "skolemization". The bound regions
/// are replaced by skolemized markers, which don't satisfy any relation
/// not explicity provided.
///
/// There are 2 kinds of skolemized regions in rustc: `ReFree` and
/// `ReSkolemized`. When checking an item's body, `ReFree` is supposed
/// to be used. These also support explicit bounds: both the internally-stored
/// *scope*, which the region is assumed to outlive, as well as other
/// relations stored in the `FreeRegionMap`. Note that these relations
/// aren't checked when you `make_subregion` (or `mk_eqty`), only by
/// `resolve_regions_and_report_errors`.
///
/// When working with higher-ranked types, some region relations aren't
/// yet known, so you can't just call `resolve_regions_and_report_errors`.
/// `ReSkolemized` is designed for this purpose. In these contexts,
/// there's also the risk that some inference variable laying around will
/// get unified with your skolemized region: if you want to check whether
/// `for<'a> Foo<'_>: 'a`, and you substitute your bound region `'a`
/// with a skolemized region `'%a`, the variable `'_` would just be
/// instantiated to the skolemized region `'%a`, which is wrong because
/// the inference variable is supposed to satisfy the relation
/// *for every value of the skolemized region*. To ensure that doesn't
/// happen, you can use `leak_check`. This is more clearly explained
/// by infer/higher_ranked/README.md.
///
/// [1] http://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
/// [2] http://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/
#[derive(Clone, PartialEq, Eq, Hash, Copy)]
pub enum Region {
    // Region bound in a type or fn declaration which will be
    // substituted 'early' -- that is, at the same time when type
    // parameters are substituted.
    ReEarlyBound(EarlyBoundRegion),

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
    ReVar(RegionVid),

    /// A skolemized region - basically the higher-ranked version of ReFree.
    /// Should not exist after typeck.
    ReSkolemized(SkolemizedRegionVid, BoundRegion),

    /// Empty lifetime is for data that is never accessed.
    /// Bottom in the region lattice. We treat ReEmpty somewhat
    /// specially; at least right now, we do not generate instances of
    /// it during the GLB computations, but rather
    /// generate an error instead. This is to improve error messages.
    /// The only way to get an instance of ReEmpty is to have a region
    /// variable with no constraints.
    ReEmpty,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub struct EarlyBoundRegion {
    pub param_id: ast::NodeId,
    pub space: subst::ParamSpace,
    pub index: u32,
    pub name: ast::Name,
}

/// Upvars do not get their own node-id. Instead, we use the pair of
/// the original var id (that is, the root variable that is referenced
/// by the upvar) and the id of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    ///    let x: &mut isize = ...;
    ///    let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
    ///    let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
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
#[derive(PartialEq, Clone, Debug, Copy)]
pub enum UpvarCapture {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by reference.
    ByRef(UpvarBorrow),
}

#[derive(PartialEq, Clone, Copy)]
pub struct UpvarBorrow {
    /// The kind of borrow: by-ref upvars have access to shared
    /// immutable borrows, which are not part of the normal language
    /// syntax.
    pub kind: BorrowKind,

    /// Region of the resulting reference.
    pub region: ty::Region,
}

pub type UpvarCaptureMap = FnvHashMap<UpvarId, UpvarCapture>;

#[derive(Copy, Clone)]
pub struct ClosureUpvar<'tcx> {
    pub def: def::Def,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

impl Region {
    pub fn is_bound(&self) -> bool {
        match *self {
            ty::ReEarlyBound(..) => true,
            ty::ReLateBound(..) => true,
            _ => false
        }
    }

    pub fn needs_infer(&self) -> bool {
        match *self {
            ty::ReVar(..) | ty::ReSkolemized(..) => true,
            _ => false
        }
    }

    pub fn escapes_depth(&self, depth: u32) -> bool {
        match *self {
            ty::ReLateBound(debruijn, _) => debruijn.depth > depth,
            _ => false,
        }
    }

    /// Returns the depth of `self` from the (1-based) binding level `depth`
    pub fn from_depth(&self, depth: u32) -> Region {
        match *self {
            ty::ReLateBound(debruijn, r) => ty::ReLateBound(DebruijnIndex {
                depth: debruijn.depth - (depth - 1)
            }, r),
            r => r
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Copy)]
/// A "free" region `fr` can be interpreted as "some region
/// at least as big as the scope `fr.scope`".
pub struct FreeRegion {
    pub scope: region::CodeExtent,
    pub bound_region: BoundRegion
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Copy, Debug)]
pub enum BoundRegion {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(u32),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The def-id is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(DefId, ast::Name),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(u32),

    // Anonymous region for the implicit env pointer parameter
    // to a closure
    BrEnv
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in libsyntax/ast.rs as well.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeVariants<'tcx> {
    /// The primitive boolean type. Written as `bool`.
    TyBool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point).  Written as `char`.
    TyChar,

    /// A primitive signed integer type. For example, `i32`.
    TyInt(ast::IntTy),

    /// A primitive unsigned integer type. For example, `u32`.
    TyUint(ast::UintTy),

    /// A primitive floating-point type. For example, `f64`.
    TyFloat(ast::FloatTy),

    /// An enumerated type, defined with `enum`.
    ///
    /// Substs here, possibly against intuition, *may* contain `TyParam`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `TyEnum` corresponds to an enum
    /// definition and not a concrete use of it. To get the correct `TyEnum`
    /// from the tcx, use the `NodeId` from the `ast::Ty` and look it up in
    /// the `ast_ty_to_ty_cache`. This is probably true for `TyStruct` as
    /// well.
    TyEnum(AdtDef<'tcx>, &'tcx Substs<'tcx>),

    /// A structure type, defined with `struct`.
    ///
    /// See warning about substitutions for enumerated types.
    TyStruct(AdtDef<'tcx>, &'tcx Substs<'tcx>),

    /// `Box<T>`; this is nominally a struct in the documentation, but is
    /// special-cased internally. For example, it is possible to implicitly
    /// move the contents of a box out of that box, and methods of any type
    /// can have type `Box<Self>`.
    TyBox(Ty<'tcx>),

    /// The pointee of a string slice. Written as `str`.
    TyStr,

    /// An array with the given length. Written as `[T; n]`.
    TyArray(Ty<'tcx>, usize),

    /// The pointee of an array slice.  Written as `[T]`.
    TySlice(Ty<'tcx>),

    /// A raw pointer. Written as `*mut T` or `*const T`
    TyRawPtr(TypeAndMut<'tcx>),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&a mut T` or `&'a T`.
    TyRef(&'tcx Region, TypeAndMut<'tcx>),

    /// If the def-id is Some(_), then this is the type of a specific
    /// fn item. Otherwise, if None(_), it a fn pointer type.
    ///
    /// FIXME: Conflating function pointers and the type of a
    /// function is probably a terrible idea; a function pointer is a
    /// value with a specific type, but a function can be polymorphic
    /// or dynamically dispatched.
    TyBareFn(Option<DefId>, &'tcx BareFnTy<'tcx>),

    /// A trait, defined with `trait`.
    TyTrait(Box<TraitTy<'tcx>>),

    /// The anonymous type of a closure. Used to represent the type of
    /// `|a| a`.
    TyClosure(DefId, Box<ClosureSubsts<'tcx>>),

    /// A tuple type.  For example, `(i32, bool)`.
    TyTuple(Vec<Ty<'tcx>>),

    /// The projection of an associated type.  For example,
    /// `<T as Trait<..>>::N`.
    TyProjection(ProjectionTy<'tcx>),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    TyParam(ParamTy),

    /// A type variable used during type-checking.
    TyInfer(InferTy),

    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    TyError,
}

/// A closure can be modeled as a struct that looks like:
///
///     struct Closure<'l0...'li, T0...Tj, U0...Uk> {
///         upvar0: U0,
///         ...
///         upvark: Uk
///     }
///
/// where 'l0...'li and T0...Tj are the lifetime and type parameters
/// in scope on the function that defined the closure, and U0...Uk are
/// type parameters representing the types of its upvars (borrowed, if
/// appropriate).
///
/// So, for example, given this function:
///
///     fn foo<'a, T>(data: &'a mut T) {
///          do(|| data.count += 1)
///     }
///
/// the type of the closure would be something like:
///
///     struct Closure<'a, T, U0> {
///         data: U0
///     }
///
/// Note that the type of the upvar is not specified in the struct.
/// You may wonder how the impl would then be able to use the upvar,
/// if it doesn't know it's type? The answer is that the impl is
/// (conceptually) not fully generic over Closure but rather tied to
/// instances with the expected upvar types:
///
///     impl<'b, 'a, T> FnMut() for Closure<'a, T, &'b mut &'a mut T> {
///         ...
///     }
///
/// You can see that the *impl* fully specified the type of the upvar
/// and thus knows full well that `data` has type `&'b mut &'a mut T`.
/// (Here, I am assuming that `data` is mut-borrowed.)
///
/// Now, the last question you may ask is: Why include the upvar types
/// as extra type parameters? The reason for this design is that the
/// upvar types can reference lifetimes that are internal to the
/// creating function. In my example above, for example, the lifetime
/// `'b` represents the extent of the closure itself; this is some
/// subset of `foo`, probably just the extent of the call to the to
/// `do()`. If we just had the lifetime/type parameters from the
/// enclosing function, we couldn't name this lifetime `'b`. Note that
/// there can also be lifetimes in the types of the upvars themselves,
/// if one of them happens to be a reference to something that the
/// creating fn owns.
///
/// OK, you say, so why not create a more minimal set of parameters
/// that just includes the extra lifetime parameters? The answer is
/// primarily that it would be hard --- we don't know at the time when
/// we create the closure type what the full types of the upvars are,
/// nor do we know which are borrowed and which are not. In this
/// design, we can just supply a fresh type parameter and figure that
/// out later.
///
/// All right, you say, but why include the type parameters from the
/// original function then? The answer is that trans may need them
/// when monomorphizing, and they may not appear in the upvars.  A
/// closure could capture no variables but still make use of some
/// in-scope type parameter with a bound (e.g., if our example above
/// had an extra `U: Default`, and the closure called `U::default()`).
///
/// There is another reason. This design (implicitly) prohibits
/// closures from capturing themselves (except via a trait
/// object). This simplifies closure inference considerably, since it
/// means that when we infer the kind of a closure or its upvars, we
/// don't have to handle cycles where the decisions we make for
/// closure C wind up influencing the decisions we ought to make for
/// closure C (which would then require fixed point iteration to
/// handle). Plus it fixes an ICE. :P
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ClosureSubsts<'tcx> {
    /// Lifetime and type parameters from the enclosing function.
    /// These are separated out because trans wants to pass them around
    /// when monomorphizing.
    pub func_substs: &'tcx Substs<'tcx>,

    /// The types of the upvars. The list parallels the freevars and
    /// `upvar_borrows` lists. These are kept distinct so that we can
    /// easily index into them.
    pub upvar_tys: Vec<Ty<'tcx>>
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraitTy<'tcx> {
    pub principal: ty::PolyTraitRef<'tcx>,
    pub bounds: ExistentialBounds<'tcx>,
}

impl<'tcx> TraitTy<'tcx> {
    pub fn principal_def_id(&self) -> DefId {
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

        ty::Binder(TraitRef {
            def_id: self.principal.0.def_id,
            substs: tcx.mk_substs(self.principal.0.substs.with_self_ty(self_ty)),
        })
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
                let trait_ref = ty::TraitRef::new(in_projection_ty.trait_ref.def_id,
                                              substs);
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
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

pub type PolyTraitRef<'tcx> = Binder<TraitRef<'tcx>>;

impl<'tcx> PolyTraitRef<'tcx> {
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.0.self_ty()
    }

    pub fn def_id(&self) -> DefId {
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
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<TraitRef>`). Note that when we skolemize, instantiate,
/// erase, or otherwise "discharge" these bound regions, we change the
/// type from `Binder<T>` to just `T` (see
/// e.g. `liberate_late_bound_regions`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Binder<T>(pub T);

impl<T> Binder<T> {
    /// Skips the binder and returns the "bound" value. This is a
    /// risky thing to do because it's easy to get confused about
    /// debruijn indices and the like. It is usually better to
    /// discharge the binder using `no_late_bound_regions` or
    /// `replace_late_bound_regions` or something like
    /// that. `skip_binder` is only valid when you are either
    /// extracting data that has nothing to do with bound regions, you
    /// are doing some sort of test that does not involve bound
    /// regions, or you are being very careful about your depth
    /// accounting.
    ///
    /// Some examples where `skip_binder` is reasonable:
    /// - extracting the def-id from a PolyTraitRef;
    /// - comparing the self type of a PolyTraitRef to see if it is equal to
    ///   a type parameter `X`, since the type `X`  does not reference any regions
    pub fn skip_binder(&self) -> &T {
        &self.0
    }

    pub fn as_ref(&self) -> Binder<&T> {
        ty::Binder(&self.0)
    }

    pub fn map_bound_ref<F,U>(&self, f: F) -> Binder<U>
        where F: FnOnce(&T) -> U
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F,U>(self, f: F) -> Binder<U>
        where F: FnOnce(T) -> U
    {
        ty::Binder(f(self.0))
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
}

#[derive(Clone, Copy, Debug)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T
}

// Data structures used in type unification
#[derive(Clone, Debug)]
pub enum TypeError<'tcx> {
    Mismatch,
    UnsafetyMismatch(ExpectedFound<ast::Unsafety>),
    AbiMismatch(ExpectedFound<abi::Abi>),
    Mutability,
    BoxMutability,
    PtrMutability,
    RefMutability,
    VecMutability,
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<usize>),
    TyParamSize(ExpectedFound<usize>),
    ArgCount,
    RegionsDoesNotOutlive(Region, Region),
    RegionsNotSame(Region, Region),
    RegionsNoOverlap(Region, Region),
    RegionsInsufficientlyPolymorphic(BoundRegion, Region),
    RegionsOverlyPolymorphic(BoundRegion, Region),
    Sorts(ExpectedFound<Ty<'tcx>>),
    IntegerAsChar,
    IntMismatch(ExpectedFound<IntVarValue>),
    FloatMismatch(ExpectedFound<ast::FloatTy>),
    Traits(ExpectedFound<DefId>),
    BuiltinBoundsMismatch(ExpectedFound<BuiltinBounds>),
    VariadicMismatch(ExpectedFound<bool>),
    CyclicTy,
    ConvergenceMismatch(ExpectedFound<bool>),
    ProjectionNameMismatched(ExpectedFound<ast::Name>),
    ProjectionBoundsLength(ExpectedFound<usize>),
    TyParamDefaultMismatch(ExpectedFound<type_variable::Default<'tcx>>)
}

/// Bounds suitable for an existentially quantified type parameter
/// such as those that appear in object types or closure types.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct ExistentialBounds<'tcx> {
    pub region_bound: ty::Region,
    pub builtin_bounds: BuiltinBounds,
    pub projection_bounds: Vec<PolyProjectionPredicate<'tcx>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BuiltinBounds(EnumSet<BuiltinBound>);

impl BuiltinBounds {
    pub fn empty() -> BuiltinBounds {
        BuiltinBounds(EnumSet::new())
    }

    pub fn iter(&self) -> enum_set::Iter<BuiltinBound> {
        self.into_iter()
    }

    pub fn to_predicates<'tcx>(&self,
                               tcx: &ty::ctxt<'tcx>,
                               self_ty: Ty<'tcx>) -> Vec<Predicate<'tcx>> {
        self.iter().filter_map(|builtin_bound|
            match traits::trait_ref_for_builtin_bound(tcx, builtin_bound, self_ty) {
                Ok(trait_ref) => Some(trait_ref.to_predicate()),
                Err(ErrorReported) => { None }
            }
        ).collect()
    }
}

impl ops::Deref for BuiltinBounds {
    type Target = EnumSet<BuiltinBound>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl ops::DerefMut for BuiltinBounds {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<'a> IntoIterator for &'a BuiltinBounds {
    type Item = BuiltinBound;
    type IntoIter = enum_set::Iter<BuiltinBound>;
    fn into_iter(self) -> Self::IntoIter {
        (**self).into_iter()
    }
}

#[derive(Clone, RustcEncodable, PartialEq, Eq, RustcDecodable, Hash,
           Debug, Copy)]
#[repr(usize)]
pub enum BuiltinBound {
    Send,
    Sized,
    Copy,
    Sync,
}

impl CLike for BuiltinBound {
    fn to_usize(&self) -> usize {
        *self as usize
    }
    fn from_usize(v: usize) -> BuiltinBound {
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
pub struct SkolemizedRegionVid {
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
    FreshIntTy(u32),
    FreshFloatTy(u32)
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum UnconstrainedNumeric {
    UnconstrainedFloat,
    UnconstrainedInt,
    Neither,
}


impl fmt::Debug for TyVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
            FreshFloatTy(v) => write!(f, "FreshFloatTy({:?})", v)
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
#[derive(Copy, Clone)]
pub enum ObjectLifetimeDefault {
    /// Require an explicit annotation. Occurs when multiple
    /// `T:'a` constraints are found.
    Ambiguous,

    /// Use the base default, typically 'static, but in a fn body it is a fresh variable
    BaseDefault,

    /// Use the given region as the default.
    Specific(Region),
}

#[derive(Clone)]
pub struct TypeParameterDef<'tcx> {
    pub name: ast::Name,
    pub def_id: DefId,
    pub space: subst::ParamSpace,
    pub index: u32,
    pub default_def_id: DefId, // for use in error reporing about defaults
    pub default: Option<Ty<'tcx>>,
    pub object_lifetime_default: ObjectLifetimeDefault,
}

#[derive(Clone, Debug)]
pub struct RegionParameterDef {
    pub name: ast::Name,
    pub def_id: DefId,
    pub space: subst::ParamSpace,
    pub index: u32,
    pub bounds: Vec<ty::Region>,
}

impl RegionParameterDef {
    pub fn to_early_bound_region(&self) -> ty::Region {
        ty::ReEarlyBound(ty::EarlyBoundRegion {
            param_id: self.def_id.node,
            space: self.space,
            index: self.index,
            name: self.name,
        })
    }
    pub fn to_bound_region(&self) -> ty::BoundRegion {
        ty::BoundRegion::BrNamed(self.def_id, self.name)
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
#[derive(Clone)]
pub struct GenericPredicates<'tcx> {
    pub predicates: VecPerParamSpace<Predicate<'tcx>>,
}

impl<'tcx> GenericPredicates<'tcx> {
    pub fn empty() -> GenericPredicates<'tcx> {
        GenericPredicates {
            predicates: VecPerParamSpace::empty(),
        }
    }

    pub fn instantiate(&self, tcx: &ctxt<'tcx>, substs: &Substs<'tcx>)
                       -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates {
            predicates: self.predicates.subst(tcx, substs),
        }
    }

    pub fn instantiate_supertrait(&self,
                                  tcx: &ctxt<'tcx>,
                                  poly_trait_ref: &ty::PolyTraitRef<'tcx>)
                                  -> InstantiatedPredicates<'tcx>
    {
        InstantiatedPredicates {
            predicates: self.predicates.map(|pred| pred.subst_supertrait(tcx, poly_trait_ref))
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
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

    /// no syntax: T WF
    WellFormed(Ty<'tcx>),

    /// trait must be object-safe
    ObjectSafe(DefId),
}

impl<'tcx> Predicate<'tcx> {
    /// Performs a substitution suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// substitution in terms of what happens with bound regions.  See
    /// lengthy comment below for details.
    pub fn subst_supertrait(&self,
                            tcx: &ctxt<'tcx>,
                            trait_ref: &ty::PolyTraitRef<'tcx>)
                            -> ty::Predicate<'tcx>
    {
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

        let substs = &trait_ref.0.substs;
        match *self {
            Predicate::Trait(ty::Binder(ref data)) =>
                Predicate::Trait(ty::Binder(data.subst(tcx, substs))),
            Predicate::Equate(ty::Binder(ref data)) =>
                Predicate::Equate(ty::Binder(data.subst(tcx, substs))),
            Predicate::RegionOutlives(ty::Binder(ref data)) =>
                Predicate::RegionOutlives(ty::Binder(data.subst(tcx, substs))),
            Predicate::TypeOutlives(ty::Binder(ref data)) =>
                Predicate::TypeOutlives(ty::Binder(data.subst(tcx, substs))),
            Predicate::Projection(ty::Binder(ref data)) =>
                Predicate::Projection(ty::Binder(data.subst(tcx, substs))),
            Predicate::WellFormed(data) =>
                Predicate::WellFormed(data.subst(tcx, substs)),
            Predicate::ObjectSafe(trait_def_id) =>
                Predicate::ObjectSafe(trait_def_id),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>
}
pub type PolyTraitPredicate<'tcx> = ty::Binder<TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn def_id(&self) -> DefId {
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
    pub fn def_id(&self) -> DefId {
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
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: ProjectionTy<'tcx>,
    pub ty: Ty<'tcx>,
}

pub type PolyProjectionPredicate<'tcx> = Binder<ProjectionPredicate<'tcx>>;

impl<'tcx> PolyProjectionPredicate<'tcx> {
    pub fn item_name(&self) -> ast::Name {
        self.0.projection_ty.item_name // safe to skip the binder to access a name
    }

    pub fn sort_key(&self) -> (DefId, ast::Name) {
        self.0.projection_ty.sort_key()
    }
}

/// Represents the projection of an associated type. In explicit UFCS
/// form this would be written `<T as Trait<..>>::N`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ProjectionTy<'tcx> {
    /// The trait reference `T as Trait<..>`.
    pub trait_ref: ty::TraitRef<'tcx>,

    /// The name `N` of the associated type.
    pub item_name: ast::Name,
}

impl<'tcx> ProjectionTy<'tcx> {
    pub fn sort_key(&self) -> (DefId, ast::Name) {
        (self.trait_ref.def_id, self.item_name)
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for TraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        assert!(!self.has_escaping_regions());
        ty::Binder(self.clone())
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref.clone())
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

pub trait ToPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx>;
}

impl<'tcx> ToPredicate<'tcx> for TraitRef<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        // we're about to add a binder, so let's check that we don't
        // accidentally capture anything, or else that might be some
        // weird debruijn accounting.
        assert!(!self.has_escaping_regions());

        ty::Predicate::Trait(ty::Binder(ty::TraitPredicate {
            trait_ref: self.clone()
        }))
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTraitRef<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(self.to_poly_trait_predicate())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyEquatePredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::Equate(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyRegionOutlivesPredicate {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::RegionOutlives(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTypeOutlivesPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::TypeOutlives(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
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
                            .chain(Some(data.0.ty))
                            .collect()
            }
            ty::Predicate::WellFormed(data) => {
                vec![data]
            }
            ty::Predicate::ObjectSafe(_trait_def_id) => {
                vec![]
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
            Predicate::WellFormed(p) => p.has_escaping_regions(),
            Predicate::ObjectSafe(_trait_def_id) => false,
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
            Predicate::WellFormed(..) |
            Predicate::ObjectSafe(..) |
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
/// like `Foo<isize,usize>`, then the `InstantiatedPredicates` would be `[[],
/// [usize:Bar<isize>]]`.
#[derive(Clone)]
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
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>) -> TraitRef<'tcx> {
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
    /// into Obligations, and elaborated and normalized.
    pub caller_bounds: Vec<ty::Predicate<'tcx>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Scope that is attached to free regions for this scope. This
    /// is usually the id of the fn body, but for more abstract scopes
    /// like structs we often use the node-id of the struct.
    ///
    /// FIXME(#3696). It would be nice to refactor so that free
    /// regions don't have this implicit scope and instead introduce
    /// relationships in the environment.
    pub free_id: ast::NodeId,
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
            free_id: self.free_id,
        }
    }

    pub fn for_item(cx: &'a ctxt<'tcx>, id: NodeId) -> ParameterEnvironment<'a, 'tcx> {
        match cx.map.find(id) {
            Some(ast_map::NodeImplItem(ref impl_item)) => {
                match impl_item.node {
                    ast::TypeImplItem(_) => {
                        // associated types don't have their own entry (for some reason),
                        // so for now just grab environment for the impl
                        let impl_id = cx.map.get_parent(id);
                        let impl_def_id = DefId::local(impl_id);
                        let scheme = cx.lookup_item_type(impl_def_id);
                        let predicates = cx.lookup_predicates(impl_def_id);
                        cx.construct_parameter_environment(impl_item.span,
                                                           &scheme.generics,
                                                           &predicates,
                                                           id)
                    }
                    ast::ConstImplItem(_, _) => {
                        let def_id = DefId::local(id);
                        let scheme = cx.lookup_item_type(def_id);
                        let predicates = cx.lookup_predicates(def_id);
                        cx.construct_parameter_environment(impl_item.span,
                                                           &scheme.generics,
                                                           &predicates,
                                                           id)
                    }
                    ast::MethodImplItem(_, ref body) => {
                        let method_def_id = DefId::local(id);
                        match cx.impl_or_trait_item(method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                let method_bounds = &method_ty.predicates;
                                cx.construct_parameter_environment(
                                    impl_item.span,
                                    method_generics,
                                    method_bounds,
                                    body.id)
                            }
                            _ => {
                                cx.sess
                                  .bug("ParameterEnvironment::for_item(): \
                                        got non-method item from impl method?!")
                            }
                        }
                    }
                    ast::MacImplItem(_) => cx.sess.bug("unexpanded macro")
                }
            }
            Some(ast_map::NodeTraitItem(trait_item)) => {
                match trait_item.node {
                    ast::TypeTraitItem(..) => {
                        // associated types don't have their own entry (for some reason),
                        // so for now just grab environment for the trait
                        let trait_id = cx.map.get_parent(id);
                        let trait_def_id = DefId::local(trait_id);
                        let trait_def = cx.lookup_trait_def(trait_def_id);
                        let predicates = cx.lookup_predicates(trait_def_id);
                        cx.construct_parameter_environment(trait_item.span,
                                                           &trait_def.generics,
                                                           &predicates,
                                                           id)
                    }
                    ast::ConstTraitItem(..) => {
                        let def_id = DefId::local(id);
                        let scheme = cx.lookup_item_type(def_id);
                        let predicates = cx.lookup_predicates(def_id);
                        cx.construct_parameter_environment(trait_item.span,
                                                           &scheme.generics,
                                                           &predicates,
                                                           id)
                    }
                    ast::MethodTraitItem(_, ref body) => {
                        // for the body-id, use the id of the body
                        // block, unless this is a trait method with
                        // no default, then fallback to the method id.
                        let body_id = body.as_ref().map(|b| b.id).unwrap_or(id);
                        let method_def_id = DefId::local(id);
                        match cx.impl_or_trait_item(method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                let method_bounds = &method_ty.predicates;
                                cx.construct_parameter_environment(
                                    trait_item.span,
                                    method_generics,
                                    method_bounds,
                                    body_id)
                            }
                            _ => {
                                cx.sess
                                  .bug("ParameterEnvironment::for_item(): \
                                        got non-method item from provided \
                                        method?!")
                            }
                        }
                    }
                }
            }
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ast::ItemFn(_, _, _, _, _, ref body) => {
                        // We assume this is a function.
                        let fn_def_id = DefId::local(id);
                        let fn_scheme = cx.lookup_item_type(fn_def_id);
                        let fn_predicates = cx.lookup_predicates(fn_def_id);

                        cx.construct_parameter_environment(item.span,
                                                           &fn_scheme.generics,
                                                           &fn_predicates,
                                                           body.id)
                    }
                    ast::ItemEnum(..) |
                    ast::ItemStruct(..) |
                    ast::ItemImpl(..) |
                    ast::ItemConst(..) |
                    ast::ItemStatic(..) => {
                        let def_id = DefId::local(id);
                        let scheme = cx.lookup_item_type(def_id);
                        let predicates = cx.lookup_predicates(def_id);
                        cx.construct_parameter_environment(item.span,
                                                           &scheme.generics,
                                                           &predicates,
                                                           id)
                    }
                    ast::ItemTrait(..) => {
                        let def_id = DefId::local(id);
                        let trait_def = cx.lookup_trait_def(def_id);
                        let predicates = cx.lookup_predicates(def_id);
                        cx.construct_parameter_environment(item.span,
                                                           &trait_def.generics,
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
                                    cx.map.node_to_string(id)))
            }
        }
    }

    pub fn can_type_implement_copy(&self, self_type: Ty<'tcx>, span: Span)
                                   -> Result<(),CopyImplementationError> {
        let tcx = self.tcx;

        // FIXME: (@jroesch) float this code up
        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(self.clone()), false);

        let adt = match self_type.sty {
            ty::TyStruct(struct_def, substs) => {
                for field in struct_def.all_fields() {
                    let field_ty = field.ty(tcx, substs);
                    if infcx.type_moves_by_default(field_ty, span) {
                        return Err(FieldDoesNotImplementCopy(field.name))
                    }
                }
                struct_def
            }
            ty::TyEnum(enum_def, substs) => {
                for variant in &enum_def.variants {
                    for field in &variant.fields {
                        let field_ty = field.ty(tcx, substs);
                        if infcx.type_moves_by_default(field_ty, span) {
                            return Err(VariantDoesNotImplementCopy(variant.name))
                        }
                    }
                }
                enum_def
            }
            _ => return Err(TypeIsStructural),
        };

        if adt.has_dtor() {
            return Err(TypeHasDestructor)
        }

        Ok(())
    }
}

#[derive(Copy, Clone)]
pub enum CopyImplementationError {
    FieldDoesNotImplementCopy(ast::Name),
    VariantDoesNotImplementCopy(ast::Name),
    TypeIsStructural,
    TypeHasDestructor,
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

bitflags! {
    flags TraitFlags: u32 {
        const NO_TRAIT_FLAGS        = 0,
        const HAS_DEFAULT_IMPL      = 1 << 0,
        const IS_OBJECT_SAFE        = 1 << 1,
        const OBJECT_SAFETY_VALID   = 1 << 2,
        const IMPLS_VALID           = 1 << 3,
    }
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

    pub trait_ref: TraitRef<'tcx>,

    /// A list of the associated types defined in this trait. Useful
    /// for resolving `X::Foo` type markers.
    pub associated_type_names: Vec<ast::Name>,

    // Impls of this trait. To allow for quicker lookup, the impls are indexed
    // by a simplified version of their Self type: impls with a simplifiable
    // Self are stored in nonblanket_impls keyed by it, while all other impls
    // are stored in blanket_impls.

    /// Impls of the trait.
    pub nonblanket_impls: RefCell<
        FnvHashMap<fast_reject::SimplifiedType, Vec<DefId>>
    >,

    /// Blanket impls associated with the trait.
    pub blanket_impls: RefCell<Vec<DefId>>,

    /// Various flags
    pub flags: Cell<TraitFlags>
}

impl<'tcx> TraitDef<'tcx> {
    // returns None if not yet calculated
    pub fn object_safety(&self) -> Option<bool> {
        if self.flags.get().intersects(TraitFlags::OBJECT_SAFETY_VALID) {
            Some(self.flags.get().intersects(TraitFlags::IS_OBJECT_SAFE))
        } else {
            None
        }
    }

    pub fn set_object_safety(&self, is_safe: bool) {
        assert!(self.object_safety().map(|cs| cs == is_safe).unwrap_or(true));
        self.flags.set(
            self.flags.get() | if is_safe {
                TraitFlags::OBJECT_SAFETY_VALID | TraitFlags::IS_OBJECT_SAFE
            } else {
                TraitFlags::OBJECT_SAFETY_VALID
            }
        );
    }

    /// Records a trait-to-implementation mapping.
    pub fn record_impl(&self,
                       tcx: &ctxt<'tcx>,
                       impl_def_id: DefId,
                       impl_trait_ref: TraitRef<'tcx>) {
        debug!("TraitDef::record_impl for {:?}, from {:?}",
               self, impl_trait_ref);

        // We don't want to borrow_mut after we already populated all impls,
        // so check if an impl is present with an immutable borrow first.
        if let Some(sty) = fast_reject::simplify_type(tcx,
                                                      impl_trait_ref.self_ty(), false) {
            if let Some(is) = self.nonblanket_impls.borrow().get(&sty) {
                if is.contains(&impl_def_id) {
                    return // duplicate - skip
                }
            }

            self.nonblanket_impls.borrow_mut().entry(sty).or_insert(vec![]).push(impl_def_id)
        } else {
            if self.blanket_impls.borrow().contains(&impl_def_id) {
                return // duplicate - skip
            }
            self.blanket_impls.borrow_mut().push(impl_def_id)
        }
    }


    pub fn for_each_impl<F: FnMut(DefId)>(&self, tcx: &ctxt<'tcx>, mut f: F)  {
        tcx.populate_implementations_for_trait_if_necessary(self.trait_ref.def_id);

        for &impl_def_id in self.blanket_impls.borrow().iter() {
            f(impl_def_id);
        }

        for v in self.nonblanket_impls.borrow().values() {
            for &impl_def_id in v {
                f(impl_def_id);
            }
        }
    }

    /// Iterate over every impl that could possibly match the
    /// self-type `self_ty`.
    pub fn for_each_relevant_impl<F: FnMut(DefId)>(&self,
                                                   tcx: &ctxt<'tcx>,
                                                   self_ty: Ty<'tcx>,
                                                   mut f: F)
    {
        tcx.populate_implementations_for_trait_if_necessary(self.trait_ref.def_id);

        for &impl_def_id in self.blanket_impls.borrow().iter() {
            f(impl_def_id);
        }

        // simplify_type(.., false) basically replaces type parameters and
        // projections with infer-variables. This is, of course, done on
        // the impl trait-ref when it is instantiated, but not on the
        // predicate trait-ref which is passed here.
        //
        // for example, if we match `S: Copy` against an impl like
        // `impl<T:Copy> Copy for Option<T>`, we replace the type variable
        // in `Option<T>` with an infer variable, to `Option<_>` (this
        // doesn't actually change fast_reject output), but we don't
        // replace `S` with anything - this impl of course can't be
        // selected, and as there are hundreds of similar impls,
        // considering them would significantly harm performance.
        if let Some(simp) = fast_reject::simplify_type(tcx, self_ty, true) {
            if let Some(impls) = self.nonblanket_impls.borrow().get(&simp) {
                for &impl_def_id in impls {
                    f(impl_def_id);
                }
            }
        } else {
            for v in self.nonblanket_impls.borrow().values() {
                for &impl_def_id in v {
                    f(impl_def_id);
                }
            }
        }
    }

}

bitflags! {
    flags AdtFlags: u32 {
        const NO_ADT_FLAGS        = 0,
        const IS_ENUM             = 1 << 0,
        const IS_DTORCK           = 1 << 1, // is this a dtorck type?
        const IS_DTORCK_VALID     = 1 << 2,
        const IS_PHANTOM_DATA     = 1 << 3,
        const IS_SIMD             = 1 << 4,
        const IS_FUNDAMENTAL      = 1 << 5,
        const IS_NO_DROP_FLAG     = 1 << 6,
    }
}

pub type AdtDef<'tcx> = &'tcx AdtDefData<'tcx, 'static>;
pub type VariantDef<'tcx> = &'tcx VariantDefData<'tcx, 'static>;
pub type FieldDef<'tcx> = &'tcx FieldDefData<'tcx, 'static>;

// See comment on AdtDefData for explanation
pub type AdtDefMaster<'tcx> = &'tcx AdtDefData<'tcx, 'tcx>;
pub type VariantDefMaster<'tcx> = &'tcx VariantDefData<'tcx, 'tcx>;
pub type FieldDefMaster<'tcx> = &'tcx FieldDefData<'tcx, 'tcx>;

pub struct VariantDefData<'tcx, 'container: 'tcx> {
    pub did: DefId,
    pub name: Name, // struct's name if this is a struct
    pub disr_val: Disr,
    pub fields: Vec<FieldDefData<'tcx, 'container>>
}

pub struct FieldDefData<'tcx, 'container: 'tcx> {
    /// The field's DefId. NOTE: the fields of tuple-like enum variants
    /// are not real items, and don't have entries in tcache etc.
    pub did: DefId,
    /// special_idents::unnamed_field.name
    /// if this is a tuple-like field
    pub name: Name,
    pub vis: ast::Visibility,
    /// TyIVar is used here to allow for variance (see the doc at
    /// AdtDefData).
    ty: TyIVar<'tcx, 'container>
}

/// The definition of an abstract data type - a struct or enum.
///
/// These are all interned (by intern_adt_def) into the adt_defs
/// table.
///
/// Because of the possibility of nested tcx-s, this type
/// needs 2 lifetimes: the traditional variant lifetime ('tcx)
/// bounding the lifetime of the inner types is of course necessary.
/// However, it is not sufficient - types from a child tcx must
/// not be leaked into the master tcx by being stored in an AdtDefData.
///
/// The 'container lifetime ensures that by outliving the container
/// tcx and preventing shorter-lived types from being inserted. When
/// write access is not needed, the 'container lifetime can be
/// erased to 'static, which can be done by the AdtDef wrapper.
pub struct AdtDefData<'tcx, 'container: 'tcx> {
    pub did: DefId,
    pub variants: Vec<VariantDefData<'tcx, 'container>>,
    destructor: Cell<Option<DefId>>,
    flags: Cell<AdtFlags>,
}

impl<'tcx, 'container> PartialEq for AdtDefData<'tcx, 'container> {
    // AdtDefData are always interned and this is part of TyS equality
    #[inline]
    fn eq(&self, other: &Self) -> bool { self as *const _ == other as *const _ }
}

impl<'tcx, 'container> Eq for AdtDefData<'tcx, 'container> {}

impl<'tcx, 'container> Hash for AdtDefData<'tcx, 'container> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const AdtDefData).hash(s)
    }
}


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AdtKind { Struct, Enum }

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VariantKind { Dict, Tuple, Unit }

impl<'tcx, 'container> AdtDefData<'tcx, 'container> {
    fn new(tcx: &ctxt<'tcx>,
           did: DefId,
           kind: AdtKind,
           variants: Vec<VariantDefData<'tcx, 'container>>) -> Self {
        let mut flags = AdtFlags::NO_ADT_FLAGS;
        let attrs = tcx.get_attrs(did);
        if attr::contains_name(&attrs, "fundamental") {
            flags = flags | AdtFlags::IS_FUNDAMENTAL;
        }
        if attr::contains_name(&attrs, "unsafe_no_drop_flag") {
            flags = flags | AdtFlags::IS_NO_DROP_FLAG;
        }
        if tcx.lookup_simd(did) {
            flags = flags | AdtFlags::IS_SIMD;
        }
        if Some(did) == tcx.lang_items.phantom_data() {
            flags = flags | AdtFlags::IS_PHANTOM_DATA;
        }
        if let AdtKind::Enum = kind {
            flags = flags | AdtFlags::IS_ENUM;
        }
        AdtDefData {
            did: did,
            variants: variants,
            flags: Cell::new(flags),
            destructor: Cell::new(None)
        }
    }

    fn calculate_dtorck(&'tcx self, tcx: &ctxt<'tcx>) {
        if tcx.is_adt_dtorck(self) {
            self.flags.set(self.flags.get() | AdtFlags::IS_DTORCK);
        }
        self.flags.set(self.flags.get() | AdtFlags::IS_DTORCK_VALID)
    }

    /// Returns the kind of the ADT - Struct or Enum.
    #[inline]
    pub fn adt_kind(&self) -> AdtKind {
        if self.flags.get().intersects(AdtFlags::IS_ENUM) {
            AdtKind::Enum
        } else {
            AdtKind::Struct
        }
    }

    /// Returns whether this is a dtorck type. If this returns
    /// true, this type being safe for destruction requires it to be
    /// alive; Otherwise, only the contents are required to be.
    #[inline]
    pub fn is_dtorck(&'tcx self, tcx: &ctxt<'tcx>) -> bool {
        if !self.flags.get().intersects(AdtFlags::IS_DTORCK_VALID) {
            self.calculate_dtorck(tcx)
        }
        self.flags.get().intersects(AdtFlags::IS_DTORCK)
    }

    /// Returns whether this type is #[fundamental] for the purposes
    /// of coherence checking.
    #[inline]
    pub fn is_fundamental(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_FUNDAMENTAL)
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_SIMD)
    }

    /// Returns true if this is PhantomData<T>.
    #[inline]
    pub fn is_phantom_data(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_PHANTOM_DATA)
    }

    /// Returns whether this type has a destructor.
    pub fn has_dtor(&self) -> bool {
        match self.dtor_kind() {
            NoDtor => false,
            TraitDtor(..) => true
        }
    }

    /// Asserts this is a struct and returns the struct's unique
    /// variant.
    pub fn struct_variant(&self) -> &VariantDefData<'tcx, 'container> {
        assert!(self.adt_kind() == AdtKind::Struct);
        &self.variants[0]
    }

    #[inline]
    pub fn type_scheme(&self, tcx: &ctxt<'tcx>) -> TypeScheme<'tcx> {
        tcx.lookup_item_type(self.did)
    }

    #[inline]
    pub fn predicates(&self, tcx: &ctxt<'tcx>) -> GenericPredicates<'tcx> {
        tcx.lookup_predicates(self.did)
    }

    /// Returns an iterator over all fields contained
    /// by this ADT.
    #[inline]
    pub fn all_fields(&self) ->
            iter::FlatMap<
                slice::Iter<VariantDefData<'tcx, 'container>>,
                slice::Iter<FieldDefData<'tcx, 'container>>,
                for<'s> fn(&'s VariantDefData<'tcx, 'container>)
                    -> slice::Iter<'s, FieldDefData<'tcx, 'container>>
            > {
        self.variants.iter().flat_map(VariantDefData::fields_iter)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    #[inline]
    pub fn is_univariant(&self) -> bool {
        self.variants.len() == 1
    }

    pub fn is_payloadfree(&self) -> bool {
        !self.variants.is_empty() &&
            self.variants.iter().all(|v| v.fields.is_empty())
    }

    pub fn variant_with_id(&self, vid: DefId) -> &VariantDefData<'tcx, 'container> {
        self.variants
            .iter()
            .find(|v| v.did == vid)
            .expect("variant_with_id: unknown variant")
    }

    pub fn variant_of_def(&self, def: def::Def) -> &VariantDefData<'tcx, 'container> {
        match def {
            def::DefVariant(_, vid, _) => self.variant_with_id(vid),
            def::DefStruct(..) | def::DefTy(..) => self.struct_variant(),
            _ => panic!("unexpected def {:?} in variant_of_def", def)
        }
    }

    pub fn destructor(&self) -> Option<DefId> {
        self.destructor.get()
    }

    pub fn set_destructor(&self, dtor: DefId) {
        assert!(self.destructor.get().is_none());
        self.destructor.set(Some(dtor));
    }

    pub fn dtor_kind(&self) -> DtorKind {
        match self.destructor.get() {
            Some(_) => {
                TraitDtor(!self.flags.get().intersects(AdtFlags::IS_NO_DROP_FLAG))
            }
            None => NoDtor,
        }
    }
}

impl<'tcx, 'container> VariantDefData<'tcx, 'container> {
    #[inline]
    fn fields_iter(&self) -> slice::Iter<FieldDefData<'tcx, 'container>> {
        self.fields.iter()
    }

    pub fn kind(&self) -> VariantKind {
        match self.fields.get(0) {
            None => VariantKind::Unit,
            Some(&FieldDefData { name, .. }) if name == special_idents::unnamed_field.name => {
                VariantKind::Tuple
            }
            Some(_) => VariantKind::Dict
        }
    }

    pub fn is_tuple_struct(&self) -> bool {
        self.kind() == VariantKind::Tuple
    }

    #[inline]
    pub fn find_field_named(&self,
                            name: ast::Name)
                            -> Option<&FieldDefData<'tcx, 'container>> {
        self.fields.iter().find(|f| f.name == name)
    }

    #[inline]
    pub fn field_named(&self, name: ast::Name) -> &FieldDefData<'tcx, 'container> {
        self.find_field_named(name).unwrap()
    }
}

impl<'tcx, 'container> FieldDefData<'tcx, 'container> {
    pub fn new(did: DefId,
               name: Name,
               vis: ast::Visibility) -> Self {
        FieldDefData {
            did: did,
            name: name,
            vis: vis,
            ty: TyIVar::new()
        }
    }

    pub fn ty(&self, tcx: &ctxt<'tcx>, subst: &Substs<'tcx>) -> Ty<'tcx> {
        self.unsubst_ty().subst(tcx, subst)
    }

    pub fn unsubst_ty(&self) -> Ty<'tcx> {
        self.ty.unwrap()
    }

    pub fn fulfill_ty(&self, ty: Ty<'container>) {
        self.ty.fulfill(ty);
    }
}

/// Records the substitutions used to translate the polytype for an
/// item into the monotype of an item reference.
#[derive(Clone)]
pub struct ItemSubsts<'tcx> {
    pub substs: Substs<'tcx>,
}

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug, RustcEncodable, RustcDecodable)]
pub enum ClosureKind {
    // Warning: Ordering is significant here! The ordering is chosen
    // because the trait Fn is a subtrait of FnMut and so in turn, and
    // hence we order it so that Fn < FnMut < FnOnce.
    FnClosureKind,
    FnMutClosureKind,
    FnOnceClosureKind,
}

impl ClosureKind {
    pub fn trait_did(&self, cx: &ctxt) -> DefId {
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

    /// True if this a type that impls this closure kind
    /// must also implement `other`.
    pub fn extends(self, other: ty::ClosureKind) -> bool {
        match (self, other) {
            (FnClosureKind, FnClosureKind) => true,
            (FnClosureKind, FnMutClosureKind) => true,
            (FnClosureKind, FnOnceClosureKind) => true,
            (FnMutClosureKind, FnMutClosureKind) => true,
            (FnMutClosureKind, FnOnceClosureKind) => true,
            (FnOnceClosureKind, FnOnceClosureKind) => true,
            _ => false,
        }
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(arena: &'tcx TypedArena<TyS<'tcx>>,
           interner: &RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>)
           -> CommonTypes<'tcx>
    {
        let mk = |sty| ctxt::intern_ty(arena, interner, sty);
        CommonTypes {
            bool: mk(TyBool),
            char: mk(TyChar),
            err: mk(TyError),
            isize: mk(TyInt(ast::TyIs)),
            i8: mk(TyInt(ast::TyI8)),
            i16: mk(TyInt(ast::TyI16)),
            i32: mk(TyInt(ast::TyI32)),
            i64: mk(TyInt(ast::TyI64)),
            usize: mk(TyUint(ast::TyUs)),
            u8: mk(TyUint(ast::TyU8)),
            u16: mk(TyUint(ast::TyU16)),
            u32: mk(TyUint(ast::TyU32)),
            u64: mk(TyUint(ast::TyU64)),
            f32: mk(TyFloat(ast::TyF32)),
            f64: mk(TyFloat(ast::TyF64)),
        }
    }
}

struct FlagComputation {
    flags: TypeFlags,

    // maximum depth of any bound region that we have seen thus far
    depth: u32,
}

impl FlagComputation {
    fn new() -> FlagComputation {
        FlagComputation { flags: TypeFlags::empty(), depth: 0 }
    }

    fn for_sty(st: &TypeVariants) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_sty(st);
        result
    }

    fn add_flags(&mut self, flags: TypeFlags) {
        self.flags = self.flags | (flags & TypeFlags::NOMINAL_FLAGS);
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

    fn add_sty(&mut self, st: &TypeVariants) {
        match st {
            &TyBool |
            &TyChar |
            &TyInt(_) |
            &TyFloat(_) |
            &TyUint(_) |
            &TyStr => {
            }

            // You might think that we could just return TyError for
            // any type containing TyError as a component, and get
            // rid of the TypeFlags::HAS_TY_ERR flag -- likewise for ty_bot (with
            // the exception of function types that return bot).
            // But doing so caused sporadic memory corruption, and
            // neither I (tjc) nor nmatsakis could figure out why,
            // so we're doing it this way.
            &TyError => {
                self.add_flags(TypeFlags::HAS_TY_ERR)
            }

            &TyParam(ref p) => {
                self.add_flags(TypeFlags::HAS_LOCAL_NAMES);
                if p.space == subst::SelfSpace {
                    self.add_flags(TypeFlags::HAS_SELF);
                } else {
                    self.add_flags(TypeFlags::HAS_PARAMS);
                }
            }

            &TyClosure(_, ref substs) => {
                self.add_flags(TypeFlags::HAS_TY_CLOSURE);
                self.add_flags(TypeFlags::HAS_LOCAL_NAMES);
                self.add_substs(&substs.func_substs);
                self.add_tys(&substs.upvar_tys);
            }

            &TyInfer(_) => {
                self.add_flags(TypeFlags::HAS_LOCAL_NAMES); // it might, right?
                self.add_flags(TypeFlags::HAS_TY_INFER)
            }

            &TyEnum(_, substs) | &TyStruct(_, substs) => {
                self.add_substs(substs);
            }

            &TyProjection(ref data) => {
                self.add_flags(TypeFlags::HAS_PROJECTION);
                self.add_projection_ty(data);
            }

            &TyTrait(box TraitTy { ref principal, ref bounds }) => {
                let mut computation = FlagComputation::new();
                computation.add_substs(principal.0.substs);
                for projection_bound in &bounds.projection_bounds {
                    let mut proj_computation = FlagComputation::new();
                    proj_computation.add_projection_predicate(&projection_bound.0);
                    self.add_bound_computation(&proj_computation);
                }
                self.add_bound_computation(&computation);

                self.add_bounds(bounds);
            }

            &TyBox(tt) | &TyArray(tt, _) | &TySlice(tt) => {
                self.add_ty(tt)
            }

            &TyRawPtr(ref m) => {
                self.add_ty(m.ty);
            }

            &TyRef(r, ref m) => {
                self.add_region(*r);
                self.add_ty(m.ty);
            }

            &TyTuple(ref ts) => {
                self.add_tys(&ts[..]);
            }

            &TyBareFn(_, ref f) => {
                self.add_fn_sig(&f.sig);
            }
        }
    }

    fn add_ty(&mut self, ty: Ty) {
        self.add_flags(ty.flags.get());
        self.add_depth(ty.region_depth);
    }

    fn add_tys(&mut self, tys: &[Ty]) {
        for &ty in tys {
            self.add_ty(ty);
        }
    }

    fn add_fn_sig(&mut self, fn_sig: &PolyFnSig) {
        let mut computation = FlagComputation::new();

        computation.add_tys(&fn_sig.0.inputs);

        if let ty::FnConverging(output) = fn_sig.0.output {
            computation.add_ty(output);
        }

        self.add_bound_computation(&computation);
    }

    fn add_region(&mut self, r: Region) {
        match r {
            ty::ReVar(..) |
            ty::ReSkolemized(..) => { self.add_flags(TypeFlags::HAS_RE_INFER); }
            ty::ReLateBound(debruijn, _) => { self.add_depth(debruijn.depth); }
            ty::ReEarlyBound(..) => { self.add_flags(TypeFlags::HAS_RE_EARLY_BOUND); }
            ty::ReStatic => {}
            _ => { self.add_flags(TypeFlags::HAS_FREE_REGIONS); }
        }

        if !r.is_global() {
            self.add_flags(TypeFlags::HAS_LOCAL_NAMES);
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
                for &r in regions {
                    self.add_region(r);
                }
            }
        }
    }

    fn add_bounds(&mut self, bounds: &ExistentialBounds) {
        self.add_region(bounds.region_bound);
    }
}

impl<'tcx> ctxt<'tcx> {
    /// Create a type context and call the closure with a `&ty::ctxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_and_enter<F, R>(s: Session,
                                 arenas: &'tcx CtxtArenas<'tcx>,
                                 def_map: DefMap,
                                 named_region_map: resolve_lifetime::NamedRegionMap,
                                 map: ast_map::Map<'tcx>,
                                 freevars: RefCell<FreevarMap>,
                                 region_maps: RegionMaps,
                                 lang_items: middle::lang_items::LanguageItems,
                                 stability: stability::Index<'tcx>,
                                 f: F) -> (Session, R)
                                 where F: FnOnce(&ctxt<'tcx>) -> R
    {
        let interner = RefCell::new(FnvHashMap());
        let common_types = CommonTypes::new(&arenas.type_, &interner);

        tls::enter(ctxt {
            arenas: arenas,
            interner: interner,
            substs_interner: RefCell::new(FnvHashMap()),
            bare_fn_interner: RefCell::new(FnvHashMap()),
            region_interner: RefCell::new(FnvHashMap()),
            stability_interner: RefCell::new(FnvHashMap()),
            types: common_types,
            named_region_map: named_region_map,
            region_maps: region_maps,
            free_region_maps: RefCell::new(FnvHashMap()),
            item_variance_map: RefCell::new(DefIdMap()),
            variance_computed: Cell::new(false),
            sess: s,
            def_map: def_map,
            tables: RefCell::new(Tables::empty()),
            impl_trait_refs: RefCell::new(DefIdMap()),
            trait_defs: RefCell::new(DefIdMap()),
            adt_defs: RefCell::new(DefIdMap()),
            predicates: RefCell::new(DefIdMap()),
            super_predicates: RefCell::new(DefIdMap()),
            fulfilled_predicates: RefCell::new(traits::FulfilledPredicates::new()),
            map: map,
            freevars: freevars,
            tcache: RefCell::new(DefIdMap()),
            rcache: RefCell::new(FnvHashMap()),
            tc_cache: RefCell::new(FnvHashMap()),
            ast_ty_to_ty_cache: RefCell::new(NodeMap()),
            impl_or_trait_items: RefCell::new(DefIdMap()),
            trait_item_def_ids: RefCell::new(DefIdMap()),
            trait_items_cache: RefCell::new(DefIdMap()),
            ty_param_defs: RefCell::new(NodeMap()),
            normalized_cache: RefCell::new(FnvHashMap()),
            lang_items: lang_items,
            provided_method_sources: RefCell::new(DefIdMap()),
            destructors: RefCell::new(DefIdSet()),
            inherent_impls: RefCell::new(DefIdMap()),
            impl_items: RefCell::new(DefIdMap()),
            used_unsafe: RefCell::new(NodeSet()),
            used_mut_nodes: RefCell::new(NodeSet()),
            populated_external_types: RefCell::new(DefIdSet()),
            populated_external_primitive_impls: RefCell::new(DefIdSet()),
            extern_const_statics: RefCell::new(DefIdMap()),
            extern_const_variants: RefCell::new(DefIdMap()),
            extern_const_fns: RefCell::new(DefIdMap()),
            node_lint_levels: RefCell::new(FnvHashMap()),
            transmute_restrictions: RefCell::new(Vec::new()),
            stability: RefCell::new(stability),
            selection_cache: traits::SelectionCache::new(),
            repr_hint_cache: RefCell::new(DefIdMap()),
            const_qualif_map: RefCell::new(NodeMap()),
            custom_coerce_unsized_kinds: RefCell::new(DefIdMap()),
            cast_kinds: RefCell::new(NodeMap()),
            fragment_infos: RefCell::new(DefIdMap()),
       }, f)
    }

    // Type constructors

    pub fn mk_substs(&self, substs: Substs<'tcx>) -> &'tcx Substs<'tcx> {
        if let Some(substs) = self.substs_interner.borrow().get(&substs) {
            return *substs;
        }

        let substs = self.arenas.substs.alloc(substs);
        self.substs_interner.borrow_mut().insert(substs, substs);
        substs
    }

    /// Create an unsafe fn ty based on a safe fn ty.
    pub fn safe_to_unsafe_fn_ty(&self, bare_fn: &BareFnTy<'tcx>) -> Ty<'tcx> {
        assert_eq!(bare_fn.unsafety, ast::Unsafety::Normal);
        let unsafe_fn_ty_a = self.mk_bare_fn(ty::BareFnTy {
            unsafety: ast::Unsafety::Unsafe,
            abi: bare_fn.abi,
            sig: bare_fn.sig.clone()
        });
        self.mk_fn(None, unsafe_fn_ty_a)
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

    pub fn closure_kind(&self, def_id: DefId) -> ty::ClosureKind {
        *self.tables.borrow().closure_kinds.get(&def_id).unwrap()
    }

    pub fn closure_type(&self,
                        def_id: DefId,
                        substs: &ClosureSubsts<'tcx>)
                        -> ty::ClosureTy<'tcx>
    {
        self.tables.borrow().closure_tys.get(&def_id).unwrap().subst(self, &substs.func_substs)
    }

    pub fn type_parameter_def(&self,
                              node_id: ast::NodeId)
                              -> TypeParameterDef<'tcx>
    {
        self.ty_param_defs.borrow().get(&node_id).unwrap().clone()
    }

    pub fn pat_contains_ref_binding(&self, pat: &ast::Pat) -> Option<ast::Mutability> {
        pat_util::pat_contains_ref_binding(&self.def_map, pat)
    }

    pub fn arm_contains_ref_binding(&self, arm: &ast::Arm) -> Option<ast::Mutability> {
        pat_util::arm_contains_ref_binding(&self.def_map, arm)
    }

    fn intern_ty(type_arena: &'tcx TypedArena<TyS<'tcx>>,
                 interner: &RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>,
                 st: TypeVariants<'tcx>)
                 -> Ty<'tcx> {
        let ty: Ty /* don't be &mut TyS */ = {
            let mut interner = interner.borrow_mut();
            match interner.get(&st) {
                Some(ty) => return *ty,
                _ => ()
            }

            let flags = FlagComputation::for_sty(&st);

            let ty = match () {
                () => type_arena.alloc(TyS { sty: st,
                                             flags: Cell::new(flags.flags),
                                             region_depth: flags.depth, }),
            };

            interner.insert(InternedTy { ty: ty }, ty);
            ty
        };

        debug!("Interned type: {:?} Pointer: {:?}",
            ty, ty as *const TyS);
        ty
    }

    // Interns a type/name combination, stores the resulting box in cx.interner,
    // and returns the box as cast to an unsafe ptr (see comments for Ty above).
    pub fn mk_ty(&self, st: TypeVariants<'tcx>) -> Ty<'tcx> {
        ctxt::intern_ty(&self.arenas.type_, &self.interner, st)
    }

    pub fn mk_mach_int(&self, tm: ast::IntTy) -> Ty<'tcx> {
        match tm {
            ast::TyIs   => self.types.isize,
            ast::TyI8   => self.types.i8,
            ast::TyI16  => self.types.i16,
            ast::TyI32  => self.types.i32,
            ast::TyI64  => self.types.i64,
        }
    }

    pub fn mk_mach_uint(&self, tm: ast::UintTy) -> Ty<'tcx> {
        match tm {
            ast::TyUs   => self.types.usize,
            ast::TyU8   => self.types.u8,
            ast::TyU16  => self.types.u16,
            ast::TyU32  => self.types.u32,
            ast::TyU64  => self.types.u64,
        }
    }

    pub fn mk_mach_float(&self, tm: ast::FloatTy) -> Ty<'tcx> {
        match tm {
            ast::TyF32  => self.types.f32,
            ast::TyF64  => self.types.f64,
        }
    }

    pub fn mk_str(&self) -> Ty<'tcx> {
        self.mk_ty(TyStr)
    }

    pub fn mk_static_str(&self) -> Ty<'tcx> {
        self.mk_imm_ref(self.mk_region(ty::ReStatic), self.mk_str())
    }

    pub fn mk_enum(&self, def: AdtDef<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyEnum(def, substs))
    }

    pub fn mk_box(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyBox(ty))
    }

    pub fn mk_ptr(&self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRawPtr(tm))
    }

    pub fn mk_ref(&self, r: &'tcx Region, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRef(r, tm))
    }

    pub fn mk_mut_ref(&self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: ast::MutMutable})
    }

    pub fn mk_imm_ref(&self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: ast::MutImmutable})
    }

    pub fn mk_mut_ptr(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: ast::MutMutable})
    }

    pub fn mk_imm_ptr(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: ast::MutImmutable})
    }

    pub fn mk_nil_ptr(&self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_nil())
    }

    pub fn mk_array(&self, ty: Ty<'tcx>, n: usize) -> Ty<'tcx> {
        self.mk_ty(TyArray(ty, n))
    }

    pub fn mk_slice(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TySlice(ty))
    }

    pub fn mk_tup(&self, ts: Vec<Ty<'tcx>>) -> Ty<'tcx> {
        self.mk_ty(TyTuple(ts))
    }

    pub fn mk_nil(&self) -> Ty<'tcx> {
        self.mk_tup(Vec::new())
    }

    pub fn mk_bool(&self) -> Ty<'tcx> {
        self.mk_ty(TyBool)
    }

    pub fn mk_fn(&self,
                 opt_def_id: Option<DefId>,
                 fty: &'tcx BareFnTy<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyBareFn(opt_def_id, fty))
    }

    pub fn mk_ctor_fn(&self,
                      def_id: DefId,
                      input_tys: &[Ty<'tcx>],
                      output: Ty<'tcx>) -> Ty<'tcx> {
        let input_args = input_tys.iter().cloned().collect();
        self.mk_fn(Some(def_id), self.mk_bare_fn(BareFnTy {
            unsafety: ast::Unsafety::Normal,
            abi: abi::Rust,
            sig: ty::Binder(FnSig {
                inputs: input_args,
                output: ty::FnConverging(output),
                variadic: false
            })
        }))
    }

    pub fn mk_trait(&self,
                    principal: ty::PolyTraitRef<'tcx>,
                    bounds: ExistentialBounds<'tcx>)
                    -> Ty<'tcx>
    {
        assert!(bound_list_is_sorted(&bounds.projection_bounds));

        let inner = box TraitTy {
            principal: principal,
            bounds: bounds
        };
        self.mk_ty(TyTrait(inner))
    }

    pub fn mk_projection(&self,
                         trait_ref: TraitRef<'tcx>,
                         item_name: ast::Name)
                         -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        let inner = ProjectionTy { trait_ref: trait_ref, item_name: item_name };
        self.mk_ty(TyProjection(inner))
    }

    pub fn mk_struct(&self, def: AdtDef<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyStruct(def, substs))
    }

    pub fn mk_closure(&self,
                      closure_id: DefId,
                      substs: &'tcx Substs<'tcx>,
                      tys: Vec<Ty<'tcx>>)
                      -> Ty<'tcx> {
        self.mk_closure_from_closure_substs(closure_id, Box::new(ClosureSubsts {
            func_substs: substs,
            upvar_tys: tys
        }))
    }

    pub fn mk_closure_from_closure_substs(&self,
                                          closure_id: DefId,
                                          closure_substs: Box<ClosureSubsts<'tcx>>)
                                          -> Ty<'tcx> {
        self.mk_ty(TyClosure(closure_id, closure_substs))
    }

    pub fn mk_var(&self, v: TyVid) -> Ty<'tcx> {
        self.mk_infer(TyVar(v))
    }

    pub fn mk_int_var(&self, v: IntVid) -> Ty<'tcx> {
        self.mk_infer(IntVar(v))
    }

    pub fn mk_float_var(&self, v: FloatVid) -> Ty<'tcx> {
        self.mk_infer(FloatVar(v))
    }

    pub fn mk_infer(&self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(TyInfer(it))
    }

    pub fn mk_param(&self,
                    space: subst::ParamSpace,
                    index: u32,
                    name: ast::Name) -> Ty<'tcx> {
        self.mk_ty(TyParam(ParamTy { space: space, idx: index, name: name }))
    }

    pub fn mk_self_type(&self) -> Ty<'tcx> {
        self.mk_param(subst::SelfSpace, 0, special_idents::type_self.name)
    }

    pub fn mk_param_from_def(&self, def: &TypeParameterDef) -> Ty<'tcx> {
        self.mk_param(def.space, def.index, def.name)
    }
}

fn bound_list_is_sorted(bounds: &[ty::PolyProjectionPredicate]) -> bool {
    bounds.is_empty() ||
        bounds[1..].iter().enumerate().all(
            |(index, bound)| bounds[index].sort_key() <= bound.sort_key())
}

pub fn sort_bounds_list(bounds: &mut [ty::PolyProjectionPredicate]) {
    bounds.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()))
}

impl<'tcx> TyS<'tcx> {
    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```notrust
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(&'tcx self) -> TypeWalker<'tcx> {
        TypeWalker::new(self)
    }

    /// Iterator that walks the immediate children of `self`.  Hence
    /// `Foo<Bar<i32>, u32>` yields the sequence `[Bar<i32>, u32]`
    /// (but not `i32`, like `walk`).
    pub fn walk_shallow(&'tcx self) -> IntoIter<Ty<'tcx>> {
        ty_walk::walk_shallow(self)
    }

    pub fn as_opt_param_ty(&self) -> Option<ty::ParamTy> {
        match self.sty {
            ty::TyParam(ref d) => Some(d.clone()),
            _ => None,
        }
    }

    pub fn is_param(&self, space: ParamSpace, index: u32) -> bool {
        match self.sty {
            ty::TyParam(ref data) => data.space == space && data.idx == index,
            _ => false,
        }
    }

    /// Returns the regions directly referenced from this type (but
    /// not types reachable from this type via `walk_tys`). This
    /// ignores late-bound regions binders.
    pub fn regions(&self) -> Vec<ty::Region> {
        match self.sty {
            TyRef(region, _) => {
                vec![*region]
            }
            TyTrait(ref obj) => {
                let mut v = vec![obj.bounds.region_bound];
                v.push_all(obj.principal.skip_binder().substs.regions().as_slice());
                v
            }
            TyEnum(_, substs) |
            TyStruct(_, substs) => {
                substs.regions().as_slice().to_vec()
            }
            TyClosure(_, ref substs) => {
                substs.func_substs.regions().as_slice().to_vec()
            }
            TyProjection(ref data) => {
                data.trait_ref.substs.regions().as_slice().to_vec()
            }
            TyBareFn(..) |
            TyBool |
            TyChar |
            TyInt(_) |
            TyUint(_) |
            TyFloat(_) |
            TyBox(_) |
            TyStr |
            TyArray(_, _) |
            TySlice(_) |
            TyRawPtr(_) |
            TyTuple(_) |
            TyParam(_) |
            TyInfer(_) |
            TyError => {
                vec![]
            }
        }
    }

    /// Walks `ty` and any types appearing within `ty`, invoking the
    /// callback `f` on each type. If the callback returns false, then the
    /// children of the current type are ignored.
    ///
    /// Note: prefer `ty.walk()` where possible.
    pub fn maybe_walk<F>(&'tcx self, mut f: F)
        where F : FnMut(Ty<'tcx>) -> bool
    {
        let mut walker = self.walk();
        while let Some(ty) = walker.next() {
            if !f(ty) {
                walker.skip_current_subtree();
            }
        }
    }
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

    pub fn to_ty<'tcx>(self, tcx: &ctxt<'tcx>) -> Ty<'tcx> {
        tcx.mk_param(self.space, self.idx, self.name)
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

// Type utilities
impl<'tcx> TyS<'tcx> {
    pub fn is_nil(&self) -> bool {
        match self.sty {
            TyTuple(ref tys) => tys.is_empty(),
            _ => false
        }
    }

    pub fn is_empty(&self, _cx: &ctxt) -> bool {
        // FIXME(#24885): be smarter here
        match self.sty {
            TyEnum(def, _) | TyStruct(def, _) => def.is_empty(),
            _ => false
        }
    }

    pub fn is_ty_var(&self) -> bool {
        match self.sty {
            TyInfer(TyVar(_)) => true,
            _ => false
        }
    }

    pub fn is_bool(&self) -> bool { self.sty == TyBool }

    pub fn is_self(&self) -> bool {
        match self.sty {
            TyParam(ref p) => p.space == subst::SelfSpace,
            _ => false
        }
    }

    fn is_slice(&self) -> bool {
        match self.sty {
            TyRawPtr(mt) | TyRef(_, mt) => match mt.ty.sty {
                TySlice(_) | TyStr => true,
                _ => false,
            },
            _ => false
        }
    }

    pub fn is_structural(&self) -> bool {
        match self.sty {
            TyStruct(..) | TyTuple(_) | TyEnum(..) |
            TyArray(..) | TyClosure(..) => true,
            _ => self.is_slice() | self.is_trait()
        }
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        match self.sty {
            TyStruct(def, _) => def.is_simd(),
            _ => false
        }
    }

    pub fn sequence_element_type(&self, cx: &ctxt<'tcx>) -> Ty<'tcx> {
        match self.sty {
            TyArray(ty, _) | TySlice(ty) => ty,
            TyStr => cx.mk_mach_uint(ast::TyU8),
            _ => cx.sess.bug(&format!("sequence_element_type called on non-sequence value: {}",
                                      self)),
        }
    }

    pub fn simd_type(&self, cx: &ctxt<'tcx>) -> Ty<'tcx> {
        match self.sty {
            TyStruct(def, substs) => {
                def.struct_variant().fields[0].ty(cx, substs)
            }
            _ => panic!("simd_type called on invalid type")
        }
    }

    pub fn simd_size(&self, _cx: &ctxt) -> usize {
        match self.sty {
            TyStruct(def, _) => def.struct_variant().fields.len(),
            _ => panic!("simd_size called on invalid type")
        }
    }

    pub fn is_region_ptr(&self) -> bool {
        match self.sty {
            TyRef(..) => true,
            _ => false
        }
    }

    pub fn is_unsafe_ptr(&self) -> bool {
        match self.sty {
            TyRawPtr(_) => return true,
            _ => return false
        }
    }

    pub fn is_unique(&self) -> bool {
        match self.sty {
            TyBox(_) => true,
            _ => false
        }
    }

    /*
     A scalar type is one that denotes an atomic datum, with no sub-components.
     (A TyRawPtr is scalar because it represents a non-managed pointer, so its
     contents are abstract to rustc.)
    */
    pub fn is_scalar(&self) -> bool {
        match self.sty {
            TyBool | TyChar | TyInt(_) | TyFloat(_) | TyUint(_) |
            TyInfer(IntVar(_)) | TyInfer(FloatVar(_)) |
            TyBareFn(..) | TyRawPtr(_) => true,
            _ => false
        }
    }

    /// Returns true if this type is a floating point type and false otherwise.
    pub fn is_floating_point(&self) -> bool {
        match self.sty {
            TyFloat(_) |
            TyInfer(FloatVar(_)) => true,
            _ => false,
        }
    }

    pub fn ty_to_def_id(&self) -> Option<DefId> {
        match self.sty {
            TyTrait(ref tt) => Some(tt.principal_def_id()),
            TyStruct(def, _) |
            TyEnum(def, _) => Some(def.did),
            TyClosure(id, _) => Some(id),
            _ => None
        }
    }

    pub fn ty_adt_def(&self) -> Option<AdtDef<'tcx>> {
        match self.sty {
            TyStruct(adt, _) | TyEnum(adt, _) => Some(adt),
            _ => None
        }
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
        InteriorUnsafe                      = 0b0000_0000__0000_0000__0010,
        InteriorParam                       = 0b0000_0000__0000_0000__0100,
        // InteriorAll                         = 0b00000000__00000000__1111,

        // Things that are owned by the value (second and third nibbles):
        OwnsOwned                           = 0b0000_0000__0000_0001__0000,
        OwnsDtor                            = 0b0000_0000__0000_0010__0000,
        OwnsAll                             = 0b0000_0000__1111_1111__0000,

        // Things that mean drop glue is necessary
        NeedsDrop                           = 0b0000_0000__0000_0111__0000,

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

    pub fn owns_owned(&self) -> bool {
        self.intersects(TC::OwnsOwned)
    }

    pub fn interior_param(&self) -> bool {
        self.intersects(TC::InteriorParam)
    }

    pub fn interior_unsafe(&self) -> bool {
        self.intersects(TC::InteriorUnsafe)
    }

    pub fn needs_drop(&self, _: &ctxt) -> bool {
        self.intersects(TC::NeedsDrop)
    }

    /// Includes only those bits that still apply when indirected through a `Box` pointer
    pub fn owned_pointer(&self) -> TypeContents {
        TC::OwnsOwned | (*self & TC::OwnsAll)
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

impl<'tcx> TyS<'tcx> {
    pub fn type_contents(&'tcx self, cx: &ctxt<'tcx>) -> TypeContents {
        return memoized(&cx.tc_cache, self, |ty| {
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
                // usize and isize are ffi-unsafe
                TyUint(ast::TyUs) | TyInt(ast::TyIs) => {
                    TC::None
                }

                // Scalar and unique types are sendable, and durable
                TyInfer(ty::FreshIntTy(_)) | TyInfer(ty::FreshFloatTy(_)) |
                TyBool | TyInt(_) | TyUint(_) | TyFloat(_) |
                TyBareFn(..) | ty::TyChar => {
                    TC::None
                }

                TyBox(typ) => {
                    tc_ty(cx, typ, cache).owned_pointer()
                }

                TyTrait(_) => {
                    TC::All - TC::InteriorParam
                }

                TyRawPtr(_) => {
                    TC::None
                }

                TyRef(_, _) => {
                    TC::None
                }

                TyArray(ty, _) => {
                    tc_ty(cx, ty, cache)
                }

                TySlice(ty) => {
                    tc_ty(cx, ty, cache)
                }
                TyStr => TC::None,

                TyClosure(_, ref substs) => {
                    TypeContents::union(&substs.upvar_tys, |ty| tc_ty(cx, &ty, cache))
                }

                TyTuple(ref tys) => {
                    TypeContents::union(&tys[..],
                                        |ty| tc_ty(cx, *ty, cache))
                }

                TyStruct(def, substs) | TyEnum(def, substs) => {
                    let mut res =
                        TypeContents::union(&def.variants, |v| {
                            TypeContents::union(&v.fields, |f| {
                                tc_ty(cx, f.ty(cx, substs), cache)
                            })
                        });

                    if def.has_dtor() {
                        res = res | TC::OwnsDtor;
                    }

                    apply_lang_items(cx, def.did, res)
                }

                TyProjection(..) |
                TyParam(_) => {
                    TC::All
                }

                TyInfer(_) |
                TyError => {
                    cx.sess.bug("asked to compute contents of error type");
                }
            };

            cache.insert(ty, result);
            result
        }

        fn apply_lang_items(cx: &ctxt, did: DefId, tc: TypeContents)
                            -> TypeContents {
            if Some(did) == cx.lang_items.unsafe_cell_type() {
                tc | TC::InteriorUnsafe
            } else {
                tc
            }
        }
    }

    fn impls_bound<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                       bound: ty::BuiltinBound,
                       span: Span)
                       -> bool
    {
        let tcx = param_env.tcx;
        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(param_env.clone()), false);

        let is_impld = traits::type_known_to_meet_builtin_bound(&infcx,
                                                                self, bound, span);

        debug!("Ty::impls_bound({:?}, {:?}) = {:?}",
               self, bound, is_impld);

        is_impld
    }

    // FIXME (@jroesch): I made this public to use it, not sure if should be private
    pub fn moves_by_default<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                           span: Span) -> bool {
        if self.flags.get().intersects(TypeFlags::MOVENESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::MOVES_BY_DEFAULT);
        }

        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyRawPtr(..) | TyBareFn(..) | TyRef(_, TypeAndMut {
                mutbl: ast::MutImmutable, ..
            }) => Some(false),

            TyStr | TyBox(..) | TyRef(_, TypeAndMut {
                mutbl: ast::MutMutable, ..
            }) => Some(true),

            TyArray(..) | TySlice(_) | TyTrait(..) | TyTuple(..) |
            TyClosure(..) | TyEnum(..) | TyStruct(..) |
            TyProjection(..) | TyParam(..) | TyInfer(..) | TyError => None
        }.unwrap_or_else(|| !self.impls_bound(param_env, ty::BoundCopy, span));

        if !self.has_param_types() && !self.has_self_ty() {
            self.flags.set(self.flags.get() | if result {
                TypeFlags::MOVENESS_CACHED | TypeFlags::MOVES_BY_DEFAULT
            } else {
                TypeFlags::MOVENESS_CACHED
            });
        }

        result
    }

    #[inline]
    pub fn is_sized<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                        span: Span) -> bool
    {
        if self.flags.get().intersects(TypeFlags::SIZEDNESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::IS_SIZED);
        }

        self.is_sized_uncached(param_env, span)
    }

    fn is_sized_uncached<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                             span: Span) -> bool {
        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyBox(..) | TyRawPtr(..) | TyRef(..) | TyBareFn(..) |
            TyArray(..) | TyTuple(..) | TyClosure(..) => Some(true),

            TyStr | TyTrait(..) | TySlice(_) => Some(false),

            TyEnum(..) | TyStruct(..) | TyProjection(..) | TyParam(..) |
            TyInfer(..) | TyError => None
        }.unwrap_or_else(|| self.impls_bound(param_env, ty::BoundSized, span));

        if !self.has_param_types() && !self.has_self_ty() {
            self.flags.set(self.flags.get() | if result {
                TypeFlags::SIZEDNESS_CACHED | TypeFlags::IS_SIZED
            } else {
                TypeFlags::SIZEDNESS_CACHED
            });
        }

        result
    }

    // True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
    pub fn is_instantiable(&'tcx self, cx: &ctxt<'tcx>) -> bool {
        fn type_requires<'tcx>(cx: &ctxt<'tcx>, seen: &mut Vec<AdtDef<'tcx>>,
                               r_ty: Ty<'tcx>, ty: Ty<'tcx>) -> bool {
            debug!("type_requires({:?}, {:?})?",
                   r_ty, ty);

            let r = r_ty == ty || subtypes_require(cx, seen, r_ty, ty);

            debug!("type_requires({:?}, {:?})? {:?}",
                   r_ty, ty, r);
            return r;
        }

        fn subtypes_require<'tcx>(cx: &ctxt<'tcx>, seen: &mut Vec<AdtDef<'tcx>>,
                                  r_ty: Ty<'tcx>, ty: Ty<'tcx>) -> bool {
            debug!("subtypes_require({:?}, {:?})?",
                   r_ty, ty);

            let r = match ty.sty {
                // fixed length vectors need special treatment compared to
                // normal vectors, since they don't necessarily have the
                // possibility to have length zero.
                TyArray(_, 0) => false, // don't need no contents
                TyArray(ty, _) => type_requires(cx, seen, r_ty, ty),

                TyBool |
                TyChar |
                TyInt(_) |
                TyUint(_) |
                TyFloat(_) |
                TyStr |
                TyBareFn(..) |
                TyParam(_) |
                TyProjection(_) |
                TySlice(_) => {
                    false
                }
                TyBox(typ) => {
                    type_requires(cx, seen, r_ty, typ)
                }
                TyRef(_, ref mt) => {
                    type_requires(cx, seen, r_ty, mt.ty)
                }

                TyRawPtr(..) => {
                    false           // unsafe ptrs can always be NULL
                }

                TyTrait(..) => {
                    false
                }

                TyStruct(def, substs) | TyEnum(def, substs) => {
                    if seen.contains(&def) {
                        // FIXME(#27497) ???
                        false
                    } else if def.is_empty() {
                        // HACK: required for empty types to work. This
                        // check is basically a lint anyway.
                        false
                    } else {
                        seen.push(def);
                        let r = def.variants.iter().all(|v| v.fields.iter().any(|f| {
                            type_requires(cx, seen, r_ty, f.ty(cx, substs))
                        }));
                        seen.pop().unwrap();
                        r
                    }
                }

                TyError |
                TyInfer(_) |
                TyClosure(..) => {
                    // this check is run on type definitions, so we don't expect to see
                    // inference by-products or closure types
                    cx.sess.bug(&format!("requires check invoked on inapplicable type: {:?}", ty))
                }

                TyTuple(ref ts) => {
                    ts.iter().any(|ty| type_requires(cx, seen, r_ty, *ty))
                }
            };

            debug!("subtypes_require({:?}, {:?})? {:?}",
                   r_ty, ty, r);

            return r;
        }

        let mut seen = Vec::new();
        !subtypes_require(cx, &mut seen, self, self)
    }
}

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
///
/// The ordering of the cases is significant. They are sorted so that cmp::max
/// will keep the "more erroneous" of two values.
#[derive(Copy, Clone, PartialOrd, Ord, Eq, PartialEq, Debug)]
pub enum Representability {
    Representable,
    ContainsRecursive,
    SelfRecursive,
}

impl<'tcx> TyS<'tcx> {
    /// Check whether a type is representable. This means it cannot contain unboxed
    /// structural recursion. This check is needed for structs and enums.
    pub fn is_representable(&'tcx self, cx: &ctxt<'tcx>, sp: Span) -> Representability {

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
                TyTuple(ref ts) => {
                    find_nonrepresentable(cx, sp, seen, ts.iter().cloned())
                }
                // Fixed-length vectors.
                // FIXME(#11924) Behavior undecided for zero-length vectors.
                TyArray(ty, _) => {
                    is_type_structurally_recursive(cx, sp, seen, ty)
                }
                TyStruct(def, substs) | TyEnum(def, substs) => {
                    find_nonrepresentable(cx,
                                          sp,
                                          seen,
                                          def.all_fields().map(|f| f.ty(cx, substs)))
                }
                TyClosure(..) => {
                    // this check is run on type definitions, so we don't expect
                    // to see closure types
                    cx.sess.bug(&format!("requires check invoked on inapplicable type: {:?}", ty))
                }
                _ => Representable,
            }
        }

        fn same_struct_or_enum<'tcx>(ty: Ty<'tcx>, def: AdtDef<'tcx>) -> bool {
            match ty.sty {
                TyStruct(ty_def, _) | TyEnum(ty_def, _) => {
                     ty_def == def
                }
                _ => false
            }
        }

        fn same_type<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
            match (&a.sty, &b.sty) {
                (&TyStruct(did_a, ref substs_a), &TyStruct(did_b, ref substs_b)) |
                (&TyEnum(did_a, ref substs_a), &TyEnum(did_b, ref substs_b)) => {
                    if did_a != did_b {
                        return false;
                    }

                    let types_a = substs_a.types.get_slice(subst::TypeSpace);
                    let types_b = substs_b.types.get_slice(subst::TypeSpace);

                    let mut pairs = types_a.iter().zip(types_b);

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
            debug!("is_type_structurally_recursive: {:?}", ty);

            match ty.sty {
                TyStruct(def, _) | TyEnum(def, _) => {
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
                                if same_struct_or_enum(seen_type, def) {
                                    debug!("SelfRecursive: {:?} contains {:?}",
                                           seen_type,
                                           ty);
                                    return SelfRecursive;
                                }
                            }
                            None => {}
                        }

                        // We also need to know whether the first item contains other types
                        // that are structurally recursive. If we don't catch this case, we
                        // will recurse infinitely for some inputs.
                        //
                        // It is important that we DO take generic parameters into account
                        // here, so that code like this is considered SelfRecursive, not
                        // ContainsRecursive:
                        //
                        // struct Foo { Option<Option<Foo>> }

                        for &seen_type in iter {
                            if same_type(ty, seen_type) {
                                debug!("ContainsRecursive: {:?} contains {:?}",
                                       seen_type,
                                       ty);
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

        debug!("is_type_representable: {:?}", self);

        // To avoid a stack overflow when checking an enum variant or struct that
        // contains a different, structurally recursive type, maintain a stack
        // of seen types and check recursion for each of them (issues #3008, #3779).
        let mut seen: Vec<Ty> = Vec::new();
        let r = is_type_structurally_recursive(cx, sp, &mut seen, self);
        debug!("is_type_representable: {:?} is {:?}", self, r);
        r
    }

    pub fn is_trait(&self) -> bool {
        match self.sty {
            TyTrait(..) => true,
            _ => false
        }
    }

    pub fn is_integral(&self) -> bool {
        match self.sty {
            TyInfer(IntVar(_)) | TyInt(_) | TyUint(_) => true,
            _ => false
        }
    }

    pub fn is_fresh(&self) -> bool {
        match self.sty {
            TyInfer(FreshTy(_)) => true,
            TyInfer(FreshIntTy(_)) => true,
            TyInfer(FreshFloatTy(_)) => true,
            _ => false
        }
    }

    pub fn is_uint(&self) -> bool {
        match self.sty {
            TyInfer(IntVar(_)) | TyUint(ast::TyUs) => true,
            _ => false
        }
    }

    pub fn is_char(&self) -> bool {
        match self.sty {
            TyChar => true,
            _ => false
        }
    }

    pub fn is_bare_fn(&self) -> bool {
        match self.sty {
            TyBareFn(..) => true,
            _ => false
        }
    }

    pub fn is_bare_fn_item(&self) -> bool {
        match self.sty {
            TyBareFn(Some(_), _) => true,
            _ => false
        }
    }

    pub fn is_fp(&self) -> bool {
        match self.sty {
            TyInfer(FloatVar(_)) | TyFloat(_) => true,
            _ => false
        }
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integral() || self.is_fp()
    }

    pub fn is_signed(&self) -> bool {
        match self.sty {
            TyInt(_) => true,
            _ => false
        }
    }

    pub fn is_machine(&self) -> bool {
        match self.sty {
            TyInt(ast::TyIs) | TyUint(ast::TyUs) => false,
            TyInt(..) | TyUint(..) | TyFloat(..) => true,
            _ => false
        }
    }

    // Returns the type and mutability of *ty.
    //
    // The parameter `explicit` indicates if this is an *explicit* dereference.
    // Some types---notably unsafe ptrs---can only be dereferenced explicitly.
    pub fn builtin_deref(&self, explicit: bool) -> Option<TypeAndMut<'tcx>> {
        match self.sty {
            TyBox(ty) => {
                Some(TypeAndMut {
                    ty: ty,
                    mutbl: ast::MutImmutable,
                })
            },
            TyRef(_, mt) => Some(mt),
            TyRawPtr(mt) if explicit => Some(mt),
            _ => None
        }
    }

    // Returns the type of ty[i]
    pub fn builtin_index(&self) -> Option<Ty<'tcx>> {
        match self.sty {
            TyArray(ty, _) | TySlice(ty) => Some(ty),
            _ => None
        }
    }

    pub fn fn_sig(&self) -> &'tcx PolyFnSig<'tcx> {
        match self.sty {
            TyBareFn(_, ref f) => &f.sig,
            _ => panic!("Ty::fn_sig() called on non-fn type: {:?}", self)
        }
    }

    /// Returns the ABI of the given function.
    pub fn fn_abi(&self) -> abi::Abi {
        match self.sty {
            TyBareFn(_, ref f) => f.abi,
            _ => panic!("Ty::fn_abi() called on non-fn type"),
        }
    }

    // Type accessors for substructures of types
    pub fn fn_args(&self) -> ty::Binder<Vec<Ty<'tcx>>> {
        self.fn_sig().inputs()
    }

    pub fn fn_ret(&self) -> Binder<FnOutput<'tcx>> {
        self.fn_sig().output()
    }

    pub fn is_fn(&self) -> bool {
        match self.sty {
            TyBareFn(..) => true,
            _ => false
        }
    }

    /// See `expr_ty_adjusted`
    pub fn adjust<F>(&'tcx self, cx: &ctxt<'tcx>,
                     span: Span,
                     expr_id: ast::NodeId,
                     adjustment: Option<&AutoAdjustment<'tcx>>,
                     mut method_type: F)
                     -> Ty<'tcx> where
        F: FnMut(MethodCall) -> Option<Ty<'tcx>>,
    {
        if let TyError = self.sty {
            return self;
        }

        return match adjustment {
            Some(adjustment) => {
                match *adjustment {
                   AdjustReifyFnPointer => {
                        match self.sty {
                            ty::TyBareFn(Some(_), b) => {
                                cx.mk_fn(None, b)
                            }
                            _ => {
                                cx.sess.bug(
                                    &format!("AdjustReifyFnPointer adjustment on non-fn-item: \
                                              {:?}", self));
                            }
                        }
                    }

                   AdjustUnsafeFnPointer => {
                        match self.sty {
                            ty::TyBareFn(None, b) => cx.safe_to_unsafe_fn_ty(b),
                            ref b => {
                                cx.sess.bug(
                                    &format!("AdjustReifyFnPointer adjustment on non-fn-item: \
                                             {:?}",
                                            b));
                            }
                        }
                   }

                    AdjustDerefRef(ref adj) => {
                        let mut adjusted_ty = self;

                        if !adjusted_ty.references_error() {
                            for i in 0..adj.autoderefs {
                                let method_call = MethodCall::autoderef(expr_id, i as u32);
                                match method_type(method_call) {
                                    Some(method_ty) => {
                                        // Overloaded deref operators have all late-bound
                                        // regions fully instantiated and coverge.
                                        let fn_ret =
                                            cx.no_late_bound_regions(&method_ty.fn_ret()).unwrap();
                                        adjusted_ty = fn_ret.unwrap();
                                    }
                                    None => {}
                                }
                                match adjusted_ty.builtin_deref(true) {
                                    Some(mt) => { adjusted_ty = mt.ty; }
                                    None => {
                                        cx.sess.span_bug(
                                            span,
                                            &format!("the {}th autoderef failed: {}",
                                                    i,
                                                     adjusted_ty)
                                            );
                                    }
                                }
                            }
                        }

                        if let Some(target) = adj.unsize {
                            target
                        } else {
                            adjusted_ty.adjust_for_autoref(cx, adj.autoref)
                        }
                    }
                }
            }
            None => self
        };
    }

    pub fn adjust_for_autoref(&'tcx self, cx: &ctxt<'tcx>,
                              autoref: Option<AutoRef<'tcx>>)
                              -> Ty<'tcx> {
        match autoref {
            None => self,
            Some(AutoPtr(r, m)) => {
                cx.mk_ref(r, TypeAndMut { ty: self, mutbl: m })
            }
            Some(AutoUnsafe(m)) => {
                cx.mk_ptr(TypeAndMut { ty: self, mutbl: m })
            }
        }
    }

    fn sort_string(&self, cx: &ctxt) -> String {

        match self.sty {
            TyBool | TyChar | TyInt(_) |
            TyUint(_) | TyFloat(_) | TyStr => self.to_string(),
            TyTuple(ref tys) if tys.is_empty() => self.to_string(),

            TyEnum(def, _) => format!("enum `{}`", cx.item_path_str(def.did)),
            TyBox(_) => "box".to_string(),
            TyArray(_, n) => format!("array of {} elements", n),
            TySlice(_) => "slice".to_string(),
            TyRawPtr(_) => "*-ptr".to_string(),
            TyRef(_, _) => "&-ptr".to_string(),
            TyBareFn(Some(_), _) => format!("fn item"),
            TyBareFn(None, _) => "fn pointer".to_string(),
            TyTrait(ref inner) => {
                format!("trait {}", cx.item_path_str(inner.principal_def_id()))
            }
            TyStruct(def, _) => {
                format!("struct `{}`", cx.item_path_str(def.did))
            }
            TyClosure(..) => "closure".to_string(),
            TyTuple(_) => "tuple".to_string(),
            TyInfer(TyVar(_)) => "inferred type".to_string(),
            TyInfer(IntVar(_)) => "integral variable".to_string(),
            TyInfer(FloatVar(_)) => "floating-point variable".to_string(),
            TyInfer(FreshTy(_)) => "skolemized type".to_string(),
            TyInfer(FreshIntTy(_)) => "skolemized integral type".to_string(),
            TyInfer(FreshFloatTy(_)) => "skolemized floating-point type".to_string(),
            TyProjection(_) => "associated type".to_string(),
            TyParam(ref p) => {
                if p.space == subst::SelfSpace {
                    "Self".to_string()
                } else {
                    "type parameter".to_string()
                }
            }
            TyError => "type error".to_string(),
        }
    }
}
/// Explains the source of a type err in a short, human readable way. This is meant to be placed
/// in parentheses after some larger message. You should also invoke `note_and_explain_type_err()`
/// afterwards to present additional details, particularly when it comes to lifetime-related
/// errors.
impl<'tcx> fmt::Display for TypeError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TypeError::*;

        match *self {
            CyclicTy => write!(f, "cyclic type of infinite size"),
            Mismatch => write!(f, "types differ"),
            UnsafetyMismatch(values) => {
                write!(f, "expected {} fn, found {} fn",
                       values.expected,
                       values.found)
            }
            AbiMismatch(values) => {
                write!(f, "expected {} fn, found {} fn",
                       values.expected,
                       values.found)
            }
            Mutability => write!(f, "values differ in mutability"),
            BoxMutability => {
                write!(f, "boxed values differ in mutability")
            }
            VecMutability => write!(f, "vectors differ in mutability"),
            PtrMutability => write!(f, "pointers differ in mutability"),
            RefMutability => write!(f, "references differ in mutability"),
            TyParamSize(values) => {
                write!(f, "expected a type with {} type params, \
                           found one with {} type params",
                       values.expected,
                       values.found)
            }
            FixedArraySize(values) => {
                write!(f, "expected an array with a fixed size of {} elements, \
                           found one with {} elements",
                       values.expected,
                       values.found)
            }
            TupleSize(values) => {
                write!(f, "expected a tuple with {} elements, \
                           found one with {} elements",
                       values.expected,
                       values.found)
            }
            ArgCount => {
                write!(f, "incorrect number of function parameters")
            }
            RegionsDoesNotOutlive(..) => {
                write!(f, "lifetime mismatch")
            }
            RegionsNotSame(..) => {
                write!(f, "lifetimes are not the same")
            }
            RegionsNoOverlap(..) => {
                write!(f, "lifetimes do not intersect")
            }
            RegionsInsufficientlyPolymorphic(br, _) => {
                write!(f, "expected bound lifetime parameter {}, \
                           found concrete lifetime", br)
            }
            RegionsOverlyPolymorphic(br, _) => {
                write!(f, "expected concrete lifetime, \
                           found bound lifetime parameter {}", br)
            }
            Sorts(values) => tls::with(|tcx| {
                // A naive approach to making sure that we're not reporting silly errors such as:
                // (expected closure, found closure).
                let expected_str = values.expected.sort_string(tcx);
                let found_str = values.found.sort_string(tcx);
                if expected_str == found_str {
                    write!(f, "expected {}, found a different {}", expected_str, found_str)
                } else {
                    write!(f, "expected {}, found {}", expected_str, found_str)
                }
            }),
            Traits(values) => tls::with(|tcx| {
                write!(f, "expected trait `{}`, found trait `{}`",
                       tcx.item_path_str(values.expected),
                       tcx.item_path_str(values.found))
            }),
            BuiltinBoundsMismatch(values) => {
                if values.expected.is_empty() {
                    write!(f, "expected no bounds, found `{}`",
                           values.found)
                } else if values.found.is_empty() {
                    write!(f, "expected bounds `{}`, found no bounds",
                           values.expected)
                } else {
                    write!(f, "expected bounds `{}`, found bounds `{}`",
                           values.expected,
                           values.found)
                }
            }
            IntegerAsChar => {
                write!(f, "expected an integral type, found `char`")
            }
            IntMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`",
                       values.expected,
                       values.found)
            }
            FloatMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`",
                       values.expected,
                       values.found)
            }
            VariadicMismatch(ref values) => {
                write!(f, "expected {} fn, found {} function",
                       if values.expected { "variadic" } else { "non-variadic" },
                       if values.found { "variadic" } else { "non-variadic" })
            }
            ConvergenceMismatch(ref values) => {
                write!(f, "expected {} fn, found {} function",
                       if values.expected { "converging" } else { "diverging" },
                       if values.found { "converging" } else { "diverging" })
            }
            ProjectionNameMismatched(ref values) => {
                write!(f, "expected {}, found {}",
                       values.expected,
                       values.found)
            }
            ProjectionBoundsLength(ref values) => {
                write!(f, "expected {} associated type bindings, found {}",
                       values.expected,
                       values.found)
            },
            TyParamDefaultMismatch(ref values) => {
                write!(f, "conflicting type parameter defaults `{}` and `{}`",
                       values.expected.ty,
                       values.found.ty)
            }
        }
    }
}

/// Helper for looking things up in the various maps that are populated during
/// typeck::collect (e.g., `cx.impl_or_trait_items`, `cx.tcache`, etc).  All of
/// these share the pattern that if the id is local, it should have been loaded
/// into the map by the `typeck::collect` phase.  If the def-id is external,
/// then we have to go consult the crate loading code (and cache the result for
/// the future).
fn lookup_locally_or_in_crate_store<V, F>(descr: &str,
                                          def_id: DefId,
                                          map: &RefCell<DefIdMap<V>>,
                                          load_external: F) -> V where
    V: Clone,
    F: FnOnce() -> V,
{
    match map.borrow().get(&def_id).cloned() {
        Some(v) => { return v; }
        None => { }
    }

    if def_id.is_local() {
        panic!("No def'n found for {:?} in tcx.{}", def_id, descr);
    }
    let v = load_external();
    map.borrow_mut().insert(def_id, v.clone());
    v
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
    /// Returns the type of element at index `i` in tuple or tuple-like type `t`.
    /// For an enum `t`, `variant` is None only if `t` is a univariant enum.
    pub fn positional_element_ty(&self,
                                 ty: Ty<'tcx>,
                                 i: usize,
                                 variant: Option<DefId>) -> Option<Ty<'tcx>> {
        match (&ty.sty, variant) {
            (&TyStruct(def, substs), None) => {
                def.struct_variant().fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), Some(vid)) => {
                def.variant_with_id(vid).fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), None) => {
                assert!(def.is_univariant());
                def.variants[0].fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyTuple(ref v), None) => v.get(i).cloned(),
            _ => None
        }
    }

    /// Returns the type of element at field `n` in struct or struct-like type `t`.
    /// For an enum `t`, `variant` must be some def id.
    pub fn named_element_ty(&self,
                            ty: Ty<'tcx>,
                            n: ast::Name,
                            variant: Option<DefId>) -> Option<Ty<'tcx>> {
        match (&ty.sty, variant) {
            (&TyStruct(def, substs), None) => {
                def.struct_variant().find_field_named(n).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), Some(vid)) => {
                def.variant_with_id(vid).find_field_named(n).map(|f| f.ty(self, substs))
            }
            _ => return None
        }
    }

    pub fn node_id_to_type(&self, id: ast::NodeId) -> Ty<'tcx> {
        match self.node_id_to_type_opt(id) {
           Some(ty) => ty,
           None => self.sess.bug(
               &format!("node_id_to_type: no type for node `{}`",
                        self.map.node_to_string(id)))
        }
    }

    pub fn node_id_to_type_opt(&self, id: ast::NodeId) -> Option<Ty<'tcx>> {
        self.tables.borrow().node_types.get(&id).cloned()
    }

    pub fn node_id_item_substs(&self, id: ast::NodeId) -> ItemSubsts<'tcx> {
        match self.tables.borrow().item_substs.get(&id) {
            None => ItemSubsts::empty(),
            Some(ts) => ts.clone(),
        }
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &ast::Pat) -> Ty<'tcx> {
        self.node_id_to_type(pat.id)
    }
    pub fn pat_ty_opt(&self, pat: &ast::Pat) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(pat.id)
    }

    // Returns the type of an expression as a monotype.
    //
    // NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
    // some cases, we insert `AutoAdjustment` annotations such as auto-deref or
    // auto-ref.  The type returned by this function does not consider such
    // adjustments.  See `expr_ty_adjusted()` instead.
    //
    // NB (2): This type doesn't provide type parameter substitutions; e.g. if you
    // ask for the type of "id" in "id(3)", it will return "fn(&isize) -> isize"
    // instead of "fn(ty) -> T with T = isize".
    pub fn expr_ty(&self, expr: &ast::Expr) -> Ty<'tcx> {
        self.node_id_to_type(expr.id)
    }

    pub fn expr_ty_opt(&self, expr: &ast::Expr) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(expr.id)
    }

    /// Returns the type of `expr`, considering any `AutoAdjustment`
    /// entry recorded for that expression.
    ///
    /// It would almost certainly be better to store the adjusted ty in with
    /// the `AutoAdjustment`, but I opted not to do this because it would
    /// require serializing and deserializing the type and, although that's not
    /// hard to do, I just hate that code so much I didn't want to touch it
    /// unless it was to fix it properly, which seemed a distraction from the
    /// thread at hand! -nmatsakis
    pub fn expr_ty_adjusted(&self, expr: &ast::Expr) -> Ty<'tcx> {
        self.expr_ty(expr)
            .adjust(self, expr.span, expr.id,
                    self.tables.borrow().adjustments.get(&expr.id),
                    |method_call| {
            self.tables.borrow().method_map.get(&method_call).map(|method| method.ty)
        })
    }

    pub fn expr_span(&self, id: NodeId) -> Span {
        match self.map.find(id) {
            Some(ast_map::NodeExpr(e)) => {
                e.span
            }
            Some(f) => {
                self.sess.bug(&format!("Node id {} is not an expr: {:?}",
                                       id, f));
            }
            None => {
                self.sess.bug(&format!("Node id {} is not present \
                                        in the node map", id));
            }
        }
    }

    pub fn local_var_name_str(&self, id: NodeId) -> InternedString {
        match self.map.find(id) {
            Some(ast_map::NodeLocal(pat)) => {
                match pat.node {
                    ast::PatIdent(_, ref path1, _) => path1.node.name.as_str(),
                    _ => {
                        self.sess.bug(&format!("Variable id {} maps to {:?}, not local", id, pat));
                    },
                }
            },
            r => self.sess.bug(&format!("Variable id {} maps to {:?}, not local", id, r)),
        }
    }

    pub fn resolve_expr(&self, expr: &ast::Expr) -> def::Def {
        match self.def_map.borrow().get(&expr.id) {
            Some(def) => def.full_def(),
            None => {
                self.sess.span_bug(expr.span, &format!(
                    "no def-map entry for expr {}", expr.id));
            }
        }
    }

    pub fn expr_is_lval(&self, expr: &ast::Expr) -> bool {
         match expr.node {
            ast::ExprPath(..) => {
                // We can't use resolve_expr here, as this needs to run on broken
                // programs. We don't need to through - associated items are all
                // rvalues.
                match self.def_map.borrow().get(&expr.id) {
                    Some(&def::PathResolution {
                        base_def: def::DefStatic(..), ..
                    }) | Some(&def::PathResolution {
                        base_def: def::DefUpvar(..), ..
                    }) | Some(&def::PathResolution {
                        base_def: def::DefLocal(..), ..
                    }) => {
                        true
                    }

                    Some(..) => false,

                    None => self.sess.span_bug(expr.span, &format!(
                        "no def for path {}", expr.id))
                }
            }

            ast::ExprUnary(ast::UnDeref, _) |
            ast::ExprField(..) |
            ast::ExprTupField(..) |
            ast::ExprIndex(..) => {
                true
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
            ast::ExprVec(..) |
            ast::ExprBreak(..) |
            ast::ExprAgain(..) |
            ast::ExprRet(..) |
            ast::ExprWhile(..) |
            ast::ExprLoop(..) |
            ast::ExprAssign(..) |
            ast::ExprInlineAsm(..) |
            ast::ExprAssignOp(..) |
            ast::ExprLit(_) |
            ast::ExprUnary(..) |
            ast::ExprBox(..) |
            ast::ExprAddrOf(..) |
            ast::ExprBinary(..) |
            ast::ExprCast(..) => {
                false
            }

            ast::ExprParen(ref e) => self.expr_is_lval(e),

            ast::ExprIfLet(..) |
            ast::ExprWhileLet(..) |
            ast::ExprForLoop(..) |
            ast::ExprMac(..) => {
                self.sess.span_bug(
                    expr.span,
                    "macro expression remains after expansion");
            }
        }
    }

    pub fn note_and_explain_type_err(&self, err: &TypeError<'tcx>, sp: Span) {
        use self::TypeError::*;

        match err.clone() {
            RegionsDoesNotOutlive(subregion, superregion) => {
                self.note_and_explain_region("", subregion, "...");
                self.note_and_explain_region("...does not necessarily outlive ",
                                           superregion, "");
            }
            RegionsNotSame(region1, region2) => {
                self.note_and_explain_region("", region1, "...");
                self.note_and_explain_region("...is not the same lifetime as ",
                                           region2, "");
            }
            RegionsNoOverlap(region1, region2) => {
                self.note_and_explain_region("", region1, "...");
                self.note_and_explain_region("...does not overlap ",
                                           region2, "");
            }
            RegionsInsufficientlyPolymorphic(_, conc_region) => {
                self.note_and_explain_region("concrete lifetime that was found is ",
                                           conc_region, "");
            }
            RegionsOverlyPolymorphic(_, ty::ReVar(_)) => {
                // don't bother to print out the message below for
                // inference variables, it's not very illuminating.
            }
            RegionsOverlyPolymorphic(_, conc_region) => {
                self.note_and_explain_region("expected concrete lifetime is ",
                                           conc_region, "");
            }
            Sorts(values) => {
                let expected_str = values.expected.sort_string(self);
                let found_str = values.found.sort_string(self);
                if expected_str == found_str && expected_str == "closure" {
                    self.sess.span_note(sp,
                        &format!("no two closures, even if identical, have the same type"));
                    self.sess.span_help(sp,
                        &format!("consider boxing your closure and/or \
                                  using it as a trait object"));
                }
            },
            TyParamDefaultMismatch(values) => {
                let expected = values.expected;
                let found = values.found;
                self.sess.span_note(sp,
                                    &format!("conflicting type parameter defaults `{}` and `{}`",
                                             expected.ty,
                                             found.ty));

                match (expected.def_id.is_local(),
                       self.map.opt_span(expected.def_id.node)) {
                    (true, Some(span)) => {
                        self.sess.span_note(span,
                                            &format!("a default was defined here..."));
                    }
                    (_, _) => {
                        self.sess.note(
                            &format!("a default is defined on `{}`",
                                     self.item_path_str(expected.def_id)));
                    }
                }

                self.sess.span_note(
                    expected.origin_span,
                    &format!("...that was applied to an unconstrained type variable here"));

                match (found.def_id.is_local(),
                       self.map.opt_span(found.def_id.node)) {
                    (true, Some(span)) => {
                        self.sess.span_note(span,
                                            &format!("a second default was defined here..."));
                    }
                    (_, _) => {
                        self.sess.note(
                            &format!("a second default is defined on `{}`",
                                     self.item_path_str(found.def_id)));
                    }
                }

                self.sess.span_note(
                    found.origin_span,
                    &format!("...that also applies to the same type variable here"));
            }
            _ => {}
        }
    }

    pub fn provided_source(&self, id: DefId) -> Option<DefId> {
        self.provided_method_sources.borrow().get(&id).cloned()
    }

    pub fn provided_trait_methods(&self, id: DefId) -> Vec<Rc<Method<'tcx>>> {
        if id.is_local() {
            if let ItemTrait(_, _, _, ref ms) = self.map.expect_item(id.node).node {
                ms.iter().filter_map(|ti| {
                    if let ast::MethodTraitItem(_, Some(_)) = ti.node {
                        match self.impl_or_trait_item(DefId::local(ti.id)) {
                            MethodTraitItem(m) => Some(m),
                            _ => {
                                self.sess.bug("provided_trait_methods(): \
                                               non-method item found from \
                                               looking up provided method?!")
                            }
                        }
                    } else {
                        None
                    }
                }).collect()
            } else {
                self.sess.bug(&format!("provided_trait_methods: `{:?}` is not a trait", id))
            }
        } else {
            csearch::get_provided_trait_methods(self, id)
        }
    }

    pub fn associated_consts(&self, id: DefId) -> Vec<Rc<AssociatedConst<'tcx>>> {
        if id.is_local() {
            match self.map.expect_item(id.node).node {
                ItemTrait(_, _, _, ref tis) => {
                    tis.iter().filter_map(|ti| {
                        if let ast::ConstTraitItem(_, _) = ti.node {
                            match self.impl_or_trait_item(DefId::local(ti.id)) {
                                ConstTraitItem(ac) => Some(ac),
                                _ => {
                                    self.sess.bug("associated_consts(): \
                                                   non-const item found from \
                                                   looking up a constant?!")
                                }
                            }
                        } else {
                            None
                        }
                    }).collect()
                }
                ItemImpl(_, _, _, _, _, ref iis) => {
                    iis.iter().filter_map(|ii| {
                        if let ast::ConstImplItem(_, _) = ii.node {
                            match self.impl_or_trait_item(DefId::local(ii.id)) {
                                ConstTraitItem(ac) => Some(ac),
                                _ => {
                                    self.sess.bug("associated_consts(): \
                                                   non-const item found from \
                                                   looking up a constant?!")
                                }
                            }
                        } else {
                            None
                        }
                    }).collect()
                }
                _ => {
                    self.sess.bug(&format!("associated_consts: `{:?}` is not a trait \
                                            or impl", id))
                }
            }
        } else {
            csearch::get_associated_consts(self, id)
        }
    }

    pub fn trait_items(&self, trait_did: DefId) -> Rc<Vec<ImplOrTraitItem<'tcx>>> {
        let mut trait_items = self.trait_items_cache.borrow_mut();
        match trait_items.get(&trait_did).cloned() {
            Some(trait_items) => trait_items,
            None => {
                let def_ids = self.trait_item_def_ids(trait_did);
                let items: Rc<Vec<ImplOrTraitItem>> =
                    Rc::new(def_ids.iter()
                                   .map(|d| self.impl_or_trait_item(d.def_id()))
                                   .collect());
                trait_items.insert(trait_did, items.clone());
                items
            }
        }
    }

    pub fn trait_impl_polarity(&self, id: DefId) -> Option<ast::ImplPolarity> {
        if id.is_local() {
            match self.map.find(id.node) {
                Some(ast_map::NodeItem(item)) => {
                    match item.node {
                        ast::ItemImpl(_, polarity, _, _, _, _) => Some(polarity),
                        _ => None
                    }
                }
                _ => None
            }
        } else {
            csearch::get_impl_polarity(self, id)
        }
    }

    pub fn custom_coerce_unsized_kind(&self, did: DefId) -> CustomCoerceUnsized {
        memoized(&self.custom_coerce_unsized_kinds, did, |did: DefId| {
            let (kind, src) = if did.krate != LOCAL_CRATE {
                (csearch::get_custom_coerce_unsized_kind(self, did), "external")
            } else {
                (None, "local")
            };

            match kind {
                Some(kind) => kind,
                None => {
                    self.sess.bug(&format!("custom_coerce_unsized_kind: \
                                            {} impl `{}` is missing its kind",
                                           src, self.item_path_str(did)));
                }
            }
        })
    }

    pub fn impl_or_trait_item(&self, id: DefId) -> ImplOrTraitItem<'tcx> {
        lookup_locally_or_in_crate_store(
            "impl_or_trait_items", id, &self.impl_or_trait_items,
            || csearch::get_impl_or_trait_item(self, id))
    }

    pub fn trait_item_def_ids(&self, id: DefId) -> Rc<Vec<ImplOrTraitItemId>> {
        lookup_locally_or_in_crate_store(
            "trait_item_def_ids", id, &self.trait_item_def_ids,
            || Rc::new(csearch::get_trait_item_def_ids(&self.sess.cstore, id)))
    }

    /// Returns the trait-ref corresponding to a given impl, or None if it is
    /// an inherent impl.
    pub fn impl_trait_ref(&self, id: DefId) -> Option<TraitRef<'tcx>> {
        lookup_locally_or_in_crate_store(
            "impl_trait_refs", id, &self.impl_trait_refs,
            || csearch::get_impl_trait(self, id))
    }

    /// Returns whether this DefId refers to an impl
    pub fn is_impl(&self, id: DefId) -> bool {
        if id.is_local() {
            if let Some(ast_map::NodeItem(
                &ast::Item { node: ast::ItemImpl(..), .. })) = self.map.find(id.node) {
                true
            } else {
                false
            }
        } else {
            csearch::is_impl(&self.sess.cstore, id)
        }
    }

    pub fn trait_ref_to_def_id(&self, tr: &ast::TraitRef) -> DefId {
        self.def_map.borrow().get(&tr.ref_id).expect("no def-map entry for trait").def_id()
    }

    pub fn try_add_builtin_trait(&self,
                                 trait_def_id: DefId,
                                 builtin_bounds: &mut EnumSet<BuiltinBound>)
                                 -> bool
    {
        //! Checks whether `trait_ref` refers to one of the builtin
        //! traits, like `Send`, and adds the corresponding
        //! bound to the set `builtin_bounds` if so. Returns true if `trait_ref`
        //! is a builtin trait.

        match self.lang_items.to_builtin_kind(trait_def_id) {
            Some(bound) => { builtin_bounds.insert(bound); true }
            None => false
        }
    }

    pub fn item_path_str(&self, id: DefId) -> String {
        self.with_path(id, |path| ast_map::path_to_string(path))
    }

    pub fn with_path<T, F>(&self, id: DefId, f: F) -> T where
        F: FnOnce(ast_map::PathElems) -> T,
    {
        if id.is_local() {
            self.map.with_path(id.node, f)
        } else {
            f(csearch::get_item_path(self, id).iter().cloned().chain(LinkedPath::empty()))
        }
    }

    pub fn item_name(&self, id: DefId) -> ast::Name {
        if id.is_local() {
            self.map.get_path_elem(id.node).name()
        } else {
            csearch::get_item_name(self, id)
        }
    }

    /// Returns `(normalized_type, ty)`, where `normalized_type` is the
    /// IntType representation of one of {i64,i32,i16,i8,u64,u32,u16,u8},
    /// and `ty` is the original type (i.e. may include `isize` or
    /// `usize`).
    pub fn enum_repr_type(&self, opt_hint: Option<&attr::ReprAttr>)
                          -> (attr::IntType, Ty<'tcx>) {
        let repr_type = match opt_hint {
            // Feed in the given type
            Some(&attr::ReprInt(_, int_t)) => int_t,
            // ... but provide sensible default if none provided
            //
            // NB. Historically `fn enum_variants` generate i64 here, while
            // rustc_typeck::check would generate isize.
            _ => SignedInt(ast::TyIs),
        };

        let repr_type_ty = repr_type.to_ty(self);
        let repr_type = match repr_type {
            SignedInt(ast::TyIs) =>
                SignedInt(self.sess.target.int_type),
            UnsignedInt(ast::TyUs) =>
                UnsignedInt(self.sess.target.uint_type),
            other => other
        };

        (repr_type, repr_type_ty)
    }

    // Register a given item type
    pub fn register_item_type(&self, did: DefId, ty: TypeScheme<'tcx>) {
        self.tcache.borrow_mut().insert(did, ty);
    }

    // If the given item is in an external crate, looks up its type and adds it to
    // the type cache. Returns the type parameters and type.
    pub fn lookup_item_type(&self, did: DefId) -> TypeScheme<'tcx> {
        lookup_locally_or_in_crate_store(
            "tcache", did, &self.tcache,
            || csearch::get_type(self, did))
    }

    /// Given the did of a trait, returns its canonical trait ref.
    pub fn lookup_trait_def(&self, did: DefId) -> &'tcx TraitDef<'tcx> {
        lookup_locally_or_in_crate_store(
            "trait_defs", did, &self.trait_defs,
            || self.arenas.trait_defs.alloc(csearch::get_trait_def(self, did))
        )
    }

    /// Given the did of an ADT, return a master reference to its
    /// definition. Unless you are planning on fulfilling the ADT's fields,
    /// use lookup_adt_def instead.
    pub fn lookup_adt_def_master(&self, did: DefId) -> AdtDefMaster<'tcx> {
        lookup_locally_or_in_crate_store(
            "adt_defs", did, &self.adt_defs,
            || csearch::get_adt_def(self, did)
        )
    }

    /// Given the did of an ADT, return a reference to its definition.
    pub fn lookup_adt_def(&self, did: DefId) -> AdtDef<'tcx> {
        // when reverse-variance goes away, a transmute::<AdtDefMaster,AdtDef>
        // woud be needed here.
        self.lookup_adt_def_master(did)
    }

    /// Return the list of all interned ADT definitions
    pub fn adt_defs(&self) -> Vec<AdtDef<'tcx>> {
        self.adt_defs.borrow().values().cloned().collect()
    }

    /// Given the did of an item, returns its full set of predicates.
    pub fn lookup_predicates(&self, did: DefId) -> GenericPredicates<'tcx> {
        lookup_locally_or_in_crate_store(
            "predicates", did, &self.predicates,
            || csearch::get_predicates(self, did))
    }

    /// Given the did of a trait, returns its superpredicates.
    pub fn lookup_super_predicates(&self, did: DefId) -> GenericPredicates<'tcx> {
        lookup_locally_or_in_crate_store(
            "super_predicates", did, &self.super_predicates,
            || csearch::get_super_predicates(self, did))
    }

    /// Get the attributes of a definition.
    pub fn get_attrs(&self, did: DefId) -> Cow<'tcx, [ast::Attribute]> {
        if did.is_local() {
            Cow::Borrowed(self.map.attrs(did.node))
        } else {
            Cow::Owned(csearch::get_item_attrs(&self.sess.cstore, did))
        }
    }

    /// Determine whether an item is annotated with an attribute
    pub fn has_attr(&self, did: DefId, attr: &str) -> bool {
        self.get_attrs(did).iter().any(|item| item.check_name(attr))
    }

    /// Determine whether an item is annotated with `#[repr(packed)]`
    pub fn lookup_packed(&self, did: DefId) -> bool {
        self.lookup_repr_hints(did).contains(&attr::ReprPacked)
    }

    /// Determine whether an item is annotated with `#[simd]`
    pub fn lookup_simd(&self, did: DefId) -> bool {
        self.has_attr(did, "simd")
            || self.lookup_repr_hints(did).contains(&attr::ReprSimd)
    }

    /// Obtain the representation annotation for a struct definition.
    pub fn lookup_repr_hints(&self, did: DefId) -> Rc<Vec<attr::ReprAttr>> {
        memoized(&self.repr_hint_cache, did, |did: DefId| {
            Rc::new(if did.is_local() {
                self.get_attrs(did).iter().flat_map(|meta| {
                    attr::find_repr_attrs(self.sess.diagnostic(), meta).into_iter()
                }).collect()
            } else {
                csearch::get_repr_attrs(&self.sess.cstore, did)
            })
        })
    }

    /// Returns the deeply last field of nested structures, or the same type,
    /// if not a structure at all. Corresponds to the only possible unsized
    /// field, and its type can be used to determine unsizing strategy.
    pub fn struct_tail(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        while let TyStruct(def, substs) = ty.sty {
            match def.struct_variant().fields.last() {
                Some(f) => ty = f.ty(self, substs),
                None => break
            }
        }
        ty
    }

    /// Same as applying struct_tail on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    pub fn struct_lockstep_tails(&self,
                                 source: Ty<'tcx>,
                                 target: Ty<'tcx>)
                                 -> (Ty<'tcx>, Ty<'tcx>) {
        let (mut a, mut b) = (source, target);
        while let (&TyStruct(a_def, a_substs), &TyStruct(b_def, b_substs)) = (&a.sty, &b.sty) {
            if a_def != b_def {
                break;
            }
            if let Some(f) = a_def.struct_variant().fields.last() {
                a = f.ty(self, a_substs);
                b = f.ty(self, b_substs);
            } else {
                break;
            }
        }
        (a, b)
    }

    // Returns the repeat count for a repeating vector expression.
    pub fn eval_repeat_count(&self, count_expr: &ast::Expr) -> usize {
        let hint = UncheckedExprHint(self.types.usize);
        match const_eval::eval_const_expr_partial(self, count_expr, hint) {
            Ok(val) => {
                let found = match val {
                    ConstVal::Uint(count) => return count as usize,
                    ConstVal::Int(count) if count >= 0 => return count as usize,
                    const_val => const_val.description(),
                };
                span_err!(self.sess, count_expr.span, E0306,
                    "expected positive integer for repeat count, found {}",
                    found);
            }
            Err(err) => {
                let err_msg = match count_expr.node {
                    ast::ExprPath(None, ast::Path {
                        global: false,
                        ref segments,
                        ..
                    }) if segments.len() == 1 =>
                        format!("found variable"),
                    _ => match err.kind {
                        ErrKind::MiscCatchAll => format!("but found {}", err.description()),
                        _ => format!("but {}", err.description())
                    }
                };
                span_err!(self.sess, count_expr.span, E0307,
                    "expected constant integer for repeat count, {}", err_msg);
            }
        }
        0
    }

    // Iterate over a type parameter's bounded traits and any supertraits
    // of those traits, ignoring kinds.
    // Here, the supertraits are the transitive closure of the supertrait
    // relation on the supertraits from each bounded trait's constraint
    // list.
    pub fn each_bound_trait_and_supertraits<F>(&self,
                                               bounds: &[PolyTraitRef<'tcx>],
                                               mut f: F)
                                               -> bool where
        F: FnMut(PolyTraitRef<'tcx>) -> bool,
    {
        for bound_trait_ref in traits::transitive_bounds(self, bounds) {
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
    /// NB: in some cases, particularly around higher-ranked bounds,
    /// this function returns a kind of conservative approximation.
    /// That is, all regions returned by this function are definitely
    /// required, but there may be other region bounds that are not
    /// returned, as well as requirements like `for<'a> T: 'a`.
    ///
    /// Requires that trait definitions have been processed so that we can
    /// elaborate predicates and walk supertraits.
    pub fn required_region_bounds(&self,
                                  erased_self_ty: Ty<'tcx>,
                                  predicates: Vec<ty::Predicate<'tcx>>)
                                  -> Vec<ty::Region>    {
        debug!("required_region_bounds(erased_self_ty={:?}, predicates={:?})",
               erased_self_ty,
               predicates);

        assert!(!erased_self_ty.has_escaping_regions());

        traits::elaborate_predicates(self, predicates)
            .filter_map(|predicate| {
                match predicate {
                    ty::Predicate::Projection(..) |
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Equate(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
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
                            Some(r)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }

    pub fn item_variances(&self, item_id: DefId) -> Rc<ItemVariances> {
        lookup_locally_or_in_crate_store(
            "item_variance_map", item_id, &self.item_variance_map,
            || Rc::new(csearch::get_item_variances(&self.sess.cstore, item_id)))
    }

    pub fn trait_has_default_impl(&self, trait_def_id: DefId) -> bool {
        self.populate_implementations_for_trait_if_necessary(trait_def_id);

        let def = self.lookup_trait_def(trait_def_id);
        def.flags.get().intersects(TraitFlags::HAS_DEFAULT_IMPL)
    }

    /// Records a trait-to-implementation mapping.
    pub fn record_trait_has_default_impl(&self, trait_def_id: DefId) {
        let def = self.lookup_trait_def(trait_def_id);
        def.flags.set(def.flags.get() | TraitFlags::HAS_DEFAULT_IMPL)
    }

    /// Load primitive inherent implementations if necessary
    pub fn populate_implementations_for_primitive_if_necessary(&self,
                                                               primitive_def_id: DefId) {
        if primitive_def_id.is_local() {
            return
        }

        if self.populated_external_primitive_impls.borrow().contains(&primitive_def_id) {
            return
        }

        debug!("populate_implementations_for_primitive_if_necessary: searching for {:?}",
               primitive_def_id);

        let impl_items = csearch::get_impl_items(&self.sess.cstore, primitive_def_id);

        // Store the implementation info.
        self.impl_items.borrow_mut().insert(primitive_def_id, impl_items);
        self.populated_external_primitive_impls.borrow_mut().insert(primitive_def_id);
    }

    /// Populates the type context with all the inherent implementations for
    /// the given type if necessary.
    pub fn populate_inherent_implementations_for_type_if_necessary(&self,
                                                                   type_id: DefId) {
        if type_id.is_local() {
            return
        }

        if self.populated_external_types.borrow().contains(&type_id) {
            return
        }

        debug!("populate_inherent_implementations_for_type_if_necessary: searching for {:?}",
               type_id);

        let mut inherent_impls = Vec::new();
        csearch::each_inherent_implementation_for_type(&self.sess.cstore, type_id, |impl_def_id| {
            // Record the implementation.
            inherent_impls.push(impl_def_id);

            // Store the implementation info.
            let impl_items = csearch::get_impl_items(&self.sess.cstore, impl_def_id);
            self.impl_items.borrow_mut().insert(impl_def_id, impl_items);
        });

        self.inherent_impls.borrow_mut().insert(type_id, Rc::new(inherent_impls));
        self.populated_external_types.borrow_mut().insert(type_id);
    }

    /// Populates the type context with all the implementations for the given
    /// trait if necessary.
    pub fn populate_implementations_for_trait_if_necessary(&self, trait_id: DefId) {
        if trait_id.is_local() {
            return
        }

        let def = self.lookup_trait_def(trait_id);
        if def.flags.get().intersects(TraitFlags::IMPLS_VALID) {
            return;
        }

        debug!("populate_implementations_for_trait_if_necessary: searching for {:?}", def);

        if csearch::is_defaulted_trait(&self.sess.cstore, trait_id) {
            self.record_trait_has_default_impl(trait_id);
        }

        csearch::each_implementation_for_trait(&self.sess.cstore, trait_id, |impl_def_id| {
            let impl_items = csearch::get_impl_items(&self.sess.cstore, impl_def_id);
            let trait_ref = self.impl_trait_ref(impl_def_id).unwrap();
            // Record the trait->implementation mapping.
            def.record_impl(self, impl_def_id, trait_ref);

            // For any methods that use a default implementation, add them to
            // the map. This is a bit unfortunate.
            for impl_item_def_id in &impl_items {
                let method_def_id = impl_item_def_id.def_id();
                match self.impl_or_trait_item(method_def_id) {
                    MethodTraitItem(method) => {
                        if let Some(source) = method.provided_source {
                            self.provided_method_sources
                                .borrow_mut()
                                .insert(method_def_id, source);
                        }
                    }
                    _ => {}
                }
            }

            // Store the implementation info.
            self.impl_items.borrow_mut().insert(impl_def_id, impl_items);
        });

        def.flags.set(def.flags.get() | TraitFlags::IMPLS_VALID);
    }

    /// Given the def_id of an impl, return the def_id of the trait it implements.
    /// If it implements no trait, return `None`.
    pub fn trait_id_of_impl(&self, def_id: DefId) -> Option<DefId> {
        self.impl_trait_ref(def_id).map(|tr| tr.def_id)
    }

    /// If the given def ID describes a method belonging to an impl, return the
    /// ID of the impl that the method belongs to. Otherwise, return `None`.
    pub fn impl_of_method(&self, def_id: DefId) -> Option<DefId> {
        if def_id.krate != LOCAL_CRATE {
            return match csearch::get_impl_or_trait_item(self,
                                                         def_id).container() {
                TraitContainer(_) => None,
                ImplContainer(def_id) => Some(def_id),
            };
        }
        match self.impl_or_trait_items.borrow().get(&def_id).cloned() {
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
    pub fn trait_of_item(&self, def_id: DefId) -> Option<DefId> {
        if def_id.krate != LOCAL_CRATE {
            return csearch::get_trait_of_item(&self.sess.cstore, def_id, self);
        }
        match self.impl_or_trait_items.borrow().get(&def_id).cloned() {
            Some(impl_or_trait_item) => {
                match impl_or_trait_item.container() {
                    TraitContainer(def_id) => Some(def_id),
                    ImplContainer(def_id) => self.trait_id_of_impl(def_id),
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
    pub fn trait_item_of_item(&self, def_id: DefId) -> Option<ImplOrTraitItemId> {
        let impl_item = match self.impl_or_trait_items.borrow().get(&def_id) {
            Some(m) => m.clone(),
            None => return None,
        };
        let name = impl_item.name();
        match self.trait_of_item(def_id) {
            Some(trait_did) => {
                self.trait_items(trait_did).iter()
                    .find(|item| item.name() == name)
                    .map(|item| item.id())
            }
            None => None
        }
    }

    /// Creates a hash of the type `Ty` which will be the same no matter what crate
    /// context it's calculated within. This is used by the `type_id` intrinsic.
    pub fn hash_crate_independent(&self, ty: Ty<'tcx>, svh: &Svh) -> u64 {
        let mut state = SipHasher::new();
        helper(self, ty, svh, &mut state);
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
                    ReVar(..) |
                    ReSkolemized(..) => {
                        tcx.sess.bug("unexpected region found when hashing a type")
                    }
                }
            };
            let did = |state: &mut SipHasher, did: DefId| {
                let h = if did.is_local() {
                    svh.clone()
                } else {
                    tcx.sess.cstore.get_crate_hash(did.krate)
                };
                h.as_str().hash(state);
                did.node.hash(state);
            };
            let mt = |state: &mut SipHasher, mt: TypeAndMut| {
                mt.mutbl.hash(state);
            };
            let fn_sig = |state: &mut SipHasher, sig: &Binder<FnSig<'tcx>>| {
                let sig = tcx.anonymize_late_bound_regions(sig).0;
                for a in &sig.inputs { helper(tcx, *a, svh, state); }
                if let ty::FnConverging(output) = sig.output {
                    helper(tcx, output, svh, state);
                }
            };
            ty.maybe_walk(|ty| {
                match ty.sty {
                    TyBool => byte!(2),
                    TyChar => byte!(3),
                    TyInt(i) => {
                        byte!(4);
                        hash!(i);
                    }
                    TyUint(u) => {
                        byte!(5);
                        hash!(u);
                    }
                    TyFloat(f) => {
                        byte!(6);
                        hash!(f);
                    }
                    TyStr => {
                        byte!(7);
                    }
                    TyEnum(d, _) => {
                        byte!(8);
                        did(state, d.did);
                    }
                    TyBox(_) => {
                        byte!(9);
                    }
                    TyArray(_, n) => {
                        byte!(10);
                        n.hash(state);
                    }
                    TySlice(_) => {
                        byte!(11);
                    }
                    TyRawPtr(m) => {
                        byte!(12);
                        mt(state, m);
                    }
                    TyRef(r, m) => {
                        byte!(13);
                        region(state, *r);
                        mt(state, m);
                    }
                    TyBareFn(opt_def_id, ref b) => {
                        byte!(14);
                        hash!(opt_def_id);
                        hash!(b.unsafety);
                        hash!(b.abi);
                        fn_sig(state, &b.sig);
                        return false;
                    }
                    TyTrait(ref data) => {
                        byte!(17);
                        did(state, data.principal_def_id());
                        hash!(data.bounds);

                        let principal = tcx.anonymize_late_bound_regions(&data.principal).0;
                        for subty in &principal.substs.types {
                            helper(tcx, subty, svh, state);
                        }

                        return false;
                    }
                    TyStruct(d, _) => {
                        byte!(18);
                        did(state, d.did);
                    }
                    TyTuple(ref inner) => {
                        byte!(19);
                        hash!(inner.len());
                    }
                    TyParam(p) => {
                        byte!(20);
                        hash!(p.space);
                        hash!(p.idx);
                        hash!(p.name.as_str());
                    }
                    TyInfer(_) => unreachable!(),
                    TyError => byte!(21),
                    TyClosure(d, _) => {
                        byte!(22);
                        did(state, d);
                    }
                    TyProjection(ref data) => {
                        byte!(23);
                        did(state, data.trait_ref.def_id);
                        hash!(data.item_name.as_str());
                    }
                }
                true
            });
        }
    }

    /// Construct a parameter environment suitable for static contexts or other contexts where there
    /// are no free type/lifetime parameters in scope.
    pub fn empty_parameter_environment<'a>(&'a self)
                                           -> ParameterEnvironment<'a,'tcx> {
        ty::ParameterEnvironment { tcx: self,
                                   free_substs: Substs::empty(),
                                   caller_bounds: Vec::new(),
                                   implicit_region_bound: ty::ReEmpty,
                                   selection_cache: traits::SelectionCache::new(),

                                   // for an empty parameter
                                   // environment, there ARE no free
                                   // regions, so it shouldn't matter
                                   // what we use for the free id
                                   free_id: ast::DUMMY_NODE_ID }
    }

    /// Constructs and returns a substitution that can be applied to move from
    /// the "outer" view of a type or method to the "inner" view.
    /// In general, this means converting from bound parameters to
    /// free parameters. Since we currently represent bound/free type
    /// parameters in the same way, this only has an effect on regions.
    pub fn construct_free_substs(&self, generics: &Generics<'tcx>,
                                 free_id: ast::NodeId) -> Substs<'tcx> {
        // map T => T
        let mut types = VecPerParamSpace::empty();
        for def in generics.types.as_slice() {
            debug!("construct_parameter_environment(): push_types_from_defs: def={:?}",
                    def);
            types.push(def.space, self.mk_param_from_def(def));
        }

        let free_id_outlive = self.region_maps.item_extent(free_id);

        // map bound 'a => free 'a
        let mut regions = VecPerParamSpace::empty();
        for def in generics.regions.as_slice() {
            let region =
                ReFree(FreeRegion { scope: free_id_outlive,
                                    bound_region: BrNamed(def.def_id, def.name) });
            debug!("push_region_params {:?}", region);
            regions.push(def.space, region);
        }

        Substs {
            types: types,
            regions: subst::NonerasedRegions(regions)
        }
    }

    /// See `ParameterEnvironment` struct def'n for details
    pub fn construct_parameter_environment<'a>(&'a self,
                                               span: Span,
                                               generics: &ty::Generics<'tcx>,
                                               generic_predicates: &ty::GenericPredicates<'tcx>,
                                               free_id: ast::NodeId)
                                               -> ParameterEnvironment<'a, 'tcx>
    {
        //
        // Construct the free substs.
        //

        let free_substs = self.construct_free_substs(generics, free_id);
        let free_id_outlive = self.region_maps.item_extent(free_id);

        //
        // Compute the bounds on Self and the type parameters.
        //

        let bounds = generic_predicates.instantiate(self, &free_substs);
        let bounds = self.liberate_late_bound_regions(free_id_outlive, &ty::Binder(bounds));
        let predicates = bounds.predicates.into_vec();

        debug!("construct_parameter_environment: free_id={:?} free_subst={:?} predicates={:?}",
               free_id,
               free_substs,
               predicates);

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
            tcx: self,
            free_substs: free_substs,
            implicit_region_bound: ty::ReScope(free_id_outlive),
            caller_bounds: predicates,
            selection_cache: traits::SelectionCache::new(),
            free_id: free_id,
        };

        let cause = traits::ObligationCause::misc(span, free_id);
        traits::normalize_param_env_or_error(unnormalized_env, cause)
    }

    pub fn is_method_call(&self, expr_id: ast::NodeId) -> bool {
        self.tables.borrow().method_map.contains_key(&MethodCall::expr(expr_id))
    }

    pub fn is_overloaded_autoderef(&self, expr_id: ast::NodeId, autoderefs: u32) -> bool {
        self.tables.borrow().method_map.contains_key(&MethodCall::autoderef(expr_id,
                                                                            autoderefs))
    }

    pub fn upvar_capture(&self, upvar_id: ty::UpvarId) -> Option<ty::UpvarCapture> {
        Some(self.tables.borrow().upvar_capture_map.get(&upvar_id).unwrap().clone())
    }


    /// Returns true if this ADT is a dtorck type, i.e. whether it being
    /// safe for destruction requires it to be alive
    fn is_adt_dtorck(&self, adt: AdtDef<'tcx>) -> bool {
        let dtor_method = match adt.destructor() {
            Some(dtor) => dtor,
            None => return false
        };
        let impl_did = self.impl_of_method(dtor_method).unwrap_or_else(|| {
            self.sess.bug(&format!("no Drop impl for the dtor of `{:?}`", adt))
        });
        let generics = adt.type_scheme(self).generics;

        // In `impl<'a> Drop ...`, we automatically assume
        // `'a` is meaningful and thus represents a bound
        // through which we could reach borrowed data.
        //
        // FIXME (pnkfelix): In the future it would be good to
        // extend the language to allow the user to express,
        // in the impl signature, that a lifetime is not
        // actually used (something like `where 'a: ?Live`).
        if generics.has_region_params(subst::TypeSpace) {
            debug!("typ: {:?} has interesting dtor due to region params",
                   adt);
            return true;
        }

        let mut seen_items = Vec::new();
        let mut items_to_inspect = vec![impl_did];
        while let Some(item_def_id) = items_to_inspect.pop() {
            if seen_items.contains(&item_def_id) {
                continue;
            }

            for pred in self.lookup_predicates(item_def_id).predicates {
                let result = match pred {
                    ty::Predicate::Equate(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::TypeOutlives(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::Projection(..) => {
                        // For now, assume all these where-clauses
                        // may give drop implementation capabilty
                        // to access borrowed data.
                        true
                    }

                    ty::Predicate::Trait(ty::Binder(ref t_pred)) => {
                        let def_id = t_pred.trait_ref.def_id;
                        if self.trait_items(def_id).len() != 0 {
                            // If trait has items, assume it adds
                            // capability to access borrowed data.
                            true
                        } else {
                            // Trait without items is itself
                            // uninteresting from POV of dropck.
                            //
                            // However, may have parent w/ items;
                            // so schedule checking of predicates,
                            items_to_inspect.push(def_id);
                            // and say "no capability found" for now.
                            false
                        }
                    }
                };

                if result {
                    debug!("typ: {:?} has interesting dtor due to generic preds, e.g. {:?}",
                           adt, pred);
                    return true;
                }
            }

            seen_items.push(item_def_id);
        }

        debug!("typ: {:?} is dtorck-safe", adt);
        false
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

impl<'tcx> AutoAdjustment<'tcx> {
    pub fn is_identity(&self) -> bool {
        match *self {
            AdjustReifyFnPointer |
            AdjustUnsafeFnPointer => false,
            AdjustDerefRef(ref r) => r.is_identity(),
        }
    }
}

impl<'tcx> AutoDerefRef<'tcx> {
    pub fn is_identity(&self) -> bool {
        self.autoderefs == 0 && self.unsize.is_none() && self.autoref.is_none()
    }
}

impl<'tcx> ctxt<'tcx> {
    pub fn with_freevars<T, F>(&self, fid: ast::NodeId, f: F) -> T where
        F: FnOnce(&[Freevar]) -> T,
    {
        match self.freevars.borrow().get(&fid) {
            None => f(&[]),
            Some(d) => f(&d[..])
        }
    }

    /// Replace any late-bound regions bound in `value` with free variants attached to scope-id
    /// `scope_id`.
    pub fn liberate_late_bound_regions<T>(&self,
        all_outlive_scope: region::CodeExtent,
        value: &Binder<T>)
        -> T
        where T : TypeFoldable<'tcx>
    {
        ty_fold::replace_late_bound_regions(
            self, value,
            |br| ty::ReFree(ty::FreeRegion{scope: all_outlive_scope, bound_region: br})).0
    }

    /// Flattens two binding levels into one. So `for<'a> for<'b> Foo`
    /// becomes `for<'a,'b> Foo`.
    pub fn flatten_late_bound_regions<T>(&self, bound2_value: &Binder<Binder<T>>)
                                         -> Binder<T>
        where T: TypeFoldable<'tcx>
    {
        let bound0_value = bound2_value.skip_binder().skip_binder();
        let value = ty_fold::fold_regions(self, bound0_value, &mut false,
                                          |region, current_depth| {
            match region {
                ty::ReLateBound(debruijn, br) if debruijn.depth >= current_depth => {
                    // should be true if no escaping regions from bound2_value
                    assert!(debruijn.depth - current_depth <= 1);
                    ty::ReLateBound(DebruijnIndex::new(current_depth), br)
                }
                _ => {
                    region
                }
            }
        });
        Binder(value)
    }

    pub fn no_late_bound_regions<T>(&self, value: &Binder<T>) -> Option<T>
        where T : TypeFoldable<'tcx> + RegionEscape
    {
        if value.0.has_escaping_regions() {
            None
        } else {
            Some(value.0.clone())
        }
    }

    /// Replace any late-bound regions bound in `value` with `'static`. Useful in trans but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(&self, value: &Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        ty_fold::replace_late_bound_regions(self, value, |_| ty::ReStatic).0
    }

    /// Rewrite any late-bound regions so that they are anonymous.  Region numbers are
    /// assigned starting at 1 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical.  For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(&self, sig: &Binder<T>) -> Binder<T>
        where T : TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        ty::Binder(ty_fold::replace_late_bound_regions(self, sig, |_| {
            counter += 1;
            ReLateBound(ty::DebruijnIndex::new(1), BrAnon(counter))
        }).0)
    }

    pub fn make_substs_for_receiver_types(&self,
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
                  .map(|def| self.mk_param_from_def(def))
                  .collect();
        let meth_regions: Vec<ty::Region> =
            method.generics.regions.get_slice(subst::FnSpace)
                  .iter()
                  .map(|def| def.to_early_bound_region())
                  .collect();
        trait_ref.substs.clone().with_method(meth_tps, meth_regions)
    }
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

impl<'tcx> fmt::Debug for AutoAdjustment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            AdjustReifyFnPointer => {
                write!(f, "AdjustReifyFnPointer")
            }
            AdjustUnsafeFnPointer => {
                write!(f, "AdjustUnsafeFnPointer")
            }
            AdjustDerefRef(ref data) => {
                write!(f, "{:?}", data)
            }
        }
    }
}

impl<'tcx> fmt::Debug for AutoDerefRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AutoDerefRef({}, unsize={:?}, {:?})",
               self.autoderefs, self.unsize, self.autoref)
    }
}

impl<'tcx> fmt::Debug for TraitTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TraitTy({:?},{:?})",
               self.principal,
               self.bounds)
    }
}

impl<'tcx> fmt::Debug for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Predicate::Trait(ref a) => write!(f, "{:?}", a),
            Predicate::Equate(ref pair) => write!(f, "{:?}", pair),
            Predicate::RegionOutlives(ref pair) => write!(f, "{:?}", pair),
            Predicate::TypeOutlives(ref pair) => write!(f, "{:?}", pair),
            Predicate::Projection(ref pair) => write!(f, "{:?}", pair),
            Predicate::WellFormed(ty) => write!(f, "WF({:?})", ty),
            Predicate::ObjectSafe(trait_def_id) => write!(f, "ObjectSafe({:?})", trait_def_id),
        }
    }
}

// FIXME(#20298) -- all of these traits basically walk various
// structures to test whether types/regions are reachable with various
// properties. It should be possible to express them in terms of one
// common "walker" trait or something.

/// An "escaping region" is a bound region whose binder is not part of `t`.
///
/// So, for example, consider a type like the following, which has two binders:
///
///    for<'a> fn(x: for<'b> fn(&'a isize, &'b isize))
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
pub trait RegionEscape {
    fn has_escaping_regions(&self) -> bool {
        self.has_regions_escaping_depth(0)
    }

    fn has_regions_escaping_depth(&self, depth: u32) -> bool;
}

impl<'tcx> RegionEscape for Ty<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.region_depth > depth
    }
}

impl<'tcx> RegionEscape for TraitTy<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.principal.has_regions_escaping_depth(depth) ||
            self.bounds.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ExistentialBounds<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.region_bound.has_regions_escaping_depth(depth) ||
            self.projection_bounds.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for Substs<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.types.has_regions_escaping_depth(depth) ||
            self.regions.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ClosureSubsts<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.func_substs.has_regions_escaping_depth(depth) ||
            self.upvar_tys.iter().any(|t| t.has_regions_escaping_depth(depth))
    }
}

impl<T:RegionEscape> RegionEscape for Vec<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.iter().any(|t| t.has_regions_escaping_depth(depth))
    }
}

impl<'tcx> RegionEscape for FnSig<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.inputs.has_regions_escaping_depth(depth) ||
            self.output.has_regions_escaping_depth(depth)
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
            Predicate::WellFormed(ty) => ty.has_regions_escaping_depth(depth),
            Predicate::ObjectSafe(_trait_def_id) => false,
        }
    }
}

impl<'tcx,P:RegionEscape> RegionEscape for traits::Obligation<'tcx,P> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.predicate.has_regions_escaping_depth(depth)
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

impl<'tcx> RegionEscape for FnOutput<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            FnConverging(t) => t.has_regions_escaping_depth(depth),
            FnDiverging => false
        }
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

pub trait HasTypeFlags {
    fn has_type_flags(&self, flags: TypeFlags) -> bool;
    fn has_projection_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PROJECTION)
    }
    fn references_error(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_ERR)
    }
    fn has_param_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PARAMS)
    }
    fn has_self_ty(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_SELF)
    }
    fn has_infer_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER)
    }
    fn needs_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_RE_INFER)
    }
    fn needs_subst(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_SUBST)
    }
    fn has_closure_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_CLOSURE)
    }
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_EARLY_BOUND |
                            TypeFlags::HAS_RE_INFER |
                            TypeFlags::HAS_FREE_REGIONS)
    }
    /// Indicates whether this value references only 'global'
    /// types/lifetimes that are the same regardless of what fn we are
    /// in. This is used for caching. Errs on the side of returning
    /// false.
    fn is_global(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_LOCAL_NAMES)
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for Vec<T> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self[..].has_type_flags(flags)
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for [T] {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|p| p.has_type_flags(flags))
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for VecPerParamSpace<T> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|p| p.has_type_flags(flags))
    }
}

impl HasTypeFlags for abi::Abi {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl HasTypeFlags for ast::Unsafety {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl HasTypeFlags for BuiltinBounds {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl<'tcx> HasTypeFlags for ClosureTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.sig.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ClosureUpvar<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ExistentialBounds<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.projection_bounds.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::InstantiatedPredicates<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.predicates.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for Predicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        match *self {
            Predicate::Trait(ref data) => data.has_type_flags(flags),
            Predicate::Equate(ref data) => data.has_type_flags(flags),
            Predicate::RegionOutlives(ref data) => data.has_type_flags(flags),
            Predicate::TypeOutlives(ref data) => data.has_type_flags(flags),
            Predicate::Projection(ref data) => data.has_type_flags(flags),
            Predicate::WellFormed(data) => data.has_type_flags(flags),
            Predicate::ObjectSafe(_trait_def_id) => false,
        }
    }
}

impl<'tcx> HasTypeFlags for TraitPredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.trait_ref.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for EquatePredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags) || self.1.has_type_flags(flags)
    }
}

impl HasTypeFlags for Region {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        if flags.intersects(TypeFlags::HAS_LOCAL_NAMES) {
            // does this represent a region that cannot be named in a global
            // way? used in fulfillment caching.
            match *self {
                ty::ReStatic | ty::ReEmpty => {}
                _ => return true
            }
        }
        if flags.intersects(TypeFlags::HAS_RE_INFER) {
            match *self {
                ty::ReVar(_) | ty::ReSkolemized(..) => { return true }
                _ => {}
            }
        }
        false
    }
}

impl<T:HasTypeFlags,U:HasTypeFlags> HasTypeFlags for OutlivesPredicate<T,U> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags) || self.1.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ProjectionPredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.projection_ty.has_type_flags(flags) || self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ProjectionTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.trait_ref.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for Ty<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.flags.get().intersects(flags)
    }
}

impl<'tcx> HasTypeFlags for TypeAndMut<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for TraitRef<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.substs.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for subst::Substs<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.types.has_type_flags(flags) || match self.regions {
            subst::ErasedRegions => false,
            subst::NonerasedRegions(ref r) => r.has_type_flags(flags)
        }
    }
}

impl<'tcx,T> HasTypeFlags for Option<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|t| t.has_type_flags(flags))
    }
}

impl<'tcx,T> HasTypeFlags for Rc<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        (**self).has_type_flags(flags)
    }
}

impl<'tcx,T> HasTypeFlags for Box<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        (**self).has_type_flags(flags)
    }
}

impl<T> HasTypeFlags for Binder<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for FnOutput<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        match *self {
            FnConverging(t) => t.has_type_flags(flags),
            FnDiverging => false,
        }
    }
}

impl<'tcx> HasTypeFlags for FnSig<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.inputs.iter().any(|t| t.has_type_flags(flags)) ||
            self.output.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for BareFnTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.sig.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ClosureSubsts<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.func_substs.has_type_flags(flags) ||
            self.upvar_tys.iter().any(|t| t.has_type_flags(flags))
    }
}

impl<'tcx> fmt::Debug for ClosureTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ClosureTy({},{:?},{})",
               self.unsafety,
               self.sig,
               self.abi)
    }
}

impl<'tcx> fmt::Debug for ClosureUpvar<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ClosureUpvar({:?},{:?})",
               self.def,
               self.ty)
    }
}

impl<'a, 'tcx> fmt::Debug for ParameterEnvironment<'a, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParameterEnvironment(\
            free_substs={:?}, \
            implicit_region_bound={:?}, \
            caller_bounds={:?})",
            self.free_substs,
            self.implicit_region_bound,
            self.caller_bounds)
    }
}

impl<'tcx> fmt::Debug for ObjectLifetimeDefault {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ObjectLifetimeDefault::Ambiguous => write!(f, "Ambiguous"),
            ObjectLifetimeDefault::BaseDefault => write!(f, "BaseDefault"),
            ObjectLifetimeDefault::Specific(ref r) => write!(f, "{:?}", r),
        }
    }
}
