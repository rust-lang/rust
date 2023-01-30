//! Type context book-keeping.

#![allow(rustc::usage_of_ty_tykind)]

pub mod tls;

use crate::arena::Arena;
use crate::dep_graph::{DepGraph, DepKindStruct};
use crate::infer::canonical::{CanonicalVarInfo, CanonicalVarInfos};
use crate::lint::struct_lint_level;
use crate::middle::codegen_fn_attrs::CodegenFnAttrs;
use crate::middle::resolve_lifetime;
use crate::middle::stability;
use crate::mir::interpret::{self, Allocation, ConstAllocation};
use crate::mir::{
    Body, BorrowCheckResult, Field, Local, Place, PlaceElem, ProjectionKind, Promoted,
};
use crate::thir::Thir;
use crate::traits;
use crate::ty::query::{self, TyCtxtAt};
use crate::ty::{
    self, AdtDef, AdtDefData, AdtKind, Binder, Const, ConstData, DefIdTree, FloatTy, FloatVar,
    FloatVid, GenericParamDefKind, ImplPolarity, InferTy, IntTy, IntVar, IntVid, List, ParamConst,
    ParamTy, PolyExistentialPredicate, PolyFnSig, Predicate, PredicateKind, Region, RegionKind,
    ReprOptions, TraitObjectVisitor, Ty, TyKind, TyVar, TyVid, TypeAndMut, TypeckResults, UintTy,
    Visibility,
};
use crate::ty::{GenericArg, InternalSubsts, SubstsRef};
use rustc_ast as ast;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sharded::{IntoPointer, ShardedHashMap};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{self, Lock, Lrc, ReadGuard, WorkerLocal};
use rustc_errors::{
    DecorateLint, DiagnosticBuilder, DiagnosticMessage, ErrorGuaranteed, MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::definitions::Definitions;
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{
    Constness, ExprKind, HirId, ImplItemKind, ItemKind, Node, TraitCandidate, TraitItemKind,
};
use rustc_index::vec::IndexVec;
use rustc_macros::HashStable;
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use rustc_session::config::CrateType;
use rustc_session::cstore::{CrateStoreDyn, Untracked};
use rustc_session::lint::Lint;
use rustc_session::Limit;
use rustc_session::Session;
use rustc_span::def_id::{DefPathHash, StableCrateId};
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Layout, LayoutS, TargetDataLayout, VariantIdx};
use rustc_target::spec::abi;
use rustc_type_ir::sty::TyKind::*;
use rustc_type_ir::WithCachedTypeInfo;
use rustc_type_ir::{DynKind, InternAs, InternIteratorElement, Interner, TypeFlags};

use std::any::Any;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::ops::{Bound, Deref};

const TINY_CONST_EVAL_LIMIT: Limit = Limit(20);

pub trait OnDiskCache<'tcx>: rustc_data_structures::sync::Sync {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    fn new(sess: &'tcx Session, data: Mmap, start_pos: usize) -> Self
    where
        Self: Sized;

    fn new_empty(source_map: &'tcx SourceMap) -> Self
    where
        Self: Sized;

    fn drop_serialized_data(&self, tcx: TyCtxt<'tcx>);

    fn serialize(&self, tcx: TyCtxt<'tcx>, encoder: FileEncoder) -> FileEncodeResult;
}

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx> Interner for TyCtxt<'tcx> {
    type AdtDef = ty::AdtDef<'tcx>;
    type SubstsRef = ty::SubstsRef<'tcx>;
    type DefId = DefId;
    type Ty = Ty<'tcx>;
    type Const = ty::Const<'tcx>;
    type Region = Region<'tcx>;
    type TypeAndMut = TypeAndMut<'tcx>;
    type Mutability = hir::Mutability;
    type Movability = hir::Movability;
    type PolyFnSig = PolyFnSig<'tcx>;
    type ListBinderExistentialPredicate = &'tcx List<PolyExistentialPredicate<'tcx>>;
    type BinderListTy = Binder<'tcx, &'tcx List<Ty<'tcx>>>;
    type ListTy = &'tcx List<Ty<'tcx>>;
    type AliasTy = ty::AliasTy<'tcx>;
    type ParamTy = ParamTy;
    type BoundTy = ty::BoundTy;
    type PlaceholderType = ty::PlaceholderType;
    type InferTy = InferTy;
    type ErrorGuaranteed = ErrorGuaranteed;
    type PredicateKind = ty::PredicateKind<'tcx>;
    type AllocId = crate::mir::interpret::AllocId;

    type EarlyBoundRegion = ty::EarlyBoundRegion;
    type BoundRegion = ty::BoundRegion;
    type FreeRegion = ty::FreeRegion;
    type RegionVid = ty::RegionVid;
    type PlaceholderRegion = ty::PlaceholderRegion;
}

type InternedSet<'tcx, T> = ShardedHashMap<InternedInSet<'tcx, T>, ()>;

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc. are allocated from.
    arena: &'tcx WorkerLocal<Arena<'tcx>>,

    // Specifically use a speedy hash algorithm for these hash sets, since
    // they're accessed quite often.
    type_: InternedSet<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>,
    const_lists: InternedSet<'tcx, List<ty::Const<'tcx>>>,
    substs: InternedSet<'tcx, InternalSubsts<'tcx>>,
    canonical_var_infos: InternedSet<'tcx, List<CanonicalVarInfo<'tcx>>>,
    region: InternedSet<'tcx, RegionKind<'tcx>>,
    poly_existential_predicates: InternedSet<'tcx, List<PolyExistentialPredicate<'tcx>>>,
    predicate: InternedSet<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
    predicates: InternedSet<'tcx, List<Predicate<'tcx>>>,
    projs: InternedSet<'tcx, List<ProjectionKind>>,
    place_elems: InternedSet<'tcx, List<PlaceElem<'tcx>>>,
    const_: InternedSet<'tcx, ConstData<'tcx>>,
    const_allocation: InternedSet<'tcx, Allocation>,
    bound_variable_kinds: InternedSet<'tcx, List<ty::BoundVariableKind>>,
    layout: InternedSet<'tcx, LayoutS<VariantIdx>>,
    adt_def: InternedSet<'tcx, AdtDefData>,
}

impl<'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx WorkerLocal<Arena<'tcx>>) -> CtxtInterners<'tcx> {
        CtxtInterners {
            arena,
            type_: Default::default(),
            const_lists: Default::default(),
            substs: Default::default(),
            region: Default::default(),
            poly_existential_predicates: Default::default(),
            canonical_var_infos: Default::default(),
            predicate: Default::default(),
            predicates: Default::default(),
            projs: Default::default(),
            place_elems: Default::default(),
            const_: Default::default(),
            const_allocation: Default::default(),
            bound_variable_kinds: Default::default(),
            layout: Default::default(),
            adt_def: Default::default(),
        }
    }

    /// Interns a type.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline(never)]
    fn intern_ty(&self, kind: TyKind<'tcx>, sess: &Session, untracked: &Untracked) -> Ty<'tcx> {
        Ty(Interned::new_unchecked(
            self.type_
                .intern(kind, |kind| {
                    let flags = super::flags::FlagComputation::for_kind(&kind);
                    let stable_hash = self.stable_hash(&flags, sess, untracked, &kind);

                    InternedInSet(self.arena.alloc(WithCachedTypeInfo {
                        internee: kind,
                        stable_hash,
                        flags: flags.flags,
                        outer_exclusive_binder: flags.outer_exclusive_binder,
                    }))
                })
                .0,
        ))
    }

    fn stable_hash<'a, T: HashStable<StableHashingContext<'a>>>(
        &self,
        flags: &ty::flags::FlagComputation,
        sess: &'a Session,
        untracked: &'a Untracked,
        val: &T,
    ) -> Fingerprint {
        // It's impossible to hash inference variables (and will ICE), so we don't need to try to cache them.
        // Without incremental, we rarely stable-hash types, so let's not do it proactively.
        if flags.flags.intersects(TypeFlags::NEEDS_INFER) || sess.opts.incremental.is_none() {
            Fingerprint::ZERO
        } else {
            let mut hasher = StableHasher::new();
            let mut hcx = StableHashingContext::new(sess, untracked);
            val.hash_stable(&mut hcx, &mut hasher);
            hasher.finish()
        }
    }

    #[inline(never)]
    fn intern_predicate(
        &self,
        kind: Binder<'tcx, PredicateKind<'tcx>>,
        sess: &Session,
        untracked: &Untracked,
    ) -> Predicate<'tcx> {
        Predicate(Interned::new_unchecked(
            self.predicate
                .intern(kind, |kind| {
                    let flags = super::flags::FlagComputation::for_predicate(kind);

                    let stable_hash = self.stable_hash(&flags, sess, untracked, &kind);

                    InternedInSet(self.arena.alloc(WithCachedTypeInfo {
                        internee: kind,
                        stable_hash,
                        flags: flags.flags,
                        outer_exclusive_binder: flags.outer_exclusive_binder,
                    }))
                })
                .0,
        ))
    }
}

pub struct CommonTypes<'tcx> {
    pub unit: Ty<'tcx>,
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub isize: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub i128: Ty<'tcx>,
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub u128: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub str_: Ty<'tcx>,
    pub never: Ty<'tcx>,
    pub self_param: Ty<'tcx>,

    /// Dummy type used for the `Self` of a `TraitRef` created for converting
    /// a trait object, and which gets removed in `ExistentialTraitRef`.
    /// This type must not appear anywhere in other converted types.
    pub trait_object_dummy_self: Ty<'tcx>,
}

pub struct CommonLifetimes<'tcx> {
    /// `ReStatic`
    pub re_static: Region<'tcx>,

    /// Erased region, used outside of type inference.
    pub re_erased: Region<'tcx>,
}

pub struct CommonConsts<'tcx> {
    pub unit: Const<'tcx>,
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(
        interners: &CtxtInterners<'tcx>,
        sess: &Session,
        untracked: &Untracked,
    ) -> CommonTypes<'tcx> {
        let mk = |ty| interners.intern_ty(ty, sess, untracked);

        CommonTypes {
            unit: mk(Tuple(List::empty())),
            bool: mk(Bool),
            char: mk(Char),
            never: mk(Never),
            isize: mk(Int(ty::IntTy::Isize)),
            i8: mk(Int(ty::IntTy::I8)),
            i16: mk(Int(ty::IntTy::I16)),
            i32: mk(Int(ty::IntTy::I32)),
            i64: mk(Int(ty::IntTy::I64)),
            i128: mk(Int(ty::IntTy::I128)),
            usize: mk(Uint(ty::UintTy::Usize)),
            u8: mk(Uint(ty::UintTy::U8)),
            u16: mk(Uint(ty::UintTy::U16)),
            u32: mk(Uint(ty::UintTy::U32)),
            u64: mk(Uint(ty::UintTy::U64)),
            u128: mk(Uint(ty::UintTy::U128)),
            f32: mk(Float(ty::FloatTy::F32)),
            f64: mk(Float(ty::FloatTy::F64)),
            str_: mk(Str),
            self_param: mk(ty::Param(ty::ParamTy { index: 0, name: kw::SelfUpper })),

            trait_object_dummy_self: mk(Infer(ty::FreshTy(0))),
        }
    }
}

impl<'tcx> CommonLifetimes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonLifetimes<'tcx> {
        let mk = |r| {
            Region(Interned::new_unchecked(
                interners.region.intern(r, |r| InternedInSet(interners.arena.alloc(r))).0,
            ))
        };

        CommonLifetimes { re_static: mk(ty::ReStatic), re_erased: mk(ty::ReErased) }
    }
}

impl<'tcx> CommonConsts<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>, types: &CommonTypes<'tcx>) -> CommonConsts<'tcx> {
        let mk_const = |c| {
            Const(Interned::new_unchecked(
                interners.const_.intern(c, |c| InternedInSet(interners.arena.alloc(c))).0,
            ))
        };

        CommonConsts {
            unit: mk_const(ty::ConstData {
                kind: ty::ConstKind::Value(ty::ValTree::zst()),
                ty: types.unit,
            }),
        }
    }
}

/// This struct contains information regarding the `ReFree(FreeRegion)` corresponding to a lifetime
/// conflict.
#[derive(Debug)]
pub struct FreeRegionInfo {
    /// `LocalDefId` corresponding to FreeRegion
    pub def_id: LocalDefId,
    /// the bound region corresponding to FreeRegion
    pub boundregion: ty::BoundRegionKind,
    /// checks if bound region is in Impl Item
    pub is_impl_item: bool,
}

/// This struct should only be created by `create_def`.
#[derive(Copy, Clone)]
pub struct TyCtxtFeed<'tcx, KEY: Copy> {
    pub tcx: TyCtxt<'tcx>,
    // Do not allow direct access, as downstream code must not mutate this field.
    key: KEY,
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn feed_unit_query(self) -> TyCtxtFeed<'tcx, ()> {
        TyCtxtFeed { tcx: self, key: () }
    }
    pub fn feed_local_crate(self) -> TyCtxtFeed<'tcx, CrateNum> {
        TyCtxtFeed { tcx: self, key: LOCAL_CRATE }
    }
}

impl<'tcx, KEY: Copy> TyCtxtFeed<'tcx, KEY> {
    #[inline(always)]
    pub fn key(&self) -> KEY {
        self.key
    }
}

impl<'tcx> TyCtxtFeed<'tcx, LocalDefId> {
    #[inline(always)]
    pub fn def_id(&self) -> LocalDefId {
        self.key
    }
}

/// The central data structure of the compiler. It stores references
/// to the various **arenas** and also houses the results of the
/// various **compiler queries** that have been performed. See the
/// [rustc dev guide] for more details.
///
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/ty.html
#[derive(Copy, Clone)]
#[rustc_diagnostic_item = "TyCtxt"]
#[rustc_pass_by_value]
pub struct TyCtxt<'tcx> {
    gcx: &'tcx GlobalCtxt<'tcx>,
}

impl<'tcx> Deref for TyCtxt<'tcx> {
    type Target = &'tcx GlobalCtxt<'tcx>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.gcx
    }
}

pub struct GlobalCtxt<'tcx> {
    pub arena: &'tcx WorkerLocal<Arena<'tcx>>,
    pub hir_arena: &'tcx WorkerLocal<hir::Arena<'tcx>>,

    interners: CtxtInterners<'tcx>,

    pub sess: &'tcx Session,

    /// This only ever stores a `LintStore` but we don't want a dependency on that type here.
    ///
    /// FIXME(Centril): consider `dyn LintStoreMarker` once
    /// we can upcast to `Any` for some additional type safety.
    pub lint_store: Lrc<dyn Any + sync::Sync + sync::Send>,

    pub dep_graph: DepGraph,

    pub prof: SelfProfilerRef,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Common lifetimes, pre-interned for your convenience.
    pub lifetimes: CommonLifetimes<'tcx>,

    /// Common consts, pre-interned for your convenience.
    pub consts: CommonConsts<'tcx>,

    untracked: Untracked,

    /// This provides access to the incremental compilation on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure.
    /// This is `None` if we are not incremental compilation mode
    pub on_disk_cache: Option<&'tcx dyn OnDiskCache<'tcx>>,

    pub queries: &'tcx dyn query::QueryEngine<'tcx>,
    pub query_caches: query::QueryCaches<'tcx>,
    pub(crate) query_kinds: &'tcx [DepKindStruct<'tcx>],

    // Internal caches for metadata decoding. No need to track deps on this.
    pub ty_rcache: Lock<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,
    pub pred_rcache: Lock<FxHashMap<ty::CReaderCacheKey, Predicate<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    /// Stores memory for globals (statics/consts).
    pub(crate) alloc_map: Lock<interpret::AllocMap<'tcx>>,
}

impl<'tcx> TyCtxt<'tcx> {
    /// Expects a body and returns its codegen attributes.
    ///
    /// Unlike `codegen_fn_attrs`, this returns `CodegenFnAttrs::EMPTY` for
    /// constants.
    pub fn body_codegen_attrs(self, def_id: DefId) -> &'tcx CodegenFnAttrs {
        let def_kind = self.def_kind(def_id);
        if def_kind.has_codegen_attrs() {
            self.codegen_fn_attrs(def_id)
        } else if matches!(
            def_kind,
            DefKind::AnonConst | DefKind::AssocConst | DefKind::Const | DefKind::InlineConst
        ) {
            CodegenFnAttrs::EMPTY
        } else {
            bug!(
                "body_codegen_fn_attrs called on unexpected definition: {:?} {:?}",
                def_id,
                def_kind
            )
        }
    }

    pub fn typeck_opt_const_arg(
        self,
        def: ty::WithOptConstParam<LocalDefId>,
    ) -> &'tcx TypeckResults<'tcx> {
        if let Some(param_did) = def.const_param_did {
            self.typeck_const_arg((def.did, param_did))
        } else {
            self.typeck(def.did)
        }
    }

    pub fn mir_borrowck_opt_const_arg(
        self,
        def: ty::WithOptConstParam<LocalDefId>,
    ) -> &'tcx BorrowCheckResult<'tcx> {
        if let Some(param_did) = def.const_param_did {
            self.mir_borrowck_const_arg((def.did, param_did))
        } else {
            self.mir_borrowck(def.did)
        }
    }

    pub fn alloc_steal_thir(self, thir: Thir<'tcx>) -> &'tcx Steal<Thir<'tcx>> {
        self.arena.alloc(Steal::new(thir))
    }

    pub fn alloc_steal_mir(self, mir: Body<'tcx>) -> &'tcx Steal<Body<'tcx>> {
        self.arena.alloc(Steal::new(mir))
    }

    pub fn alloc_steal_promoted(
        self,
        promoted: IndexVec<Promoted, Body<'tcx>>,
    ) -> &'tcx Steal<IndexVec<Promoted, Body<'tcx>>> {
        self.arena.alloc(Steal::new(promoted))
    }

    pub fn alloc_adt_def(
        self,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, ty::VariantDef>,
        repr: ReprOptions,
    ) -> ty::AdtDef<'tcx> {
        self.intern_adt_def(ty::AdtDefData::new(self, did, kind, variants, repr))
    }

    /// Allocates a read-only byte or string literal for `mir::interpret`.
    pub fn allocate_bytes(self, bytes: &[u8]) -> interpret::AllocId {
        // Create an allocation that just contains these bytes.
        let alloc = interpret::Allocation::from_bytes_byte_aligned_immutable(bytes);
        let alloc = self.intern_const_alloc(alloc);
        self.create_memory_alloc(alloc)
    }

    /// Returns a range of the start/end indices specified with the
    /// `rustc_layout_scalar_valid_range` attribute.
    // FIXME(eddyb) this is an awkward spot for this method, maybe move it?
    pub fn layout_scalar_valid_range(self, def_id: DefId) -> (Bound<u128>, Bound<u128>) {
        let get = |name| {
            let Some(attr) = self.get_attr(def_id, name) else {
                return Bound::Unbounded;
            };
            debug!("layout_scalar_valid_range: attr={:?}", attr);
            if let Some(
                &[
                    ast::NestedMetaItem::Lit(ast::MetaItemLit {
                        kind: ast::LitKind::Int(a, _),
                        ..
                    }),
                ],
            ) = attr.meta_item_list().as_deref()
            {
                Bound::Included(a)
            } else {
                self.sess
                    .delay_span_bug(attr.span, "invalid rustc_layout_scalar_valid_range attribute");
                Bound::Unbounded
            }
        };
        (
            get(sym::rustc_layout_scalar_valid_range_start),
            get(sym::rustc_layout_scalar_valid_range_end),
        )
    }

    pub fn lift<T: Lift<'tcx>>(self, value: T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Creates a type context and call the closure with a `TyCtxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_global_ctxt(
        s: &'tcx Session,
        lint_store: Lrc<dyn Any + sync::Send + sync::Sync>,
        arena: &'tcx WorkerLocal<Arena<'tcx>>,
        hir_arena: &'tcx WorkerLocal<hir::Arena<'tcx>>,
        untracked: Untracked,
        dep_graph: DepGraph,
        on_disk_cache: Option<&'tcx dyn OnDiskCache<'tcx>>,
        queries: &'tcx dyn query::QueryEngine<'tcx>,
        query_kinds: &'tcx [DepKindStruct<'tcx>],
    ) -> GlobalCtxt<'tcx> {
        let data_layout = s.target.parse_data_layout().unwrap_or_else(|err| {
            s.emit_fatal(err);
        });
        let interners = CtxtInterners::new(arena);
        let common_types = CommonTypes::new(&interners, s, &untracked);
        let common_lifetimes = CommonLifetimes::new(&interners);
        let common_consts = CommonConsts::new(&interners, &common_types);

        GlobalCtxt {
            sess: s,
            lint_store,
            arena,
            hir_arena,
            interners,
            dep_graph,
            prof: s.prof.clone(),
            types: common_types,
            lifetimes: common_lifetimes,
            consts: common_consts,
            untracked,
            on_disk_cache,
            queries,
            query_caches: query::QueryCaches::default(),
            query_kinds,
            ty_rcache: Default::default(),
            pred_rcache: Default::default(),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            data_layout,
            alloc_map: Lock::new(interpret::AllocMap::new()),
        }
    }

    /// Constructs a `TyKind::Error` type with current `ErrorGuaranteed`
    #[track_caller]
    pub fn ty_error_with_guaranteed(self, reported: ErrorGuaranteed) -> Ty<'tcx> {
        self.mk_ty(Error(reported))
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` to ensure it gets used.
    #[track_caller]
    pub fn ty_error(self) -> Ty<'tcx> {
        self.ty_error_with_message(DUMMY_SP, "TyKind::Error constructed but no error reported")
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` with the given `msg` to
    /// ensure it gets used.
    #[track_caller]
    pub fn ty_error_with_message<S: Into<MultiSpan>>(self, span: S, msg: &str) -> Ty<'tcx> {
        let reported = self.sess.delay_span_bug(span, msg);
        self.mk_ty(Error(reported))
    }

    /// Like [TyCtxt::ty_error] but for constants, with current `ErrorGuaranteed`
    #[track_caller]
    pub fn const_error_with_guaranteed(
        self,
        ty: Ty<'tcx>,
        reported: ErrorGuaranteed,
    ) -> Const<'tcx> {
        self.mk_const(ty::ConstKind::Error(reported), ty)
    }

    /// Like [TyCtxt::ty_error] but for constants.
    #[track_caller]
    pub fn const_error(self, ty: Ty<'tcx>) -> Const<'tcx> {
        self.const_error_with_message(
            ty,
            DUMMY_SP,
            "ty::ConstKind::Error constructed but no error reported",
        )
    }

    /// Like [TyCtxt::ty_error_with_message] but for constants.
    #[track_caller]
    pub fn const_error_with_message<S: Into<MultiSpan>>(
        self,
        ty: Ty<'tcx>,
        span: S,
        msg: &str,
    ) -> Const<'tcx> {
        let reported = self.sess.delay_span_bug(span, msg);
        self.mk_const(ty::ConstKind::Error(reported), ty)
    }

    pub fn consider_optimizing<T: Fn() -> String>(self, msg: T) -> bool {
        let cname = self.crate_name(LOCAL_CRATE);
        self.sess.consider_optimizing(cname.as_str(), msg)
    }

    /// Obtain all lang items of this crate and all dependencies (recursively)
    pub fn lang_items(self) -> &'tcx rustc_hir::lang_items::LanguageItems {
        self.get_lang_items(())
    }

    /// Obtain the given diagnostic item's `DefId`. Use `is_diagnostic_item` if you just want to
    /// compare against another `DefId`, since `is_diagnostic_item` is cheaper.
    pub fn get_diagnostic_item(self, name: Symbol) -> Option<DefId> {
        self.all_diagnostic_items(()).name_to_id.get(&name).copied()
    }

    /// Obtain the diagnostic item's name
    pub fn get_diagnostic_name(self, id: DefId) -> Option<Symbol> {
        self.diagnostic_items(id.krate).id_to_name.get(&id).copied()
    }

    /// Check whether the diagnostic item with the given `name` has the given `DefId`.
    pub fn is_diagnostic_item(self, name: Symbol, did: DefId) -> bool {
        self.diagnostic_items(did.krate).name_to_id.get(&name) == Some(&did)
    }

    /// Returns `true` if the node pointed to by `def_id` is a generator for an async construct.
    pub fn generator_is_async(self, def_id: DefId) -> bool {
        matches!(self.generator_kind(def_id), Some(hir::GeneratorKind::Async(_)))
    }

    pub fn stability(self) -> &'tcx stability::Index {
        self.stability_index(())
    }

    pub fn features(self) -> &'tcx rustc_feature::Features {
        self.features_query(())
    }

    pub fn def_key(self, id: DefId) -> rustc_hir::definitions::DefKey {
        // Accessing the DefKey is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.definitions_untracked().def_key(id)
        } else {
            self.untracked.cstore.def_key(id)
        }
    }

    /// Converts a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned `DefPath`).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> rustc_hir::definitions::DefPath {
        // Accessing the DefPath is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.definitions_untracked().def_path(id)
        } else {
            self.untracked.cstore.def_path(id)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> rustc_hir::definitions::DefPathHash {
        // Accessing the DefPathHash is ok, it is incr. comp. stable.
        if let Some(def_id) = def_id.as_local() {
            self.definitions_untracked().def_path_hash(def_id)
        } else {
            self.untracked.cstore.def_path_hash(def_id)
        }
    }

    #[inline]
    pub fn stable_crate_id(self, crate_num: CrateNum) -> StableCrateId {
        if crate_num == LOCAL_CRATE {
            self.sess.local_stable_crate_id()
        } else {
            self.untracked.cstore.stable_crate_id(crate_num)
        }
    }

    /// Maps a StableCrateId to the corresponding CrateNum. This method assumes
    /// that the crate in question has already been loaded by the CrateStore.
    #[inline]
    pub fn stable_crate_id_to_crate_num(self, stable_crate_id: StableCrateId) -> CrateNum {
        if stable_crate_id == self.sess.local_stable_crate_id() {
            LOCAL_CRATE
        } else {
            self.untracked.cstore.stable_crate_id_to_crate_num(stable_crate_id)
        }
    }

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    pub fn def_path_hash_to_def_id(self, hash: DefPathHash, err: &mut dyn FnMut() -> !) -> DefId {
        debug!("def_path_hash_to_def_id({:?})", hash);

        let stable_crate_id = hash.stable_crate_id();

        // If this is a DefPathHash from the local crate, we can look up the
        // DefId in the tcx's `Definitions`.
        if stable_crate_id == self.sess.local_stable_crate_id() {
            self.untracked.definitions.read().local_def_path_hash_to_def_id(hash, err).to_def_id()
        } else {
            // If this is a DefPathHash from an upstream crate, let the CrateStore map
            // it to a DefId.
            let cstore = &*self.untracked.cstore;
            let cnum = cstore.stable_crate_id_to_crate_num(stable_crate_id);
            cstore.def_path_hash_to_def_id(cnum, hash)
        }
    }

    pub fn def_path_debug_str(self, def_id: DefId) -> String {
        // We are explicitly not going through queries here in order to get
        // crate name and stable crate id since this code is called from debug!()
        // statements within the query system and we'd run into endless
        // recursion otherwise.
        let (crate_name, stable_crate_id) = if def_id.is_local() {
            (self.crate_name(LOCAL_CRATE), self.sess.local_stable_crate_id())
        } else {
            let cstore = &*self.untracked.cstore;
            (cstore.crate_name(def_id.krate), cstore.stable_crate_id(def_id.krate))
        };

        format!(
            "{}[{:04x}]{}",
            crate_name,
            // Don't print the whole stable crate id. That's just
            // annoying in debug output.
            stable_crate_id.to_u64() >> 8 * 6,
            self.def_path(def_id).to_string_no_crate_verbose()
        )
    }
}

impl<'tcx> TyCtxtAt<'tcx> {
    /// Create a new definition within the incr. comp. engine.
    pub fn create_def(
        self,
        parent: LocalDefId,
        data: hir::definitions::DefPathData,
    ) -> TyCtxtFeed<'tcx, LocalDefId> {
        // This function modifies `self.definitions` using a side-effect.
        // We need to ensure that these side effects are re-run by the incr. comp. engine.
        // Depending on the forever-red node will tell the graph that the calling query
        // needs to be re-evaluated.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        // The following call has the side effect of modifying the tables inside `definitions`.
        // These very tables are relied on by the incr. comp. engine to decode DepNodes and to
        // decode the on-disk cache.
        //
        // Any LocalDefId which is used within queries, either as key or result, either:
        // - has been created before the construction of the TyCtxt;
        // - has been created by this call to `create_def`.
        // As a consequence, this LocalDefId is always re-created before it is needed by the incr.
        // comp. engine itself.
        //
        // This call also writes to the value of `source_span` and `expn_that_defined` queries.
        // This is fine because:
        // - those queries are `eval_always` so we won't miss their result changing;
        // - this write will have happened before these queries are called.
        let key = self.untracked.definitions.write().create_def(parent, data);

        let feed = TyCtxtFeed { tcx: self.tcx, key };
        feed.def_span(self.span);
        feed
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn iter_local_def_id(self) -> impl Iterator<Item = LocalDefId> + 'tcx {
        // Create a dependency to the red node to be sure we re-execute this when the amount of
        // definitions change.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        let definitions = &self.untracked.definitions;
        std::iter::from_generator(|| {
            let mut i = 0;

            // Recompute the number of definitions each time, because our caller may be creating
            // new ones.
            while i < { definitions.read().num_definitions() } {
                let local_def_index = rustc_span::def_id::DefIndex::from_usize(i);
                yield LocalDefId { local_def_index };
                i += 1;
            }

            // Leak a read lock once we finish iterating on definitions, to prevent adding new ones.
            definitions.leak();
        })
    }

    pub fn def_path_table(self) -> &'tcx rustc_hir::definitions::DefPathTable {
        // Create a dependency to the crate to be sure we re-execute this when the amount of
        // definitions change.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        // Leak a read lock once we start iterating on definitions, to prevent adding new ones
        // while iterating. If some query needs to add definitions, it should be `ensure`d above.
        let definitions = self.untracked.definitions.leak();
        definitions.def_path_table()
    }

    pub fn def_path_hash_to_def_index_map(
        self,
    ) -> &'tcx rustc_hir::def_path_hash_map::DefPathHashMap {
        // Create a dependency to the crate to be sure we re-execute this when the amount of
        // definitions change.
        self.ensure().hir_crate(());
        // Leak a read lock once we start iterating on definitions, to prevent adding new ones
        // while iterating. If some query needs to add definitions, it should be `ensure`d above.
        let definitions = self.untracked.definitions.leak();
        definitions.def_path_hash_to_def_index_map()
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    pub fn cstore_untracked(self) -> &'tcx CrateStoreDyn {
        &*self.untracked.cstore
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    #[inline]
    pub fn definitions_untracked(self) -> ReadGuard<'tcx, Definitions> {
        self.untracked.definitions.read()
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    #[inline]
    pub fn source_span_untracked(self, def_id: LocalDefId) -> Span {
        self.untracked.source_span.get(def_id).copied().unwrap_or(DUMMY_SP)
    }

    #[inline(always)]
    pub fn with_stable_hashing_context<R>(
        self,
        f: impl FnOnce(StableHashingContext<'_>) -> R,
    ) -> R {
        f(StableHashingContext::new(self.sess, &self.untracked))
    }

    pub fn serialize_query_result_cache(self, encoder: FileEncoder) -> FileEncodeResult {
        self.on_disk_cache.as_ref().map_or(Ok(0), |c| c.serialize(self, encoder))
    }

    /// If `true`, we should use lazy normalization for constants, otherwise
    /// we still evaluate them eagerly.
    #[inline]
    pub fn lazy_normalization(self) -> bool {
        let features = self.features();
        // Note: We only use lazy normalization for generic const expressions.
        features.generic_const_exprs
    }

    #[inline]
    pub fn local_crate_exports_generics(self) -> bool {
        debug_assert!(self.sess.opts.share_generics());

        self.sess.crate_types().iter().any(|crate_type| {
            match crate_type {
                CrateType::Executable
                | CrateType::Staticlib
                | CrateType::ProcMacro
                | CrateType::Cdylib => false,

                // FIXME rust-lang/rust#64319, rust-lang/rust#64872:
                // We want to block export of generics from dylibs,
                // but we must fix rust-lang/rust#65890 before we can
                // do that robustly.
                CrateType::Dylib => true,

                CrateType::Rlib => true,
            }
        })
    }

    /// Returns the `DefId` and the `BoundRegionKind` corresponding to the given region.
    pub fn is_suitable_region(self, region: Region<'tcx>) -> Option<FreeRegionInfo> {
        let (suitable_region_binding_scope, bound_region) = match *region {
            ty::ReFree(ref free_region) => {
                (free_region.scope.expect_local(), free_region.bound_region)
            }
            ty::ReEarlyBound(ref ebr) => (
                self.local_parent(ebr.def_id.expect_local()),
                ty::BoundRegionKind::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let is_impl_item = match self.hir().find_by_def_id(suitable_region_binding_scope) {
            Some(Node::Item(..) | Node::TraitItem(..)) => false,
            Some(Node::ImplItem(..)) => {
                self.is_bound_region_in_impl_item(suitable_region_binding_scope)
            }
            _ => return None,
        };

        Some(FreeRegionInfo {
            def_id: suitable_region_binding_scope,
            boundregion: bound_region,
            is_impl_item,
        })
    }

    /// Given a `DefId` for an `fn`, return all the `dyn` and `impl` traits in its return type.
    pub fn return_type_impl_or_dyn_traits(
        self,
        scope_def_id: LocalDefId,
    ) -> Vec<&'tcx hir::Ty<'tcx>> {
        let hir_id = self.hir().local_def_id_to_hir_id(scope_def_id);
        let Some(hir::FnDecl { output: hir::FnRetTy::Return(hir_output), .. }) = self.hir().fn_decl_by_hir_id(hir_id) else {
            return vec![];
        };

        let mut v = TraitObjectVisitor(vec![], self.hir());
        v.visit_ty(hir_output);
        v.0
    }

    /// Given a `DefId` for an `fn`, return all the `dyn` and `impl` traits in its return type and associated alias span when type alias is used
    pub fn return_type_impl_or_dyn_traits_with_type_alias(
        self,
        scope_def_id: LocalDefId,
    ) -> Option<(Vec<&'tcx hir::Ty<'tcx>>, Span)> {
        let hir_id = self.hir().local_def_id_to_hir_id(scope_def_id);
        let mut v = TraitObjectVisitor(vec![], self.hir());
        // when the return type is a type alias
        if let Some(hir::FnDecl { output: hir::FnRetTy::Return(hir_output), .. }) = self.hir().fn_decl_by_hir_id(hir_id)
            && let hir::TyKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { res: hir::def::Res::Def(DefKind::TyAlias, def_id), .. }, )) = hir_output.kind
            && let Some(local_id) = def_id.as_local()
            && let Some(alias_ty) = self.hir().get_by_def_id(local_id).alias_ty() // it is type alias
            && let Some(alias_generics) = self.hir().get_by_def_id(local_id).generics()
        {
            v.visit_ty(alias_ty);
            if !v.0.is_empty() {
                return Some((v.0, alias_generics.span));
            }
        }
        return None;
    }

    pub fn return_type_impl_trait(self, scope_def_id: LocalDefId) -> Option<(Ty<'tcx>, Span)> {
        // `type_of()` will fail on these (#55796, #86483), so only allow `fn`s or closures.
        match self.hir().get_by_def_id(scope_def_id) {
            Node::Item(&hir::Item { kind: ItemKind::Fn(..), .. }) => {}
            Node::TraitItem(&hir::TraitItem { kind: TraitItemKind::Fn(..), .. }) => {}
            Node::ImplItem(&hir::ImplItem { kind: ImplItemKind::Fn(..), .. }) => {}
            Node::Expr(&hir::Expr { kind: ExprKind::Closure { .. }, .. }) => {}
            _ => return None,
        }

        let ret_ty = self.type_of(scope_def_id);
        match ret_ty.kind() {
            ty::FnDef(_, _) => {
                let sig = ret_ty.fn_sig(self);
                let output = self.erase_late_bound_regions(sig.output());
                if output.is_impl_trait() {
                    let hir_id = self.hir().local_def_id_to_hir_id(scope_def_id);
                    let fn_decl = self.hir().fn_decl_by_hir_id(hir_id).unwrap();
                    Some((output, fn_decl.output.span()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Checks if the bound region is in Impl Item.
    pub fn is_bound_region_in_impl_item(self, suitable_region_binding_scope: LocalDefId) -> bool {
        let container_id = self.parent(suitable_region_binding_scope.to_def_id());
        if self.impl_trait_ref(container_id).is_some() {
            // For now, we do not try to target impls of traits. This is
            // because this message is going to suggest that the user
            // change the fn signature, but they may not be free to do so,
            // since the signature must match the trait.
            //
            // FIXME(#42706) -- in some cases, we could do better here.
            return true;
        }
        false
    }

    /// Determines whether identifiers in the assembly have strict naming rules.
    /// Currently, only NVPTX* targets need it.
    pub fn has_strict_asm_symbol_naming(self) -> bool {
        self.sess.target.arch.contains("nvptx")
    }

    /// Returns `&'static core::panic::Location<'static>`.
    pub fn caller_location_ty(self) -> Ty<'tcx> {
        self.mk_imm_ref(
            self.lifetimes.re_static,
            self.bound_type_of(self.require_lang_item(LangItem::PanicLocation, None))
                .subst(self, self.mk_substs([self.lifetimes.re_static.into()].iter())),
        )
    }

    /// Returns a displayable description and article for the given `def_id` (e.g. `("a", "struct")`).
    pub fn article_and_description(self, def_id: DefId) -> (&'static str, &'static str) {
        match self.def_kind(def_id) {
            DefKind::Generator => match self.generator_kind(def_id).unwrap() {
                rustc_hir::GeneratorKind::Async(..) => ("an", "async closure"),
                rustc_hir::GeneratorKind::Gen => ("a", "generator"),
            },
            def_kind => (def_kind.article(), def_kind.descr(def_id)),
        }
    }

    pub fn type_length_limit(self) -> Limit {
        self.limits(()).type_length_limit
    }

    pub fn recursion_limit(self) -> Limit {
        self.limits(()).recursion_limit
    }

    pub fn move_size_limit(self) -> Limit {
        self.limits(()).move_size_limit
    }

    pub fn const_eval_limit(self) -> Limit {
        if self.sess.opts.unstable_opts.tiny_const_eval_limit {
            TINY_CONST_EVAL_LIMIT
        } else {
            self.limits(()).const_eval_limit
        }
    }

    pub fn all_traits(self) -> impl Iterator<Item = DefId> + 'tcx {
        iter::once(LOCAL_CRATE)
            .chain(self.crates(()).iter().copied())
            .flat_map(move |cnum| self.traits_in_crate(cnum).iter().copied())
    }

    #[inline]
    pub fn local_visibility(self, def_id: LocalDefId) -> Visibility {
        self.visibility(def_id).expect_local()
    }
}

/// A trait implemented for all `X<'a>` types that can be safely and
/// efficiently converted to `X<'tcx>` as long as they are part of the
/// provided `TyCtxt<'tcx>`.
/// This can be done, for example, for `Ty<'tcx>` or `SubstsRef<'tcx>`
/// by looking them up in their respective interners.
///
/// However, this is still not the best implementation as it does
/// need to compare the components, even for interned values.
/// It would be more efficient if `TypedArena` provided a way to
/// determine whether the address is in the allocated range.
///
/// `None` is returned if the value or one of the components is not part
/// of the provided context.
/// For `Ty`, `None` can be returned if either the type interner doesn't
/// contain the `TyKind` key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g., `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx>: fmt::Debug {
    type Lifted: fmt::Debug + 'tcx;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted>;
}

macro_rules! nop_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for $ty {
            type Lifted = $lifted;
            fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                if tcx.interners.$set.contains_pointer_to(&InternedInSet(&*self.0.0)) {
                    // SAFETY: `self` is interned and therefore valid
                    // for the entire lifetime of the `TyCtxt`.
                    Some(unsafe { mem::transmute(self) })
                } else {
                    None
                }
            }
        }
    };
}

// Can't use the macros as we have reuse the `substs` here.
//
// See `intern_type_list` for more info.
impl<'a, 'tcx> Lift<'tcx> for &'a List<Ty<'a>> {
    type Lifted = &'tcx List<Ty<'tcx>>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        if self.is_empty() {
            return Some(List::empty());
        }
        if tcx.interners.substs.contains_pointer_to(&InternedInSet(self.as_substs())) {
            // SAFETY: `self` is interned and therefore valid
            // for the entire lifetime of the `TyCtxt`.
            Some(unsafe { mem::transmute::<&'a List<Ty<'a>>, &'tcx List<Ty<'tcx>>>(self) })
        } else {
            None
        }
    }
}

macro_rules! nop_list_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for &'a List<$ty> {
            type Lifted = &'tcx List<$lifted>;
            fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                if self.is_empty() {
                    return Some(List::empty());
                }
                if tcx.interners.$set.contains_pointer_to(&InternedInSet(self)) {
                    Some(unsafe { mem::transmute(self) })
                } else {
                    None
                }
            }
        }
    };
}

nop_lift! {type_; Ty<'a> => Ty<'tcx>}
nop_lift! {region; Region<'a> => Region<'tcx>}
nop_lift! {const_; Const<'a> => Const<'tcx>}
nop_lift! {const_allocation; ConstAllocation<'a> => ConstAllocation<'tcx>}
nop_lift! {predicate; Predicate<'a> => Predicate<'tcx>}

nop_list_lift! {poly_existential_predicates; PolyExistentialPredicate<'a> => PolyExistentialPredicate<'tcx>}
nop_list_lift! {predicates; Predicate<'a> => Predicate<'tcx>}
nop_list_lift! {canonical_var_infos; CanonicalVarInfo<'a> => CanonicalVarInfo<'tcx>}
nop_list_lift! {projs; ProjectionKind => ProjectionKind}
nop_list_lift! {bound_variable_kinds; ty::BoundVariableKind => ty::BoundVariableKind}

// This is the impl for `&'a InternalSubsts<'a>`.
nop_list_lift! {substs; GenericArg<'a> => GenericArg<'tcx>}

CloneLiftImpls! { for<'tcx> {
    Constness, traits::WellFormedLoc, ImplPolarity, crate::mir::ReturnConstraint,
} }

macro_rules! sty_debug_print {
    ($fmt: expr, $ctxt: expr, $($variant: ident),*) => {{
        // Curious inner module to allow variant names to be used as
        // variable names.
        #[allow(non_snake_case)]
        mod inner {
            use crate::ty::{self, TyCtxt};
            use crate::ty::context::InternedInSet;

            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                lt_infer: usize,
                ty_infer: usize,
                ct_infer: usize,
                all_infer: usize,
            }

            pub fn go(fmt: &mut std::fmt::Formatter<'_>, tcx: TyCtxt<'_>) -> std::fmt::Result {
                let mut total = DebugStat {
                    total: 0,
                    lt_infer: 0,
                    ty_infer: 0,
                    ct_infer: 0,
                    all_infer: 0,
                };
                $(let mut $variant = total;)*

                let shards = tcx.interners.type_.lock_shards();
                let types = shards.iter().flat_map(|shard| shard.keys());
                for &InternedInSet(t) in types {
                    let variant = match t.internee {
                        ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
                            ty::Float(..) | ty::Str | ty::Never => continue,
                        ty::Error(_) => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let lt = t.flags.intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.intersects(ty::TypeFlags::HAS_TY_INFER);
                    let ct = t.flags.intersects(ty::TypeFlags::HAS_CT_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if lt { total.lt_infer += 1; variant.lt_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if ct { total.ct_infer += 1; variant.ct_infer += 1 }
                    if lt && ty && ct { total.all_infer += 1; variant.all_infer += 1 }
                }
                writeln!(fmt, "Ty interner             total           ty lt ct all")?;
                $(writeln!(fmt, "    {:18}: {uses:6} {usespc:4.1}%, \
                            {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    stringify!($variant),
                    uses = $variant.total,
                    usespc = $variant.total as f64 * 100.0 / total.total as f64,
                    ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = $variant.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = $variant.ct_infer as f64 * 100.0  / total.total as f64,
                    all = $variant.all_infer as f64 * 100.0  / total.total as f64)?;
                )*
                writeln!(fmt, "                  total {uses:6}        \
                          {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    uses = total.total,
                    ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = total.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = total.ct_infer as f64 * 100.0  / total.total as f64,
                    all = total.all_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($fmt, $ctxt)
    }}
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn debug_stats(self) -> impl std::fmt::Debug + 'tcx {
        struct DebugStats<'tcx>(TyCtxt<'tcx>);

        impl<'tcx> std::fmt::Debug for DebugStats<'tcx> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                sty_debug_print!(
                    fmt,
                    self.0,
                    Adt,
                    Array,
                    Slice,
                    RawPtr,
                    Ref,
                    FnDef,
                    FnPtr,
                    Placeholder,
                    Generator,
                    GeneratorWitness,
                    GeneratorWitnessMIR,
                    Dynamic,
                    Closure,
                    Tuple,
                    Bound,
                    Param,
                    Infer,
                    Alias,
                    Foreign
                )?;

                writeln!(fmt, "InternalSubsts interner: #{}", self.0.interners.substs.len())?;
                writeln!(fmt, "Region interner: #{}", self.0.interners.region.len())?;
                writeln!(
                    fmt,
                    "Const Allocation interner: #{}",
                    self.0.interners.const_allocation.len()
                )?;
                writeln!(fmt, "Layout interner: #{}", self.0.interners.layout.len())?;

                Ok(())
            }
        }

        DebugStats(self)
    }
}

// This type holds a `T` in the interner. The `T` is stored in the arena and
// this type just holds a pointer to it, but it still effectively owns it. It
// impls `Borrow` so that it can be looked up using the original
// (non-arena-memory-owning) types.
struct InternedInSet<'tcx, T: ?Sized>(&'tcx T);

impl<'tcx, T: 'tcx + ?Sized> Clone for InternedInSet<'tcx, T> {
    fn clone(&self) -> Self {
        InternedInSet(self.0)
    }
}

impl<'tcx, T: 'tcx + ?Sized> Copy for InternedInSet<'tcx, T> {}

impl<'tcx, T: 'tcx + ?Sized> IntoPointer for InternedInSet<'tcx, T> {
    fn into_pointer(&self) -> *const () {
        self.0 as *const _ as *const ()
    }
}

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx, T> Borrow<T> for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn borrow(&self) -> &T {
        &self.0.internee
    }
}

impl<'tcx, T: PartialEq> PartialEq for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn eq(&self, other: &InternedInSet<'tcx, WithCachedTypeInfo<T>>) -> bool {
        // The `Borrow` trait requires that `x.borrow() == y.borrow()` equals
        // `x == y`.
        self.0.internee == other.0.internee
    }
}

impl<'tcx, T: Eq> Eq for InternedInSet<'tcx, WithCachedTypeInfo<T>> {}

impl<'tcx, T: Hash> Hash for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        // The `Borrow` trait requires that `x.borrow().hash(s) == x.hash(s)`.
        self.0.internee.hash(s)
    }
}

impl<'tcx, T> Borrow<[T]> for InternedInSet<'tcx, List<T>> {
    fn borrow(&self) -> &[T] {
        &self.0[..]
    }
}

impl<'tcx, T: PartialEq> PartialEq for InternedInSet<'tcx, List<T>> {
    fn eq(&self, other: &InternedInSet<'tcx, List<T>>) -> bool {
        // The `Borrow` trait requires that `x.borrow() == y.borrow()` equals
        // `x == y`.
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for InternedInSet<'tcx, List<T>> {}

impl<'tcx, T: Hash> Hash for InternedInSet<'tcx, List<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        // The `Borrow` trait requires that `x.borrow().hash(s) == x.hash(s)`.
        self.0[..].hash(s)
    }
}

macro_rules! direct_interners {
    ($($name:ident: $method:ident($ty:ty): $ret_ctor:ident -> $ret_ty:ty,)+) => {
        $(impl<'tcx> Borrow<$ty> for InternedInSet<'tcx, $ty> {
            fn borrow<'a>(&'a self) -> &'a $ty {
                &self.0
            }
        }

        impl<'tcx> PartialEq for InternedInSet<'tcx, $ty> {
            fn eq(&self, other: &Self) -> bool {
                // The `Borrow` trait requires that `x.borrow() == y.borrow()`
                // equals `x == y`.
                self.0 == other.0
            }
        }

        impl<'tcx> Eq for InternedInSet<'tcx, $ty> {}

        impl<'tcx> Hash for InternedInSet<'tcx, $ty> {
            fn hash<H: Hasher>(&self, s: &mut H) {
                // The `Borrow` trait requires that `x.borrow().hash(s) ==
                // x.hash(s)`.
                self.0.hash(s)
            }
        }

        impl<'tcx> TyCtxt<'tcx> {
            pub fn $method(self, v: $ty) -> $ret_ty {
                $ret_ctor(Interned::new_unchecked(self.interners.$name.intern(v, |v| {
                    InternedInSet(self.interners.arena.alloc(v))
                }).0))
            }
        })+
    }
}

direct_interners! {
    region: mk_region(RegionKind<'tcx>): Region -> Region<'tcx>,
    const_: mk_const_internal(ConstData<'tcx>): Const -> Const<'tcx>,
    const_allocation: intern_const_alloc(Allocation): ConstAllocation -> ConstAllocation<'tcx>,
    layout: intern_layout(LayoutS<VariantIdx>): Layout -> Layout<'tcx>,
    adt_def: intern_adt_def(AdtDefData): AdtDef -> AdtDef<'tcx>,
}

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ty)),+ $(,)?) => (
        impl<'tcx> TyCtxt<'tcx> {
            $(pub fn $method(self, v: &[$ty]) -> &'tcx List<$ty> {
                self.interners.$field.intern_ref(v, || {
                    InternedInSet(List::from_arena(&*self.arena, v))
                }).0
            })+
        }
    );
}

slice_interners!(
    const_lists: _intern_const_list(Const<'tcx>),
    substs: _intern_substs(GenericArg<'tcx>),
    canonical_var_infos: _intern_canonical_var_infos(CanonicalVarInfo<'tcx>),
    poly_existential_predicates:
        _intern_poly_existential_predicates(PolyExistentialPredicate<'tcx>),
    predicates: _intern_predicates(Predicate<'tcx>),
    projs: _intern_projs(ProjectionKind),
    place_elems: _intern_place_elems(PlaceElem<'tcx>),
    bound_variable_kinds: _intern_bound_variable_kinds(ty::BoundVariableKind),
);

impl<'tcx> TyCtxt<'tcx> {
    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        assert_eq!(sig.unsafety(), hir::Unsafety::Normal);
        self.mk_fn_ptr(sig.map_bound(|sig| ty::FnSig { unsafety: hir::Unsafety::Unsafe, ..sig }))
    }

    /// Given the def_id of a Trait `trait_def_id` and the name of an associated item `assoc_name`
    /// returns true if the `trait_def_id` defines an associated item of name `assoc_name`.
    pub fn trait_may_define_assoc_type(self, trait_def_id: DefId, assoc_name: Ident) -> bool {
        self.super_traits_of(trait_def_id).any(|trait_did| {
            self.associated_items(trait_did)
                .find_by_name_and_kind(self, assoc_name, ty::AssocKind::Type, trait_did)
                .is_some()
        })
    }

    /// Given a `ty`, return whether it's an `impl Future<...>`.
    pub fn ty_is_opaque_future(self, ty: Ty<'_>) -> bool {
        let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = ty.kind() else { return false };
        let future_trait = self.require_lang_item(LangItem::Future, None);

        self.explicit_item_bounds(def_id).iter().any(|(predicate, _)| {
            let ty::PredicateKind::Clause(ty::Clause::Trait(trait_predicate)) = predicate.kind().skip_binder() else {
                return false;
            };
            trait_predicate.trait_ref.def_id == future_trait
                && trait_predicate.polarity == ImplPolarity::Positive
        })
    }

    /// Computes the def-ids of the transitive supertraits of `trait_def_id`. This (intentionally)
    /// does not compute the full elaborated super-predicates but just the set of def-ids. It is used
    /// to identify which traits may define a given associated type to help avoid cycle errors.
    /// Returns a `DefId` iterator.
    fn super_traits_of(self, trait_def_id: DefId) -> impl Iterator<Item = DefId> + 'tcx {
        let mut set = FxHashSet::default();
        let mut stack = vec![trait_def_id];

        set.insert(trait_def_id);

        iter::from_fn(move || -> Option<DefId> {
            let trait_did = stack.pop()?;
            let generic_predicates = self.super_predicates_of(trait_did);

            for (predicate, _) in generic_predicates.predicates {
                if let ty::PredicateKind::Clause(ty::Clause::Trait(data)) =
                    predicate.kind().skip_binder()
                {
                    if set.insert(data.def_id()) {
                        stack.push(data.def_id());
                    }
                }
            }

            Some(trait_did)
        })
    }

    /// Given a closure signature, returns an equivalent fn signature. Detuples
    /// and so forth -- so e.g., if we have a sig with `Fn<(u32, i32)>` then
    /// you would get a `fn(u32, i32)`.
    /// `unsafety` determines the unsafety of the fn signature. If you pass
    /// `hir::Unsafety::Unsafe` in the previous example, then you would get
    /// an `unsafe fn (u32, i32)`.
    /// It cannot convert a closure that requires unsafe.
    pub fn signature_unclosure(
        self,
        sig: PolyFnSig<'tcx>,
        unsafety: hir::Unsafety,
    ) -> PolyFnSig<'tcx> {
        sig.map_bound(|s| {
            let params_iter = match s.inputs()[0].kind() {
                ty::Tuple(params) => params.into_iter(),
                _ => bug!(),
            };
            self.mk_fn_sig(params_iter, s.output(), s.c_variadic, unsafety, abi::Abi::Rust)
        })
    }

    /// Same a `self.mk_region(kind)`, but avoids accessing the interners if
    /// `*r == kind`.
    #[inline]
    pub fn reuse_or_mk_region(self, r: Region<'tcx>, kind: RegionKind<'tcx>) -> Region<'tcx> {
        if *r == kind { r } else { self.mk_region(kind) }
    }

    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn mk_ty(self, st: TyKind<'tcx>) -> Ty<'tcx> {
        self.interners.intern_ty(
            st,
            self.sess,
            // This is only used to create a stable hashing context.
            &self.untracked,
        )
    }

    #[inline]
    pub fn mk_predicate(self, binder: Binder<'tcx, PredicateKind<'tcx>>) -> Predicate<'tcx> {
        self.interners.intern_predicate(
            binder,
            self.sess,
            // This is only used to create a stable hashing context.
            &self.untracked,
        )
    }

    #[inline]
    pub fn reuse_or_mk_predicate(
        self,
        pred: Predicate<'tcx>,
        binder: Binder<'tcx, PredicateKind<'tcx>>,
    ) -> Predicate<'tcx> {
        if pred.kind() != binder { self.mk_predicate(binder) } else { pred }
    }

    pub fn mk_mach_int(self, tm: IntTy) -> Ty<'tcx> {
        match tm {
            IntTy::Isize => self.types.isize,
            IntTy::I8 => self.types.i8,
            IntTy::I16 => self.types.i16,
            IntTy::I32 => self.types.i32,
            IntTy::I64 => self.types.i64,
            IntTy::I128 => self.types.i128,
        }
    }

    pub fn mk_mach_uint(self, tm: UintTy) -> Ty<'tcx> {
        match tm {
            UintTy::Usize => self.types.usize,
            UintTy::U8 => self.types.u8,
            UintTy::U16 => self.types.u16,
            UintTy::U32 => self.types.u32,
            UintTy::U64 => self.types.u64,
            UintTy::U128 => self.types.u128,
        }
    }

    pub fn mk_mach_float(self, tm: FloatTy) -> Ty<'tcx> {
        match tm {
            FloatTy::F32 => self.types.f32,
            FloatTy::F64 => self.types.f64,
        }
    }

    #[inline]
    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.lifetimes.re_static, self.types.str_)
    }

    #[inline]
    pub fn mk_adt(self, def: AdtDef<'tcx>, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        // Take a copy of substs so that we own the vectors inside.
        self.mk_ty(Adt(def, substs))
    }

    #[inline]
    pub fn mk_foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.mk_ty(Foreign(def_id))
    }

    fn mk_generic_adt(self, wrapper_def_id: DefId, ty_param: Ty<'tcx>) -> Ty<'tcx> {
        let adt_def = self.adt_def(wrapper_def_id);
        let substs =
            InternalSubsts::for_item(self, wrapper_def_id, |param, substs| match param.kind {
                GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => bug!(),
                GenericParamDefKind::Type { has_default, .. } => {
                    if param.index == 0 {
                        ty_param.into()
                    } else {
                        assert!(has_default);
                        self.bound_type_of(param.def_id).subst(self, substs).into()
                    }
                }
            });
        self.mk_ty(Adt(adt_def, substs))
    }

    #[inline]
    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(LangItem::OwnedBox, None);
        self.mk_generic_adt(def_id, ty)
    }

    #[inline]
    pub fn mk_lang_item(self, ty: Ty<'tcx>, item: LangItem) -> Option<Ty<'tcx>> {
        let def_id = self.lang_items().get(item)?;
        Some(self.mk_generic_adt(def_id, ty))
    }

    #[inline]
    pub fn mk_diagnostic_item(self, ty: Ty<'tcx>, name: Symbol) -> Option<Ty<'tcx>> {
        let def_id = self.get_diagnostic_item(name)?;
        Some(self.mk_generic_adt(def_id, ty))
    }

    #[inline]
    pub fn mk_maybe_uninit(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(LangItem::MaybeUninit, None);
        self.mk_generic_adt(def_id, ty)
    }

    #[inline]
    pub fn mk_ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(RawPtr(tm))
    }

    #[inline]
    pub fn mk_ref(self, r: Region<'tcx>, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Ref(r, tm.ty, tm.mutbl))
    }

    #[inline]
    pub fn mk_mut_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn mk_imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn mk_array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.mk_ty(Array(ty, ty::Const::from_usize(self, n)))
    }

    #[inline]
    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Slice(ty))
    }

    #[inline]
    pub fn intern_tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        self.mk_ty(Tuple(self.intern_type_list(&ts)))
    }

    pub fn mk_tup<I: InternAs<Ty<'tcx>, Ty<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|ts| self.mk_ty(Tuple(self.intern_type_list(&ts))))
    }

    #[inline]
    pub fn mk_unit(self) -> Ty<'tcx> {
        self.types.unit
    }

    #[inline]
    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.features().never_type_fallback { self.types.never } else { self.types.unit }
    }

    #[inline]
    pub fn mk_fn_def(
        self,
        def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        let substs = self.check_substs(def_id, substs);
        self.mk_ty(FnDef(def_id, substs))
    }

    #[inline(always)]
    fn check_substs(
        self,
        _def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> SubstsRef<'tcx> {
        let substs = substs.into_iter().map(Into::into);
        #[cfg(debug_assertions)]
        {
            let n = self.generics_of(_def_id).count();
            assert_eq!(
                (n, Some(n)),
                substs.size_hint(),
                "wrong number of generic parameters for {_def_id:?}: {:?}",
                substs.collect::<Vec<_>>(),
            );
        }
        self.mk_substs(substs)
    }

    #[inline]
    pub fn mk_fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnPtr(fty))
    }

    #[inline]
    pub fn mk_dynamic(
        self,
        obj: &'tcx List<PolyExistentialPredicate<'tcx>>,
        reg: ty::Region<'tcx>,
        repr: DynKind,
    ) -> Ty<'tcx> {
        self.mk_ty(Dynamic(obj, reg, repr))
    }

    #[inline]
    pub fn mk_projection(
        self,
        item_def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        self.mk_ty(Alias(ty::Projection, self.mk_alias_ty(item_def_id, substs)))
    }

    #[inline]
    pub fn mk_closure(self, closure_id: DefId, closure_substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Closure(closure_id, closure_substs))
    }

    #[inline]
    pub fn mk_generator(
        self,
        id: DefId,
        generator_substs: SubstsRef<'tcx>,
        movability: hir::Movability,
    ) -> Ty<'tcx> {
        self.mk_ty(Generator(id, generator_substs, movability))
    }

    #[inline]
    pub fn mk_generator_witness(self, types: ty::Binder<'tcx, &'tcx List<Ty<'tcx>>>) -> Ty<'tcx> {
        self.mk_ty(GeneratorWitness(types))
    }

    /// Creates a `&mut Context<'_>` [`Ty`] with erased lifetimes.
    pub fn mk_task_context(self) -> Ty<'tcx> {
        let context_did = self.require_lang_item(LangItem::Context, None);
        let context_adt_ref = self.adt_def(context_did);
        let context_substs = self.intern_substs(&[self.lifetimes.re_erased.into()]);
        let context_ty = self.mk_adt(context_adt_ref, context_substs);
        self.mk_mut_ref(self.lifetimes.re_erased, context_ty)
    }

    #[inline]
    pub fn mk_generator_witness_mir(self, id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(GeneratorWitnessMIR(id, substs))
    }

    #[inline]
    pub fn mk_ty_var(self, v: TyVid) -> Ty<'tcx> {
        self.mk_ty_infer(TyVar(v))
    }

    #[inline]
    pub fn mk_const(self, kind: impl Into<ty::ConstKind<'tcx>>, ty: Ty<'tcx>) -> Const<'tcx> {
        self.mk_const_internal(ty::ConstData { kind: kind.into(), ty })
    }

    #[inline]
    pub fn mk_int_var(self, v: IntVid) -> Ty<'tcx> {
        self.mk_ty_infer(IntVar(v))
    }

    #[inline]
    pub fn mk_float_var(self, v: FloatVid) -> Ty<'tcx> {
        self.mk_ty_infer(FloatVar(v))
    }

    #[inline]
    pub fn mk_ty_infer(self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(Infer(it))
    }

    #[inline]
    pub fn mk_ty_param(self, index: u32, name: Symbol) -> Ty<'tcx> {
        self.mk_ty(Param(ParamTy { index, name }))
    }

    pub fn mk_param_from_def(self, param: &ty::GenericParamDef) -> GenericArg<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                self.mk_region(ty::ReEarlyBound(param.to_early_bound_region_data())).into()
            }
            GenericParamDefKind::Type { .. } => self.mk_ty_param(param.index, param.name).into(),
            GenericParamDefKind::Const { .. } => self
                .mk_const(
                    ParamConst { index: param.index, name: param.name },
                    self.type_of(param.def_id),
                )
                .into(),
        }
    }

    #[inline]
    pub fn mk_opaque(self, def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Alias(ty::Opaque, self.mk_alias_ty(def_id, substs)))
    }

    pub fn mk_place_field(self, place: Place<'tcx>, f: Field, ty: Ty<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Field(f, ty))
    }

    pub fn mk_place_deref(self, place: Place<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Deref)
    }

    pub fn mk_place_downcast(
        self,
        place: Place<'tcx>,
        adt_def: AdtDef<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(
            place,
            PlaceElem::Downcast(Some(adt_def.variant(variant_index).name), variant_index),
        )
    }

    pub fn mk_place_downcast_unnamed(
        self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Downcast(None, variant_index))
    }

    pub fn mk_place_index(self, place: Place<'tcx>, index: Local) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Index(index))
    }

    /// This method copies `Place`'s projection, add an element and reintern it. Should not be used
    /// to build a full `Place` it's just a convenient way to grab a projection and modify it in
    /// flight.
    pub fn mk_place_elem(self, place: Place<'tcx>, elem: PlaceElem<'tcx>) -> Place<'tcx> {
        let mut projection = place.projection.to_vec();
        projection.push(elem);

        Place { local: place.local, projection: self.intern_place_elems(&projection) }
    }

    pub fn intern_poly_existential_predicates(
        self,
        eps: &[PolyExistentialPredicate<'tcx>],
    ) -> &'tcx List<PolyExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(
            eps.array_windows()
                .all(|[a, b]| a.skip_binder().stable_cmp(self, &b.skip_binder())
                    != Ordering::Greater)
        );
        self._intern_poly_existential_predicates(eps)
    }

    pub fn intern_predicates(self, preds: &[Predicate<'tcx>]) -> &'tcx List<Predicate<'tcx>> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        if preds.is_empty() {
            // The macro-generated method below asserts we don't intern an empty slice.
            List::empty()
        } else {
            self._intern_predicates(preds)
        }
    }

    pub fn mk_const_list<I: InternAs<ty::Const<'tcx>, &'tcx List<ty::Const<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_const_list(xs))
    }

    pub fn intern_const_list(self, cs: &[ty::Const<'tcx>]) -> &'tcx List<ty::Const<'tcx>> {
        if cs.is_empty() { List::empty() } else { self._intern_const_list(cs) }
    }

    pub fn intern_type_list(self, ts: &[Ty<'tcx>]) -> &'tcx List<Ty<'tcx>> {
        if ts.is_empty() {
            List::empty()
        } else {
            // Actually intern type lists as lists of `GenericArg`s.
            //
            // Transmuting from `Ty<'tcx>` to `GenericArg<'tcx>` is sound
            // as explained in ty_slice_as_generic_arg`. With this,
            // we guarantee that even when transmuting between `List<Ty<'tcx>>`
            // and `List<GenericArg<'tcx>>`, the uniqueness requirement for
            // lists is upheld.
            let substs = self._intern_substs(ty::subst::ty_slice_as_generic_args(ts));
            substs.try_as_type_list().unwrap()
        }
    }

    pub fn intern_substs(self, ts: &[GenericArg<'tcx>]) -> &'tcx List<GenericArg<'tcx>> {
        if ts.is_empty() { List::empty() } else { self._intern_substs(ts) }
    }

    pub fn intern_projs(self, ps: &[ProjectionKind]) -> &'tcx List<ProjectionKind> {
        if ps.is_empty() { List::empty() } else { self._intern_projs(ps) }
    }

    pub fn intern_place_elems(self, ts: &[PlaceElem<'tcx>]) -> &'tcx List<PlaceElem<'tcx>> {
        if ts.is_empty() { List::empty() } else { self._intern_place_elems(ts) }
    }

    pub fn intern_canonical_var_infos(
        self,
        ts: &[CanonicalVarInfo<'tcx>],
    ) -> CanonicalVarInfos<'tcx> {
        if ts.is_empty() { List::empty() } else { self._intern_canonical_var_infos(ts) }
    }

    pub fn intern_bound_variable_kinds(
        self,
        ts: &[ty::BoundVariableKind],
    ) -> &'tcx List<ty::BoundVariableKind> {
        if ts.is_empty() { List::empty() } else { self._intern_bound_variable_kinds(ts) }
    }

    pub fn mk_fn_sig<I>(
        self,
        inputs: I,
        output: I::Item,
        c_variadic: bool,
        unsafety: hir::Unsafety,
        abi: abi::Abi,
    ) -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
    where
        I: Iterator<Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>,
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            c_variadic,
            unsafety,
            abi,
        })
    }

    pub fn mk_poly_existential_predicates<
        I: InternAs<PolyExistentialPredicate<'tcx>, &'tcx List<PolyExistentialPredicate<'tcx>>>,
    >(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_poly_existential_predicates(xs))
    }

    pub fn mk_predicates<I: InternAs<Predicate<'tcx>, &'tcx List<Predicate<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_predicates(xs))
    }

    pub fn mk_type_list<I: InternAs<Ty<'tcx>, &'tcx List<Ty<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_type_list(xs))
    }

    pub fn mk_substs<I: InternAs<GenericArg<'tcx>, &'tcx List<GenericArg<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_substs(xs))
    }

    pub fn mk_place_elems<I: InternAs<PlaceElem<'tcx>, &'tcx List<PlaceElem<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_place_elems(xs))
    }

    pub fn mk_substs_trait(
        self,
        self_ty: Ty<'tcx>,
        rest: impl IntoIterator<Item = GenericArg<'tcx>>,
    ) -> SubstsRef<'tcx> {
        self.mk_substs(iter::once(self_ty.into()).chain(rest))
    }

    pub fn mk_trait_ref(
        self,
        trait_def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> ty::TraitRef<'tcx> {
        let substs = self.check_substs(trait_def_id, substs);
        ty::TraitRef { def_id: trait_def_id, substs, _use_mk_trait_ref_instead: () }
    }

    pub fn mk_alias_ty(
        self,
        def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> ty::AliasTy<'tcx> {
        let substs = self.check_substs(def_id, substs);
        ty::AliasTy { def_id, substs, _use_mk_alias_ty_instead: () }
    }

    pub fn mk_bound_variable_kinds<
        I: InternAs<ty::BoundVariableKind, &'tcx List<ty::BoundVariableKind>>,
    >(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_bound_variable_kinds(xs))
    }

    /// Emit a lint at `span` from a lint struct (some type that implements `DecorateLint`,
    /// typically generated by `#[derive(LintDiagnostic)]`).
    pub fn emit_spanned_lint(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorator: impl for<'a> DecorateLint<'a, ()>,
    ) {
        let msg = decorator.msg();
        let (level, src) = self.lint_level_at_node(lint, hir_id);
        struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg, |diag| {
            decorator.decorate_lint(diag)
        })
    }

    /// Emit a lint at the appropriate level for a hir node, with an associated span.
    ///
    /// Return value of the `decorate` closure is ignored, see [`struct_lint_level`] for a detailed explanation.
    ///
    /// [`struct_lint_level`]: rustc_middle::lint::struct_lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    pub fn struct_span_lint_hir(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        decorate: impl for<'a, 'b> FnOnce(
            &'b mut DiagnosticBuilder<'a, ()>,
        ) -> &'b mut DiagnosticBuilder<'a, ()>,
    ) {
        let (level, src) = self.lint_level_at_node(lint, hir_id);
        struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg, decorate);
    }

    /// Emit a lint from a lint struct (some type that implements `DecorateLint`, typically
    /// generated by `#[derive(LintDiagnostic)]`).
    pub fn emit_lint(
        self,
        lint: &'static Lint,
        id: HirId,
        decorator: impl for<'a> DecorateLint<'a, ()>,
    ) {
        self.struct_lint_node(lint, id, decorator.msg(), |diag| decorator.decorate_lint(diag))
    }

    /// Emit a lint at the appropriate level for a hir node.
    ///
    /// Return value of the `decorate` closure is ignored, see [`struct_lint_level`] for a detailed explanation.
    ///
    /// [`struct_lint_level`]: rustc_middle::lint::struct_lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    pub fn struct_lint_node(
        self,
        lint: &'static Lint,
        id: HirId,
        msg: impl Into<DiagnosticMessage>,
        decorate: impl for<'a, 'b> FnOnce(
            &'b mut DiagnosticBuilder<'a, ()>,
        ) -> &'b mut DiagnosticBuilder<'a, ()>,
    ) {
        let (level, src) = self.lint_level_at_node(lint, id);
        struct_lint_level(self.sess, lint, level, src, None, msg, decorate);
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<&'tcx [TraitCandidate]> {
        let map = self.in_scope_traits_map(id.owner)?;
        let candidates = map.get(&id.local_id)?;
        Some(&*candidates)
    }

    pub fn named_region(self, id: HirId) -> Option<resolve_lifetime::Region> {
        debug!(?id, "named_region");
        self.named_region_map(id.owner).and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn is_late_bound(self, id: HirId) -> bool {
        self.is_late_bound_map(id.owner).map_or(false, |set| set.contains(&id.local_id))
    }

    pub fn late_bound_vars(self, id: HirId) -> &'tcx List<ty::BoundVariableKind> {
        self.mk_bound_variable_kinds(
            self.late_bound_vars_map(id.owner)
                .and_then(|map| map.get(&id.local_id).cloned())
                .unwrap_or_else(|| {
                    bug!("No bound vars found for {:?} ({:?})", self.hir().node_to_string(id), id)
                })
                .iter(),
        )
    }

    /// Whether the `def_id` counts as const fn in the current crate, considering all active
    /// feature gates
    pub fn is_const_fn(self, def_id: DefId) -> bool {
        if self.is_const_fn_raw(def_id) {
            match self.lookup_const_stability(def_id) {
                Some(stability) if stability.is_const_unstable() => {
                    // has a `rustc_const_unstable` attribute, check whether the user enabled the
                    // corresponding feature gate.
                    self.features()
                        .declared_lib_features
                        .iter()
                        .any(|&(sym, _)| sym == stability.feature)
                }
                // functions without const stability are either stable user written
                // const fn or the user is using feature gates and we thus don't
                // care what they do
                _ => true,
            }
        } else {
            false
        }
    }

    /// Whether the trait impl is marked const. This does not consider stability or feature gates.
    pub fn is_const_trait_impl_raw(self, def_id: DefId) -> bool {
        let Some(local_def_id) = def_id.as_local() else { return false };
        let hir_id = self.local_def_id_to_hir_id(local_def_id);
        let node = self.hir().get(hir_id);

        matches!(
            node,
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { constness: hir::Constness::Const, .. }),
                ..
            })
        )
    }
}

impl<'tcx> TyCtxtAt<'tcx> {
    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` to ensure it gets used.
    #[track_caller]
    pub fn ty_error(self) -> Ty<'tcx> {
        self.tcx.ty_error_with_message(self.span, "TyKind::Error constructed but no error reported")
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` with the given `msg to
    /// ensure it gets used.
    #[track_caller]
    pub fn ty_error_with_message(self, msg: &str) -> Ty<'tcx> {
        self.tcx.ty_error_with_message(self.span, msg)
    }

    pub fn mk_trait_ref(
        self,
        trait_lang_item: LangItem,
        substs: impl IntoIterator<Item = impl Into<ty::GenericArg<'tcx>>>,
    ) -> ty::TraitRef<'tcx> {
        let trait_def_id = self.require_lang_item(trait_lang_item, Some(self.span));
        self.tcx.mk_trait_ref(trait_def_id, substs)
    }
}

/// Parameter attributes that can only be determined by examining the body of a function instead
/// of just its signature.
///
/// These can be useful for optimization purposes when a function is directly called. We compute
/// them and store them into the crate metadata so that downstream crates can make use of them.
///
/// Right now, we only have `read_only`, but `no_capture` and `no_alias` might be useful in the
/// future.
#[derive(Clone, Copy, PartialEq, Debug, Default, TyDecodable, TyEncodable, HashStable)]
pub struct DeducedParamAttrs {
    /// The parameter is marked immutable in the function and contains no `UnsafeCell` (i.e. its
    /// type is freeze).
    pub read_only: bool,
}

pub fn provide(providers: &mut ty::query::Providers) {
    providers.module_reexports =
        |tcx, id| tcx.resolutions(()).reexport_map.get(&id).map(|v| &v[..]);
    providers.maybe_unused_trait_imports =
        |tcx, ()| &tcx.resolutions(()).maybe_unused_trait_imports;
    providers.maybe_unused_extern_crates =
        |tcx, ()| &tcx.resolutions(()).maybe_unused_extern_crates[..];
    providers.names_imported_by_glob_use = |tcx, id| {
        tcx.arena.alloc(tcx.resolutions(()).glob_map.get(&id).cloned().unwrap_or_default())
    };

    providers.extern_mod_stmt_cnum =
        |tcx, id| tcx.resolutions(()).extern_crate_map.get(&id).cloned();
    providers.is_panic_runtime = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.sess.contains_name(tcx.hir().krate_attrs(), sym::panic_runtime)
    };
    providers.is_compiler_builtins = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.sess.contains_name(tcx.hir().krate_attrs(), sym::compiler_builtins)
    };
    providers.has_panic_handler = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        // We want to check if the panic handler was defined in this crate
        tcx.lang_items().panic_impl().map_or(false, |did| did.is_local())
    };
    providers.source_span =
        |tcx, def_id| tcx.untracked.source_span.get(def_id).copied().unwrap_or(DUMMY_SP);
}
