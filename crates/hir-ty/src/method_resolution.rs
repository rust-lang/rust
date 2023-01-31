//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in rustc_hir_analysis/check/method/probe.rs.
use std::{ops::ControlFlow, sync::Arc};

use base_db::{CrateId, Edition};
use chalk_ir::{cast::Cast, Mutability, TyKind, UniverseIndex};
use hir_def::{
    data::ImplData, item_scope::ItemScope, lang_item::LangItem, nameres::DefMap, AssocItemId,
    BlockId, ConstId, FunctionId, HasModule, ImplId, ItemContainerId, Lookup, ModuleDefId,
    ModuleId, TraitId,
};
use hir_expand::name::Name;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use stdx::never;

use crate::{
    autoderef::{self, AutoderefKind},
    db::HirDatabase,
    from_foreign_def_id,
    infer::{unify::InferenceTable, Adjust, Adjustment, AutoBorrow, OverloadedDeref, PointerCast},
    primitive::{FloatTy, IntTy, UintTy},
    static_lifetime, to_chalk_trait_id,
    utils::all_super_traits,
    AdtId, Canonical, CanonicalVarKinds, DebruijnIndex, ForeignDefId, InEnvironment, Interner,
    Scalar, Substitution, TraitEnvironment, TraitRef, TraitRefExt, Ty, TyBuilder, TyExt,
};

/// This is used as a key for indexing impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TyFingerprint {
    // These are lang item impls:
    Str,
    Slice,
    Array,
    Never,
    RawPtr(Mutability),
    Scalar(Scalar),
    // These can have user-defined impls:
    Adt(hir_def::AdtId),
    Dyn(TraitId),
    ForeignType(ForeignDefId),
    // These only exist for trait impls
    Unit,
    Unnameable,
    Function(u32),
}

impl TyFingerprint {
    /// Creates a TyFingerprint for looking up an inherent impl. Only certain
    /// types can have inherent impls: if we have some `struct S`, we can have
    /// an `impl S`, but not `impl &S`. Hence, this will return `None` for
    /// reference types and such.
    pub fn for_inherent_impl(ty: &Ty) -> Option<TyFingerprint> {
        let fp = match ty.kind(Interner) {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Scalar(scalar) => TyFingerprint::Scalar(*scalar),
            TyKind::Adt(AdtId(adt), _) => TyFingerprint::Adt(*adt),
            TyKind::Raw(mutability, ..) => TyFingerprint::RawPtr(*mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(*alias_id),
            TyKind::Dyn(_) => ty.dyn_trait().map(TyFingerprint::Dyn)?,
            _ => return None,
        };
        Some(fp)
    }

    /// Creates a TyFingerprint for looking up a trait impl.
    pub fn for_trait_impl(ty: &Ty) -> Option<TyFingerprint> {
        let fp = match ty.kind(Interner) {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Scalar(scalar) => TyFingerprint::Scalar(*scalar),
            TyKind::Adt(AdtId(adt), _) => TyFingerprint::Adt(*adt),
            TyKind::Raw(mutability, ..) => TyFingerprint::RawPtr(*mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(*alias_id),
            TyKind::Dyn(_) => ty.dyn_trait().map(TyFingerprint::Dyn)?,
            TyKind::Ref(_, _, ty) => return TyFingerprint::for_trait_impl(ty),
            TyKind::Tuple(_, subst) => {
                let first_ty = subst.interned().get(0).map(|arg| arg.assert_ty_ref(Interner));
                match first_ty {
                    Some(ty) => return TyFingerprint::for_trait_impl(ty),
                    None => TyFingerprint::Unit,
                }
            }
            TyKind::AssociatedType(_, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Closure(_, _)
            | TyKind::Generator(..)
            | TyKind::GeneratorWitness(..) => TyFingerprint::Unnameable,
            TyKind::Function(fn_ptr) => {
                TyFingerprint::Function(fn_ptr.substitution.0.len(Interner) as u32)
            }
            TyKind::Alias(_)
            | TyKind::Placeholder(_)
            | TyKind::BoundVar(_)
            | TyKind::InferenceVar(_, _)
            | TyKind::Error => return None,
        };
        Some(fp)
    }
}

pub(crate) const ALL_INT_FPS: [TyFingerprint; 12] = [
    TyFingerprint::Scalar(Scalar::Int(IntTy::I8)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I16)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I32)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I64)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I128)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::Isize)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U8)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U16)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U32)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U64)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U128)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::Usize)),
];

pub(crate) const ALL_FLOAT_FPS: [TyFingerprint; 2] = [
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F32)),
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F64)),
];

/// Trait impls defined or available in some crate.
#[derive(Debug, Eq, PartialEq)]
pub struct TraitImpls {
    // If the `Option<TyFingerprint>` is `None`, the impl may apply to any self type.
    map: FxHashMap<TraitId, FxHashMap<Option<TyFingerprint>, Vec<ImplId>>>,
}

impl TraitImpls {
    pub(crate) fn trait_impls_in_crate_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("trait_impls_in_crate_query").detail(|| format!("{krate:?}"));
        let mut impls = Self { map: FxHashMap::default() };

        let crate_def_map = db.crate_def_map(krate);
        impls.collect_def_map(db, &crate_def_map);
        impls.shrink_to_fit();

        Arc::new(impls)
    }

    pub(crate) fn trait_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let _p = profile::span("trait_impls_in_block_query");
        let mut impls = Self { map: FxHashMap::default() };

        let block_def_map = db.block_def_map(block)?;
        impls.collect_def_map(db, &block_def_map);
        impls.shrink_to_fit();

        Some(Arc::new(impls))
    }

    pub(crate) fn trait_impls_in_deps_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("trait_impls_in_deps_query").detail(|| format!("{krate:?}"));
        let crate_graph = db.crate_graph();
        let mut res = Self { map: FxHashMap::default() };

        for krate in crate_graph.transitive_deps(krate) {
            res.merge(&db.trait_impls_in_crate(krate));
        }
        res.shrink_to_fit();

        Arc::new(res)
    }

    fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
        self.map.values_mut().for_each(|map| {
            map.shrink_to_fit();
            map.values_mut().for_each(Vec::shrink_to_fit);
        });
    }

    fn collect_def_map(&mut self, db: &dyn HirDatabase, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                let target_trait = match db.impl_trait(impl_id) {
                    Some(tr) => tr.skip_binders().hir_trait_id(),
                    None => continue,
                };
                let self_ty = db.impl_self_ty(impl_id);
                let self_ty_fp = TyFingerprint::for_trait_impl(self_ty.skip_binders());
                self.map
                    .entry(target_trait)
                    .or_default()
                    .entry(self_ty_fp)
                    .or_default()
                    .push(impl_id);
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in collect_unnamed_consts(db, &module_data.scope) {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db.upcast()) {
                    self.collect_def_map(db, &block_def_map);
                }
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (trait_, other_map) in &other.map {
            let map = self.map.entry(*trait_).or_default();
            for (fp, impls) in other_map {
                map.entry(*fp).or_default().extend(impls);
            }
        }
    }

    /// Queries all trait impls for the given type.
    pub fn for_self_ty_without_blanket_impls(
        &self,
        fp: TyFingerprint,
    ) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .values()
            .flat_map(move |impls| impls.get(&Some(fp)).into_iter())
            .flat_map(|it| it.iter().copied())
    }

    /// Queries all impls of the given trait.
    pub fn for_trait(&self, trait_: TraitId) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .get(&trait_)
            .into_iter()
            .flat_map(|map| map.values().flat_map(|v| v.iter().copied()))
    }

    /// Queries all impls of `trait_` that may apply to `self_ty`.
    pub fn for_trait_and_self_ty(
        &self,
        trait_: TraitId,
        self_ty: TyFingerprint,
    ) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .get(&trait_)
            .into_iter()
            .flat_map(move |map| map.get(&Some(self_ty)).into_iter().chain(map.get(&None)))
            .flat_map(|v| v.iter().copied())
    }

    pub fn all_impls(&self) -> impl Iterator<Item = ImplId> + '_ {
        self.map.values().flat_map(|map| map.values().flat_map(|v| v.iter().copied()))
    }
}

/// Inherent impls defined in some crate.
///
/// Inherent impls can only be defined in the crate that also defines the self type of the impl
/// (note that some primitives are considered to be defined by both libcore and liballoc).
///
/// This makes inherent impl lookup easier than trait impl lookup since we only have to consider a
/// single crate.
#[derive(Debug, Eq, PartialEq)]
pub struct InherentImpls {
    map: FxHashMap<TyFingerprint, Vec<ImplId>>,
}

impl InherentImpls {
    pub(crate) fn inherent_impls_in_crate_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let mut impls = Self { map: FxHashMap::default() };

        let crate_def_map = db.crate_def_map(krate);
        impls.collect_def_map(db, &crate_def_map);
        impls.shrink_to_fit();

        Arc::new(impls)
    }

    pub(crate) fn inherent_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let mut impls = Self { map: FxHashMap::default() };
        if let Some(block_def_map) = db.block_def_map(block) {
            impls.collect_def_map(db, &block_def_map);
            impls.shrink_to_fit();
            return Some(Arc::new(impls));
        }
        None
    }

    fn shrink_to_fit(&mut self) {
        self.map.values_mut().for_each(Vec::shrink_to_fit);
        self.map.shrink_to_fit();
    }

    fn collect_def_map(&mut self, db: &dyn HirDatabase, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                let data = db.impl_data(impl_id);
                if data.target_trait.is_some() {
                    continue;
                }

                let self_ty = db.impl_self_ty(impl_id);
                let fp = TyFingerprint::for_inherent_impl(self_ty.skip_binders());
                if let Some(fp) = fp {
                    self.map.entry(fp).or_default().push(impl_id);
                }
                // `fp` should only be `None` in error cases (either erroneous code or incomplete name resolution)
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in collect_unnamed_consts(db, &module_data.scope) {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db.upcast()) {
                    self.collect_def_map(db, &block_def_map);
                }
            }
        }
    }

    pub fn for_self_ty(&self, self_ty: &Ty) -> &[ImplId] {
        match TyFingerprint::for_inherent_impl(self_ty) {
            Some(fp) => self.map.get(&fp).map(|vec| vec.as_ref()).unwrap_or(&[]),
            None => &[],
        }
    }

    pub fn all_impls(&self) -> impl Iterator<Item = ImplId> + '_ {
        self.map.values().flat_map(|v| v.iter().copied())
    }
}

pub(crate) fn incoherent_inherent_impl_crates(
    db: &dyn HirDatabase,
    krate: CrateId,
    fp: TyFingerprint,
) -> SmallVec<[CrateId; 2]> {
    let _p = profile::span("inherent_impl_crates_query");
    let mut res = SmallVec::new();
    let crate_graph = db.crate_graph();

    // should pass crate for finger print and do reverse deps

    for krate in crate_graph.transitive_deps(krate) {
        let impls = db.inherent_impls_in_crate(krate);
        if impls.map.get(&fp).map_or(false, |v| !v.is_empty()) {
            res.push(krate);
        }
    }

    res
}

fn collect_unnamed_consts<'a>(
    db: &'a dyn HirDatabase,
    scope: &'a ItemScope,
) -> impl Iterator<Item = ConstId> + 'a {
    let unnamed_consts = scope.unnamed_consts();

    // FIXME: Also treat consts named `_DERIVE_*` as unnamed, since synstructure generates those.
    // Should be removed once synstructure stops doing that.
    let synstructure_hack_consts = scope.values().filter_map(|(item, _)| match item {
        ModuleDefId::ConstId(id) => {
            let loc = id.lookup(db.upcast());
            let item_tree = loc.id.item_tree(db.upcast());
            if item_tree[loc.id.value]
                .name
                .as_ref()
                .map_or(false, |n| n.to_smol_str().starts_with("_DERIVE_"))
            {
                Some(id)
            } else {
                None
            }
        }
        _ => None,
    });

    unnamed_consts.chain(synstructure_hack_consts)
}

pub fn def_crates(
    db: &dyn HirDatabase,
    ty: &Ty,
    cur_crate: CrateId,
) -> Option<SmallVec<[CrateId; 2]>> {
    match ty.kind(Interner) {
        &TyKind::Adt(AdtId(def_id), _) => {
            let rustc_has_incoherent_inherent_impls = match def_id {
                hir_def::AdtId::StructId(id) => {
                    db.struct_data(id).rustc_has_incoherent_inherent_impls
                }
                hir_def::AdtId::UnionId(id) => {
                    db.union_data(id).rustc_has_incoherent_inherent_impls
                }
                hir_def::AdtId::EnumId(id) => db.enum_data(id).rustc_has_incoherent_inherent_impls,
            };
            Some(if rustc_has_incoherent_inherent_impls {
                db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::Adt(def_id))
            } else {
                smallvec![def_id.module(db.upcast()).krate()]
            })
        }
        &TyKind::Foreign(id) => {
            let alias = from_foreign_def_id(id);
            Some(if db.type_alias_data(alias).rustc_has_incoherent_inherent_impls {
                db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::ForeignType(id))
            } else {
                smallvec![alias.module(db.upcast()).krate()]
            })
        }
        TyKind::Dyn(_) => {
            let trait_id = ty.dyn_trait()?;
            Some(if db.trait_data(trait_id).rustc_has_incoherent_inherent_impls {
                db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::Dyn(trait_id))
            } else {
                smallvec![trait_id.module(db.upcast()).krate()]
            })
        }
        // for primitives, there may be impls in various places (core and alloc
        // mostly). We just check the whole crate graph for crates with impls
        // (cached behind a query).
        TyKind::Scalar(_)
        | TyKind::Str
        | TyKind::Slice(_)
        | TyKind::Array(..)
        | TyKind::Raw(..) => Some(db.incoherent_inherent_impl_crates(
            cur_crate,
            TyFingerprint::for_inherent_impl(ty).expect("fingerprint for primitive"),
        )),
        _ => None,
    }
}

pub fn lang_items_for_bin_op(op: syntax::ast::BinaryOp) -> Option<(Name, LangItem)> {
    use hir_expand::name;
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (name![add], LangItem::Add),
            ArithOp::Mul => (name![mul], LangItem::Mul),
            ArithOp::Sub => (name![sub], LangItem::Sub),
            ArithOp::Div => (name![div], LangItem::Div),
            ArithOp::Rem => (name![rem], LangItem::Rem),
            ArithOp::Shl => (name![shl], LangItem::Shl),
            ArithOp::Shr => (name![shr], LangItem::Shr),
            ArithOp::BitXor => (name![bitxor], LangItem::BitXor),
            ArithOp::BitOr => (name![bitor], LangItem::BitOr),
            ArithOp::BitAnd => (name![bitand], LangItem::BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (name![add_assign], LangItem::AddAssign),
            ArithOp::Mul => (name![mul_assign], LangItem::MulAssign),
            ArithOp::Sub => (name![sub_assign], LangItem::SubAssign),
            ArithOp::Div => (name![div_assign], LangItem::DivAssign),
            ArithOp::Rem => (name![rem_assign], LangItem::RemAssign),
            ArithOp::Shl => (name![shl_assign], LangItem::ShlAssign),
            ArithOp::Shr => (name![shr_assign], LangItem::ShrAssign),
            ArithOp::BitXor => (name![bitxor_assign], LangItem::BitXorAssign),
            ArithOp::BitOr => (name![bitor_assign], LangItem::BitOrAssign),
            ArithOp::BitAnd => (name![bitand_assign], LangItem::BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (name![eq], LangItem::PartialEq),
            CmpOp::Eq { negated: true } => (name![ne], LangItem::PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (name![le], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (name![lt], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (name![ge], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (name![gt], LangItem::PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}

/// Look up the method with the given name.
pub(crate) fn lookup_method(
    db: &dyn HirDatabase,
    ty: &Canonical<Ty>,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: &Name,
) -> Option<(ReceiverAdjustments, FunctionId, bool)> {
    let mut not_visible = None;
    let res = iterate_method_candidates(
        ty,
        db,
        env,
        traits_in_scope,
        visible_from_module,
        Some(name),
        LookupMode::MethodCall,
        |adjustments, f, visible| match f {
            AssocItemId::FunctionId(f) if visible => Some((adjustments, f, true)),
            AssocItemId::FunctionId(f) if not_visible.is_none() => {
                not_visible = Some((adjustments, f, false));
                None
            }
            _ => None,
        },
    );
    res.or(not_visible)
}

/// Whether we're looking up a dotted method call (like `v.len()`) or a path
/// (like `Vec::new`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LookupMode {
    /// Looking up a method call like `v.len()`: We only consider candidates
    /// that have a `self` parameter, and do autoderef.
    MethodCall,
    /// Looking up a path like `Vec::new` or `Vec::default`: We consider all
    /// candidates including associated constants, but don't do autoderef.
    Path,
}

#[derive(Clone, Copy)]
pub enum VisibleFromModule {
    /// Filter for results that are visible from the given module
    Filter(ModuleId),
    /// Include impls from the given block.
    IncludeBlock(BlockId),
    /// Do nothing special in regards visibility
    None,
}

impl From<Option<ModuleId>> for VisibleFromModule {
    fn from(module: Option<ModuleId>) -> Self {
        match module {
            Some(module) => Self::Filter(module),
            None => Self::None,
        }
    }
}

impl From<Option<BlockId>> for VisibleFromModule {
    fn from(block: Option<BlockId>) -> Self {
        match block {
            Some(block) => Self::IncludeBlock(block),
            None => Self::None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverAdjustments {
    autoref: Option<Mutability>,
    autoderefs: usize,
    unsize_array: bool,
}

impl ReceiverAdjustments {
    pub(crate) fn apply(&self, table: &mut InferenceTable<'_>, ty: Ty) -> (Ty, Vec<Adjustment>) {
        let mut ty = table.resolve_ty_shallow(&ty);
        let mut adjust = Vec::new();
        for _ in 0..self.autoderefs {
            match autoderef::autoderef_step(table, ty.clone()) {
                None => {
                    never!("autoderef not possible for {:?}", ty);
                    ty = TyKind::Error.intern(Interner);
                    break;
                }
                Some((kind, new_ty)) => {
                    ty = new_ty.clone();
                    adjust.push(Adjustment {
                        kind: Adjust::Deref(match kind {
                            // FIXME should we know the mutability here?
                            AutoderefKind::Overloaded => Some(OverloadedDeref(Mutability::Not)),
                            AutoderefKind::Builtin => None,
                        }),
                        target: new_ty,
                    });
                }
            }
        }
        if let Some(m) = self.autoref {
            ty = TyKind::Ref(m, static_lifetime(), ty).intern(Interner);
            adjust
                .push(Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(m)), target: ty.clone() });
        }
        if self.unsize_array {
            ty = 'x: {
                if let TyKind::Ref(m, l, inner) = ty.kind(Interner) {
                    if let TyKind::Array(inner, _) = inner.kind(Interner) {
                        break 'x TyKind::Ref(
                            m.clone(),
                            l.clone(),
                            TyKind::Slice(inner.clone()).intern(Interner),
                        )
                        .intern(Interner);
                    }
                }
                never!("unsize_array with non-reference-to-array {:?}", ty);
                ty
            };
            adjust.push(Adjustment {
                kind: Adjust::Pointer(PointerCast::Unsize),
                target: ty.clone(),
            });
        }
        (ty, adjust)
    }

    fn with_autoref(&self, m: Mutability) -> ReceiverAdjustments {
        Self { autoref: Some(m), ..*self }
    }
}

// This would be nicer if it just returned an iterator, but that runs into
// lifetime problems, because we need to borrow temp `CrateImplDefs`.
// FIXME add a context type here?
pub(crate) fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(ReceiverAdjustments, AssocItemId, bool) -> Option<T>,
) -> Option<T> {
    let mut slot = None;
    iterate_method_candidates_dyn(
        ty,
        db,
        env,
        traits_in_scope,
        visible_from_module,
        name,
        mode,
        &mut |adj, item, visible| {
            assert!(slot.is_none());
            if let Some(it) = callback(adj, item, visible) {
                slot = Some(it);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    );
    slot
}

pub fn lookup_impl_const(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    const_id: ConstId,
    subs: Substitution,
) -> ConstId {
    let trait_id = match const_id.lookup(db.upcast()).container {
        ItemContainerId::TraitId(id) => id,
        _ => return const_id,
    };
    let substitution = Substitution::from_iter(Interner, subs.iter(Interner));
    let trait_ref = TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution };

    let const_data = db.const_data(const_id);
    let name = match const_data.name.as_ref() {
        Some(name) => name,
        None => return const_id,
    };

    lookup_impl_assoc_item_for_trait_ref(trait_ref, db, env, name)
        .and_then(|assoc| if let AssocItemId::ConstId(id) = assoc { Some(id) } else { None })
        .unwrap_or(const_id)
}

/// Looks up the impl method that actually runs for the trait method `func`.
///
/// Returns `func` if it's not a method defined in a trait or the lookup failed.
pub fn lookup_impl_method(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    func: FunctionId,
    fn_subst: Substitution,
) -> FunctionId {
    let trait_id = match func.lookup(db.upcast()).container {
        ItemContainerId::TraitId(id) => id,
        _ => return func,
    };
    let trait_params = db.generic_params(trait_id.into()).type_or_consts.len();
    let fn_params = fn_subst.len(Interner) - trait_params;
    let trait_ref = TraitRef {
        trait_id: to_chalk_trait_id(trait_id),
        substitution: Substitution::from_iter(Interner, fn_subst.iter(Interner).skip(fn_params)),
    };

    let name = &db.function_data(func).name;
    lookup_impl_assoc_item_for_trait_ref(trait_ref, db, env, name)
        .and_then(|assoc| if let AssocItemId::FunctionId(id) = assoc { Some(id) } else { None })
        .unwrap_or(func)
}

fn lookup_impl_assoc_item_for_trait_ref(
    trait_ref: TraitRef,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    name: &Name,
) -> Option<AssocItemId> {
    let self_ty = trait_ref.self_type_parameter(Interner);
    let self_ty_fp = TyFingerprint::for_trait_impl(&self_ty)?;
    let impls = db.trait_impls_in_deps(env.krate);
    let impls = impls.for_trait_and_self_ty(trait_ref.hir_trait_id(), self_ty_fp);

    let table = InferenceTable::new(db, env);

    let impl_data = find_matching_impl(impls, table, trait_ref)?;
    impl_data.items.iter().find_map(|&it| match it {
        AssocItemId::FunctionId(f) => {
            (db.function_data(f).name == *name).then_some(AssocItemId::FunctionId(f))
        }
        AssocItemId::ConstId(c) => db
            .const_data(c)
            .name
            .as_ref()
            .map(|n| n == name)
            .and_then(|result| if result { Some(AssocItemId::ConstId(c)) } else { None }),
        AssocItemId::TypeAliasId(_) => None,
    })
}

fn find_matching_impl(
    mut impls: impl Iterator<Item = ImplId>,
    mut table: InferenceTable<'_>,
    actual_trait_ref: TraitRef,
) -> Option<Arc<ImplData>> {
    let db = table.db;
    loop {
        let impl_ = impls.next()?;
        let r = table.run_in_snapshot(|table| {
            let impl_data = db.impl_data(impl_);
            let impl_substs =
                TyBuilder::subst_for_def(db, impl_, None).fill_with_inference_vars(table).build();
            let trait_ref = db
                .impl_trait(impl_)
                .expect("non-trait method in find_matching_impl")
                .substitute(Interner, &impl_substs);

            if !table.unify(&trait_ref, &actual_trait_ref) {
                return None;
            }

            let wcs = crate::chalk_db::convert_where_clauses(db, impl_.into(), &impl_substs)
                .into_iter()
                .map(|b| b.cast(Interner));
            let goal = crate::Goal::all(Interner, wcs);
            table.try_obligation(goal).map(|_| impl_data)
        });
        if r.is_some() {
            break r;
        }
    }
}

pub fn iterate_path_candidates(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn FnMut(AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    iterate_method_candidates_dyn(
        ty,
        db,
        env,
        traits_in_scope,
        visible_from_module,
        name,
        LookupMode::Path,
        // the adjustments are not relevant for path lookup
        &mut |_, id, _| callback(id),
    )
}

pub fn iterate_method_candidates_dyn(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    match mode {
        LookupMode::MethodCall => {
            // For method calls, rust first does any number of autoderef, and
            // then one autoref (i.e. when the method takes &self or &mut self).
            // Note that when we've got a receiver like &S, even if the method
            // we find in the end takes &self, we still do the autoderef step
            // (just as rustc does an autoderef and then autoref again).

            // We have to be careful about the order we're looking at candidates
            // in here. Consider the case where we're resolving `x.clone()`
            // where `x: &Vec<_>`. This resolves to the clone method with self
            // type `Vec<_>`, *not* `&_`. I.e. we need to consider methods where
            // the receiver type exactly matches before cases where we have to
            // do autoref. But in the autoderef steps, the `&_` self type comes
            // up *before* the `Vec<_>` self type.
            //
            // On the other hand, we don't want to just pick any by-value method
            // before any by-autoref method; it's just that we need to consider
            // the methods by autoderef order of *receiver types*, not *self
            // types*.

            let mut table = InferenceTable::new(db, env.clone());
            let ty = table.instantiate_canonical(ty.clone());
            let (deref_chain, adj) = autoderef_method_receiver(&mut table, ty);

            let result = deref_chain.into_iter().zip(adj).try_for_each(|(receiver_ty, adj)| {
                iterate_method_candidates_with_autoref(
                    &receiver_ty,
                    adj,
                    db,
                    env.clone(),
                    traits_in_scope,
                    visible_from_module,
                    name,
                    callback,
                )
            });
            result
        }
        LookupMode::Path => {
            // No autoderef for path lookups
            iterate_method_candidates_for_self_ty(
                ty,
                db,
                env,
                traits_in_scope,
                visible_from_module,
                name,
                callback,
            )
        }
    }
}

fn iterate_method_candidates_with_autoref(
    receiver_ty: &Canonical<Ty>,
    first_adjustment: ReceiverAdjustments,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if receiver_ty.value.is_general_var(Interner, &receiver_ty.binders) {
        // don't try to resolve methods on unknown types
        return ControlFlow::Continue(());
    }

    iterate_method_candidates_by_receiver(
        receiver_ty,
        first_adjustment.clone(),
        db,
        env.clone(),
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )?;

    let refed = Canonical {
        value: TyKind::Ref(Mutability::Not, static_lifetime(), receiver_ty.value.clone())
            .intern(Interner),
        binders: receiver_ty.binders.clone(),
    };

    iterate_method_candidates_by_receiver(
        &refed,
        first_adjustment.with_autoref(Mutability::Not),
        db,
        env.clone(),
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )?;

    let ref_muted = Canonical {
        value: TyKind::Ref(Mutability::Mut, static_lifetime(), receiver_ty.value.clone())
            .intern(Interner),
        binders: receiver_ty.binders.clone(),
    };

    iterate_method_candidates_by_receiver(
        &ref_muted,
        first_adjustment.with_autoref(Mutability::Mut),
        db,
        env,
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )
}

fn iterate_method_candidates_by_receiver(
    receiver_ty: &Canonical<Ty>,
    receiver_adjustments: ReceiverAdjustments,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let mut table = InferenceTable::new(db, env);
    let receiver_ty = table.instantiate_canonical(receiver_ty.clone());
    let snapshot = table.snapshot();
    // We're looking for methods with *receiver* type receiver_ty. These could
    // be found in any of the derefs of receiver_ty, so we have to go through
    // that.
    let mut autoderef = autoderef::Autoderef::new(&mut table, receiver_ty.clone());
    while let Some((self_ty, _)) = autoderef.next() {
        iterate_inherent_methods(
            &self_ty,
            autoderef.table,
            name,
            Some(&receiver_ty),
            Some(receiver_adjustments.clone()),
            visible_from_module,
            &mut callback,
        )?
    }

    table.rollback_to(snapshot);

    let mut autoderef = autoderef::Autoderef::new(&mut table, receiver_ty.clone());
    while let Some((self_ty, _)) = autoderef.next() {
        iterate_trait_method_candidates(
            &self_ty,
            autoderef.table,
            traits_in_scope,
            name,
            Some(&receiver_ty),
            Some(receiver_adjustments.clone()),
            &mut callback,
        )?
    }

    ControlFlow::Continue(())
}

fn iterate_method_candidates_for_self_ty(
    self_ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let mut table = InferenceTable::new(db, env);
    let self_ty = table.instantiate_canonical(self_ty.clone());
    iterate_inherent_methods(
        &self_ty,
        &mut table,
        name,
        None,
        None,
        visible_from_module,
        &mut callback,
    )?;
    iterate_trait_method_candidates(
        &self_ty,
        &mut table,
        traits_in_scope,
        name,
        None,
        None,
        callback,
    )
}

fn iterate_trait_method_candidates(
    self_ty: &Ty,
    table: &mut InferenceTable<'_>,
    traits_in_scope: &FxHashSet<TraitId>,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;
    let env = table.trait_env.clone();
    let self_is_array = matches!(self_ty.kind(Interner), chalk_ir::TyKind::Array(..));

    let canonical_self_ty = table.canonicalize(self_ty.clone()).value;

    'traits: for &t in traits_in_scope {
        let data = db.trait_data(t);

        // Traits annotated with `#[rustc_skip_array_during_method_dispatch]` are skipped during
        // method resolution, if the receiver is an array, and we're compiling for editions before
        // 2021.
        // This is to make `[a].into_iter()` not break code with the new `IntoIterator` impl for
        // arrays.
        if data.skip_array_during_method_dispatch && self_is_array {
            // FIXME: this should really be using the edition of the method name's span, in case it
            // comes from a macro
            if db.crate_graph()[env.krate].edition < Edition::Edition2021 {
                continue;
            }
        }

        // we'll be lazy about checking whether the type implements the
        // trait, but if we find out it doesn't, we'll skip the rest of the
        // iteration
        let mut known_implemented = false;
        for &(_, item) in data.items.iter() {
            // Don't pass a `visible_from_module` down to `is_valid_candidate`,
            // since only inherent methods should be included into visibility checking.
            let visible = match is_valid_candidate(table, name, receiver_ty, item, self_ty, None) {
                IsValidCandidate::Yes => true,
                IsValidCandidate::NotVisible => false,
                IsValidCandidate::No => continue,
            };
            if !known_implemented {
                let goal = generic_implements_goal(db, env.clone(), t, &canonical_self_ty);
                if db.trait_solve(env.krate, goal.cast(Interner)).is_none() {
                    continue 'traits;
                }
            }
            known_implemented = true;
            callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
        }
    }
    ControlFlow::Continue(())
}

fn iterate_inherent_methods(
    self_ty: &Ty,
    table: &mut InferenceTable<'_>,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    visible_from_module: VisibleFromModule,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;
    let env = table.trait_env.clone();

    // For trait object types and placeholder types with trait bounds, the methods of the trait and
    // its super traits are considered inherent methods. This matters because these methods have
    // higher priority than the other traits' methods, which would be considered in
    // `iterate_trait_method_candidates()` only after this function.
    match self_ty.kind(Interner) {
        TyKind::Placeholder(_) => {
            let env = table.trait_env.clone();
            let traits = env
                .traits_in_scope_from_clauses(self_ty.clone())
                .flat_map(|t| all_super_traits(db.upcast(), t));
            iterate_inherent_trait_methods(
                self_ty,
                table,
                name,
                receiver_ty,
                receiver_adjustments.clone(),
                callback,
                traits,
            )?;
        }
        TyKind::Dyn(_) => {
            if let Some(principal_trait) = self_ty.dyn_trait() {
                let traits = all_super_traits(db.upcast(), principal_trait);
                iterate_inherent_trait_methods(
                    self_ty,
                    table,
                    name,
                    receiver_ty,
                    receiver_adjustments.clone(),
                    callback,
                    traits.into_iter(),
                )?;
            }
        }
        _ => {}
    }

    let def_crates = match def_crates(db, self_ty, env.krate) {
        Some(k) => k,
        None => return ControlFlow::Continue(()),
    };

    let (module, mut block) = match visible_from_module {
        VisibleFromModule::Filter(module) => (Some(module), module.containing_block()),
        VisibleFromModule::IncludeBlock(block) => (None, Some(block)),
        VisibleFromModule::None => (None, None),
    };

    while let Some(block_id) = block {
        if let Some(impls) = db.inherent_impls_in_block(block_id) {
            impls_for_self_ty(
                &impls,
                self_ty,
                table,
                name,
                receiver_ty,
                receiver_adjustments.clone(),
                module,
                callback,
            )?;
        }

        block = db
            .block_def_map(block_id)
            .and_then(|map| map.parent())
            .and_then(|module| module.containing_block());
    }

    for krate in def_crates {
        let impls = db.inherent_impls_in_crate(krate);
        impls_for_self_ty(
            &impls,
            self_ty,
            table,
            name,
            receiver_ty,
            receiver_adjustments.clone(),
            module,
            callback,
        )?;
    }
    return ControlFlow::Continue(());

    fn iterate_inherent_trait_methods(
        self_ty: &Ty,
        table: &mut InferenceTable<'_>,
        name: Option<&Name>,
        receiver_ty: Option<&Ty>,
        receiver_adjustments: Option<ReceiverAdjustments>,
        callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
        traits: impl Iterator<Item = TraitId>,
    ) -> ControlFlow<()> {
        let db = table.db;
        for t in traits {
            let data = db.trait_data(t);
            for &(_, item) in data.items.iter() {
                // We don't pass `visible_from_module` as all trait items should be visible.
                let visible =
                    match is_valid_candidate(table, name, receiver_ty, item, self_ty, None) {
                        IsValidCandidate::Yes => true,
                        IsValidCandidate::NotVisible => false,
                        IsValidCandidate::No => continue,
                    };
                callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
            }
        }
        ControlFlow::Continue(())
    }

    fn impls_for_self_ty(
        impls: &InherentImpls,
        self_ty: &Ty,
        table: &mut InferenceTable<'_>,
        name: Option<&Name>,
        receiver_ty: Option<&Ty>,
        receiver_adjustments: Option<ReceiverAdjustments>,
        visible_from_module: Option<ModuleId>,
        callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let db = table.db;
        let impls_for_self_ty = impls.for_self_ty(self_ty);
        for &impl_def in impls_for_self_ty {
            for &item in &db.impl_data(impl_def).items {
                let visible = match is_valid_candidate(
                    table,
                    name,
                    receiver_ty,
                    item,
                    self_ty,
                    visible_from_module,
                ) {
                    IsValidCandidate::Yes => true,
                    IsValidCandidate::NotVisible => false,
                    IsValidCandidate::No => continue,
                };
                callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
            }
        }
        ControlFlow::Continue(())
    }
}

/// Returns the receiver type for the index trait call.
pub fn resolve_indexing_op(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    ty: Canonical<Ty>,
    index_trait: TraitId,
) -> Option<ReceiverAdjustments> {
    let mut table = InferenceTable::new(db, env.clone());
    let ty = table.instantiate_canonical(ty);
    let (deref_chain, adj) = autoderef_method_receiver(&mut table, ty);
    for (ty, adj) in deref_chain.into_iter().zip(adj) {
        let goal = generic_implements_goal(db, env.clone(), index_trait, &ty);
        if db.trait_solve(env.krate, goal.cast(Interner)).is_some() {
            return Some(adj);
        }
    }
    None
}

macro_rules! check_that {
    ($cond:expr) => {
        if !$cond {
            return IsValidCandidate::No;
        }
    };
}

fn is_valid_candidate(
    table: &mut InferenceTable<'_>,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    item: AssocItemId,
    self_ty: &Ty,
    visible_from_module: Option<ModuleId>,
) -> IsValidCandidate {
    let db = table.db;
    match item {
        AssocItemId::FunctionId(m) => {
            is_valid_fn_candidate(table, m, name, receiver_ty, self_ty, visible_from_module)
        }
        AssocItemId::ConstId(c) => {
            let data = db.const_data(c);
            check_that!(receiver_ty.is_none());

            check_that!(name.map_or(true, |n| data.name.as_ref() == Some(n)));

            if let Some(from_module) = visible_from_module {
                if !db.const_visibility(c).is_visible_from(db.upcast(), from_module) {
                    cov_mark::hit!(const_candidate_not_visible);
                    return IsValidCandidate::NotVisible;
                }
            }
            if let ItemContainerId::ImplId(impl_id) = c.lookup(db.upcast()).container {
                let self_ty_matches = table.run_in_snapshot(|table| {
                    let expected_self_ty = TyBuilder::impl_self_ty(db, impl_id)
                        .fill_with_inference_vars(table)
                        .build();
                    table.unify(&expected_self_ty, self_ty)
                });
                if !self_ty_matches {
                    cov_mark::hit!(const_candidate_self_type_mismatch);
                    return IsValidCandidate::No;
                }
            }
            IsValidCandidate::Yes
        }
        _ => IsValidCandidate::No,
    }
}

enum IsValidCandidate {
    Yes,
    No,
    NotVisible,
}

fn is_valid_fn_candidate(
    table: &mut InferenceTable<'_>,
    fn_id: FunctionId,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    self_ty: &Ty,
    visible_from_module: Option<ModuleId>,
) -> IsValidCandidate {
    let db = table.db;
    let data = db.function_data(fn_id);

    check_that!(name.map_or(true, |n| n == &data.name));
    if let Some(from_module) = visible_from_module {
        if !db.function_visibility(fn_id).is_visible_from(db.upcast(), from_module) {
            cov_mark::hit!(autoderef_candidate_not_visible);
            return IsValidCandidate::NotVisible;
        }
    }
    table.run_in_snapshot(|table| {
        let container = fn_id.lookup(db.upcast()).container;
        let (impl_subst, expect_self_ty) = match container {
            ItemContainerId::ImplId(it) => {
                let subst =
                    TyBuilder::subst_for_def(db, it, None).fill_with_inference_vars(table).build();
                let self_ty = db.impl_self_ty(it).substitute(Interner, &subst);
                (subst, self_ty)
            }
            ItemContainerId::TraitId(it) => {
                let subst =
                    TyBuilder::subst_for_def(db, it, None).fill_with_inference_vars(table).build();
                let self_ty = subst.at(Interner, 0).assert_ty_ref(Interner).clone();
                (subst, self_ty)
            }
            _ => unreachable!(),
        };

        let fn_subst = TyBuilder::subst_for_def(db, fn_id, Some(impl_subst.clone()))
            .fill_with_inference_vars(table)
            .build();

        check_that!(table.unify(&expect_self_ty, self_ty));

        if let Some(receiver_ty) = receiver_ty {
            check_that!(data.has_self_param());

            let sig = db.callable_item_signature(fn_id.into());
            let expected_receiver =
                sig.map(|s| s.params()[0].clone()).substitute(Interner, &fn_subst);

            check_that!(table.unify(receiver_ty, &expected_receiver));
        }

        if let ItemContainerId::ImplId(impl_id) = container {
            // We need to consider the bounds on the impl to distinguish functions of the same name
            // for a type.
            let predicates = db.generic_predicates(impl_id.into());
            let valid = predicates
                .iter()
                .map(|predicate| {
                    let (p, b) = predicate
                        .clone()
                        .substitute(Interner, &impl_subst)
                        // Skipping the inner binders is ok, as we don't handle quantified where
                        // clauses yet.
                        .into_value_and_skipped_binders();
                    stdx::always!(b.len(Interner) == 0);
                    p
                })
                // It's ok to get ambiguity here, as we may not have enough information to prove
                // obligations. We'll check if the user is calling the selected method properly
                // later anyway.
                .all(|p| table.try_obligation(p.cast(Interner)).is_some());
            match valid {
                true => IsValidCandidate::Yes,
                false => IsValidCandidate::No,
            }
        } else {
            // For `ItemContainerId::TraitId`, we check if `self_ty` implements the trait in
            // `iterate_trait_method_candidates()`.
            // For others, this function shouldn't be called.
            IsValidCandidate::Yes
        }
    })
}

pub fn implements_trait(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env.clone(), trait_, ty);
    let solution = db.trait_solve(env.krate, goal.cast(Interner));

    solution.is_some()
}

pub fn implements_trait_unique(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env.clone(), trait_, ty);
    let solution = db.trait_solve(env.krate, goal.cast(Interner));

    matches!(solution, Some(crate::Solution::Unique(_)))
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query Chalk with it.
fn generic_implements_goal(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: TraitId,
    self_ty: &Canonical<Ty>,
) -> Canonical<InEnvironment<super::DomainGoal>> {
    let mut kinds = self_ty.binders.interned().to_vec();
    let trait_ref = TyBuilder::trait_ref(db, trait_)
        .push(self_ty.value.clone())
        .fill_with_bound_vars(DebruijnIndex::INNERMOST, kinds.len())
        .build();
    kinds.extend(trait_ref.substitution.iter(Interner).skip(1).map(|x| {
        let vk = match x.data(Interner) {
            chalk_ir::GenericArgData::Ty(_) => {
                chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General)
            }
            chalk_ir::GenericArgData::Lifetime(_) => chalk_ir::VariableKind::Lifetime,
            chalk_ir::GenericArgData::Const(c) => {
                chalk_ir::VariableKind::Const(c.data(Interner).ty.clone())
            }
        };
        chalk_ir::WithKind::new(vk, UniverseIndex::ROOT)
    }));
    let obligation = trait_ref.cast(Interner);
    Canonical {
        binders: CanonicalVarKinds::from_iter(Interner, kinds),
        value: InEnvironment::new(&env.env, obligation),
    }
}

fn autoderef_method_receiver(
    table: &mut InferenceTable<'_>,
    ty: Ty,
) -> (Vec<Canonical<Ty>>, Vec<ReceiverAdjustments>) {
    let (mut deref_chain, mut adjustments): (Vec<_>, Vec<_>) = (Vec::new(), Vec::new());
    let mut autoderef = autoderef::Autoderef::new(table, ty);
    while let Some((ty, derefs)) = autoderef.next() {
        deref_chain.push(autoderef.table.canonicalize(ty).value);
        adjustments.push(ReceiverAdjustments {
            autoref: None,
            autoderefs: derefs,
            unsize_array: false,
        });
    }
    // As a last step, we can do array unsizing (that's the only unsizing that rustc does for method receivers!)
    if let (Some((TyKind::Array(parameters, _), binders)), Some(adj)) = (
        deref_chain.last().map(|ty| (ty.value.kind(Interner), ty.binders.clone())),
        adjustments.last().cloned(),
    ) {
        let unsized_ty = TyKind::Slice(parameters.clone()).intern(Interner);
        deref_chain.push(Canonical { value: unsized_ty, binders });
        adjustments.push(ReceiverAdjustments { unsize_array: true, ..adj });
    }
    (deref_chain, adjustments)
}
