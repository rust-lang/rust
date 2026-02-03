//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use hir_expand::name::Name;
use intern::{Symbol, sym};
use stdx::impl_from;

use crate::{
    AdtId, AssocItemId, AttrDefId, Crate, EnumId, EnumVariantId, FunctionId, ImplId, MacroId,
    ModuleDefId, StaticId, StructId, TraitId, TypeAliasId, UnionId,
    attrs::AttrFlags,
    db::DefDatabase,
    nameres::{DefMap, assoc::TraitItems, crate_def_map, crate_local_def_map},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    EnumId(EnumId),
    FunctionId(FunctionId),
    ImplId(ImplId),
    StaticId(StaticId),
    StructId(StructId),
    UnionId(UnionId),
    TypeAliasId(TypeAliasId),
    TraitId(TraitId),
    EnumVariantId(EnumVariantId),
}

impl_from!(
    EnumId, FunctionId, ImplId, StaticId, StructId, UnionId, TypeAliasId, TraitId, EnumVariantId for LangItemTarget
);

/// Salsa query. This will look for lang items in a specific crate.
#[salsa_macros::tracked(returns(as_deref))]
pub fn crate_lang_items(db: &dyn DefDatabase, krate: Crate) -> Option<Box<LangItems>> {
    let _p = tracing::info_span!("crate_lang_items_query").entered();

    let mut lang_items = LangItems::default();

    let crate_def_map = crate_def_map(db, krate);

    if !crate_def_map.is_unstable_feature_enabled(&sym::lang_items) {
        return None;
    }

    for (_, module_data) in crate_def_map.modules() {
        for impl_def in module_data.scope.inherent_impls() {
            lang_items.collect_lang_item(db, impl_def);
            for &(_, assoc) in impl_def.impl_items(db).items.iter() {
                match assoc {
                    AssocItemId::FunctionId(f) => lang_items.collect_lang_item(db, f),
                    AssocItemId::TypeAliasId(t) => lang_items.collect_lang_item(db, t),
                    AssocItemId::ConstId(_) => (),
                }
            }
        }

        for def in module_data.scope.declarations() {
            match def {
                ModuleDefId::TraitId(trait_) => {
                    lang_items.collect_lang_item(db, trait_);
                    TraitItems::query(db, trait_).items.iter().for_each(|&(_, assoc_id)| {
                        match assoc_id {
                            AssocItemId::FunctionId(f) => {
                                lang_items.collect_lang_item(db, f);
                            }
                            AssocItemId::TypeAliasId(alias) => {
                                lang_items.collect_lang_item(db, alias)
                            }
                            AssocItemId::ConstId(_) => {}
                        }
                    });
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    lang_items.collect_lang_item(db, e);
                    e.enum_variants(db).variants.iter().for_each(|&(id, _, _)| {
                        lang_items.collect_lang_item(db, id);
                    });
                }
                ModuleDefId::AdtId(AdtId::StructId(s)) => {
                    lang_items.collect_lang_item(db, s);
                }
                ModuleDefId::AdtId(AdtId::UnionId(u)) => {
                    lang_items.collect_lang_item(db, u);
                }
                ModuleDefId::FunctionId(f) => {
                    lang_items.collect_lang_item(db, f);
                }
                ModuleDefId::StaticId(s) => {
                    lang_items.collect_lang_item(db, s);
                }
                ModuleDefId::TypeAliasId(t) => {
                    lang_items.collect_lang_item(db, t);
                }
                _ => {}
            }
        }
    }

    if matches!(krate.data(db).origin, base_db::CrateOrigin::Lang(base_db::LangCrateOrigin::Core)) {
        lang_items.fill_non_lang_core_items(db, crate_def_map);
    }

    if lang_items.is_empty() { None } else { Some(Box::new(lang_items)) }
}

/// Salsa query. Look for a lang items, starting from the specified crate and recursively
/// traversing its dependencies.
#[salsa_macros::tracked(returns(ref))]
pub fn lang_items(db: &dyn DefDatabase, start_crate: Crate) -> LangItems {
    let _p = tracing::info_span!("lang_items_query").entered();

    let mut result = crate_lang_items(db, start_crate).cloned().unwrap_or_default();

    // Our `CrateGraph` eagerly inserts sysroot dependencies like `core` or `std` into dependencies
    // even if the target crate has `#![no_std]`, `#![no_core]` or shadowed sysroot dependencies
    // like `dependencies.std.path = ".."`. So we use `extern_prelude()` instead of
    // `CrateData.dependencies` here, which has already come through such sysroot complexities
    // while nameres.
    //
    // See https://github.com/rust-lang/rust-analyzer/pull/20475 for details.
    for (_, (module, _)) in crate_local_def_map(db, start_crate).local(db).extern_prelude() {
        // Some crates declares themselves as extern crate like `extern crate self as core`.
        // Ignore these to prevent cycles.
        let krate = module.krate(db);
        if krate != start_crate {
            result.merge_prefer_self(lang_items(db, krate));
        }
    }

    result
}

impl LangItems {
    fn collect_lang_item<T>(&mut self, db: &dyn DefDatabase, item: T)
    where
        T: Into<AttrDefId> + Into<LangItemTarget> + Copy,
    {
        let _p = tracing::info_span!("collect_lang_item").entered();
        if let Some(lang_item) = AttrFlags::lang_item(db, item.into()) {
            self.assign_lang_item(lang_item, item.into());
        }
    }
}

fn resolve_core_trait(
    db: &dyn DefDatabase,
    core_def_map: &DefMap,
    modules: &[Symbol],
    name: Symbol,
) -> Option<TraitId> {
    let mut current = &core_def_map[core_def_map.root];
    for module in modules {
        let Some((ModuleDefId::ModuleId(cur), _)) =
            current.scope.type_(&Name::new_symbol_root(module.clone()))
        else {
            return None;
        };
        if cur.krate(db) != core_def_map.krate() || cur.block(db) != core_def_map.block_id() {
            return None;
        }
        current = &core_def_map[cur];
    }
    let Some((ModuleDefId::TraitId(trait_), _)) = current.scope.type_(&Name::new_symbol_root(name))
    else {
        return None;
    };
    Some(trait_)
}

fn resolve_core_macro(
    db: &dyn DefDatabase,
    core_def_map: &DefMap,
    modules: &[Symbol],
    name: Symbol,
) -> Option<MacroId> {
    let mut current = &core_def_map[core_def_map.root];
    for module in modules {
        let Some((ModuleDefId::ModuleId(cur), _)) =
            current.scope.type_(&Name::new_symbol_root(module.clone()))
        else {
            return None;
        };
        if cur.krate(db) != core_def_map.krate() || cur.block(db) != core_def_map.block_id() {
            return None;
        }
        current = &core_def_map[cur];
    }
    current.scope.makro(&Name::new_symbol_root(name))
}

#[salsa::tracked(returns(as_deref))]
pub(crate) fn crate_notable_traits(db: &dyn DefDatabase, krate: Crate) -> Option<Box<[TraitId]>> {
    let mut traits = Vec::new();

    let crate_def_map = crate_def_map(db, krate);

    for (_, module_data) in crate_def_map.modules() {
        for def in module_data.scope.declarations() {
            if let ModuleDefId::TraitId(trait_) = def
                && AttrFlags::query(db, trait_.into()).contains(AttrFlags::IS_DOC_NOTABLE_TRAIT)
            {
                traits.push(trait_);
            }
        }
    }

    if traits.is_empty() { None } else { Some(traits.into_iter().collect()) }
}

macro_rules! language_item_table {
    (
        $LangItems:ident =>
        $( $(#[$attr:meta])* $lang_item:ident, $module:ident :: $name:ident, $target:ident; )*

        @non_lang_core_traits:

        $( core::$($non_lang_trait_module:ident)::*, $non_lang_trait:ident; )*

        @non_lang_core_macros:

        $( core::$($non_lang_macro_module:ident)::*, $non_lang_macro:ident, $non_lang_macro_field:ident; )*
    ) => {
        #[allow(non_snake_case)] // FIXME: Should we remove this?
        #[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
        pub struct $LangItems {
            $(
                $(#[$attr])*
                pub $lang_item: Option<$target>,
            )*
            $(
                pub $non_lang_trait: Option<TraitId>,
            )*
            $(
                pub $non_lang_macro_field: Option<MacroId>,
            )*
        }

        impl LangItems {
            fn is_empty(&self) -> bool {
                $( self.$lang_item.is_none() )&&*
            }

            /// Merges `self` with `other`, with preference to `self` items.
            fn merge_prefer_self(&mut self, other: &Self) {
                $( self.$lang_item = self.$lang_item.or(other.$lang_item); )*
                $( self.$non_lang_trait = self.$non_lang_trait.or(other.$non_lang_trait); )*
                $( self.$non_lang_macro_field = self.$non_lang_macro_field.or(other.$non_lang_macro_field); )*
            }

            fn assign_lang_item(&mut self, name: Symbol, target: LangItemTarget) {
                match name {
                    $(
                        _ if name == $module::$name => {
                            if let LangItemTarget::$target(target) = target {
                                self.$lang_item = Some(target);
                            }
                        }
                    )*
                    _ => {}
                }
            }

            fn fill_non_lang_core_items(&mut self, db: &dyn DefDatabase, core_def_map: &DefMap) {
                $( self.$non_lang_trait = resolve_core_trait(db, core_def_map, &[ $(sym::$non_lang_trait_module),* ], sym::$non_lang_trait); )*
                $( self.$non_lang_macro_field = resolve_core_macro(db, core_def_map, &[ $(sym::$non_lang_macro_module),* ], sym::$non_lang_macro); )*
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum LangItemEnum {
            $(
                $(#[$attr])*
                $lang_item,
            )*
        }

        impl LangItemEnum {
            #[inline]
            pub fn from_lang_items(self, lang_items: &LangItems) -> Option<LangItemTarget> {
                match self {
                    $( LangItemEnum::$lang_item => lang_items.$lang_item.map(Into::into), )*
                }
            }

            #[inline]
            pub fn from_symbol(symbol: &Symbol) -> Option<Self> {
                match symbol {
                    $( _ if *symbol == $module::$name => Some(Self::$lang_item), )*
                    _ => None,
                }
            }
        }
    }
}

language_item_table! { LangItems =>
//  Variant name,            Name,                     Target;
    Sized,                   sym::sized,               TraitId;
    MetaSized,               sym::meta_sized,          TraitId;
    PointeeSized,            sym::pointee_sized,       TraitId;
    Unsize,                  sym::unsize,              TraitId;
    /// Trait injected by `#[derive(PartialEq)]`, (i.e. "Partial EQ").
    StructuralPeq,           sym::structural_peq,      TraitId;
    /// Trait injected by `#[derive(Eq)]`, (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           sym::structural_teq,      TraitId;
    Copy,                    sym::copy,                TraitId;
    Clone,                   sym::clone,               TraitId;
    TrivialClone,            sym::trivial_clone,       TraitId;
    Sync,                    sym::sync,                TraitId;
    DiscriminantKind,        sym::discriminant_kind,   TraitId;
    /// The associated item of the `DiscriminantKind` trait.
    Discriminant,            sym::discriminant_type,   TypeAliasId;

    PointeeTrait,            sym::pointee_trait,       TraitId;
    Metadata,                sym::metadata_type,       TypeAliasId;
    DynMetadata,             sym::dyn_metadata,        StructId;

    Freeze,                  sym::freeze,              TraitId;

    FnPtrTrait,              sym::fn_ptr_trait,        TraitId;
    FnPtrAddr,               sym::fn_ptr_addr,         FunctionId;

    Drop,                    sym::drop,                TraitId;
    Destruct,                sym::destruct,            TraitId;

    CoerceUnsized,           sym::coerce_unsized,      TraitId;
    DispatchFromDyn,         sym::dispatch_from_dyn,   TraitId;

    // language items relating to transmutability
    TransmuteOpts,           sym::transmute_opts,      StructId;
    TransmuteTrait,          sym::transmute_trait,     TraitId;

    Add,                     sym::add,                 TraitId;
    Sub,                     sym::sub,                 TraitId;
    Mul,                     sym::mul,                 TraitId;
    Div,                     sym::div,                 TraitId;
    Rem,                     sym::rem,                 TraitId;
    Neg,                     sym::neg,                 TraitId;
    Not,                     sym::not,                 TraitId;
    BitXor,                  sym::bitxor,              TraitId;
    BitAnd,                  sym::bitand,              TraitId;
    BitOr,                   sym::bitor,               TraitId;
    Shl,                     sym::shl,                 TraitId;
    Shr,                     sym::shr,                 TraitId;
    AddAssign,               sym::add_assign,          TraitId;
    SubAssign,               sym::sub_assign,          TraitId;
    MulAssign,               sym::mul_assign,          TraitId;
    DivAssign,               sym::div_assign,          TraitId;
    RemAssign,               sym::rem_assign,          TraitId;
    BitXorAssign,            sym::bitxor_assign,       TraitId;
    BitAndAssign,            sym::bitand_assign,       TraitId;
    BitOrAssign,             sym::bitor_assign,        TraitId;
    ShlAssign,               sym::shl_assign,          TraitId;
    ShrAssign,               sym::shr_assign,          TraitId;
    Index,                   sym::index,               TraitId;
    IndexMut,                sym::index_mut,           TraitId;

    UnsafeCell,              sym::unsafe_cell,         StructId;
    UnsafePinned,            sym::unsafe_pinned,       StructId;
    VaList,                  sym::va_list,             StructId;

    Deref,                   sym::deref,               TraitId;
    DerefMut,                sym::deref_mut,           TraitId;
    DerefTarget,             sym::deref_target,        TypeAliasId;
    Receiver,                sym::receiver,            TraitId;
    ReceiverTarget,           sym::receiver_target,    TypeAliasId;

    Fn,                      sym::fn_,                 TraitId;
    FnMut,                   sym::fn_mut,              TraitId;
    FnOnce,                  sym::fn_once,             TraitId;
    AsyncFn,                 sym::async_fn,            TraitId;
    AsyncFnMut,              sym::async_fn_mut,        TraitId;
    AsyncFnOnce,             sym::async_fn_once,       TraitId;

    CallRefFuture,           sym::call_ref_future,     TypeAliasId;
    CallOnceFuture,          sym::call_once_future,    TypeAliasId;
    AsyncFnOnceOutput,       sym::async_fn_once_output, TypeAliasId;

    FnOnceOutput,            sym::fn_once_output,      TypeAliasId;

    Future,                  sym::future_trait,        TraitId;
    CoroutineState,          sym::coroutine_state,     EnumId;
    Coroutine,               sym::coroutine,           TraitId;
    CoroutineReturn,         sym::coroutine_return,    TypeAliasId;
    CoroutineYield,          sym::coroutine_yield,     TypeAliasId;
    Unpin,                   sym::unpin,               TraitId;
    Pin,                     sym::pin,                 StructId;

    PartialEq,               sym::eq,                  TraitId;
    PartialOrd,              sym::partial_ord,         TraitId;
    CVoid,                   sym::c_void,              EnumId;

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               FunctionId;
    PanicNounwind,           sym::panic_nounwind,      FunctionId;
    PanicFmt,                sym::panic_fmt,           FunctionId;
    PanicDisplay,            sym::panic_display,       FunctionId;
    ConstPanicFmt,           sym::const_panic_fmt,     FunctionId;
    PanicBoundsCheck,        sym::panic_bounds_check,  FunctionId;
    PanicMisalignedPointerDereference, sym::panic_misaligned_pointer_dereference, FunctionId;
    PanicInfo,               sym::panic_info,          StructId;
    PanicLocation,           sym::panic_location,      StructId;
    PanicImpl,               sym::panic_impl,          FunctionId;
    PanicCannotUnwind,       sym::panic_cannot_unwind, FunctionId;
    PanicNullPointerDereference, sym::panic_null_pointer_dereference, FunctionId;
    /// libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         FunctionId;

    // Lang items needed for `format_args!()`.
    FormatAlignment,         sym::format_alignment,    EnumId;
    FormatArgument,          sym::format_argument,     StructId;
    FormatArguments,         sym::format_arguments,    StructId;
    FormatCount,             sym::format_count,        EnumId;
    FormatPlaceholder,       sym::format_placeholder,  StructId;
    FormatUnsafeArg,         sym::format_unsafe_arg,   StructId;

    ExchangeMalloc,          sym::exchange_malloc,     FunctionId;
    BoxFree,                 sym::box_free,            FunctionId;
    DropInPlace,             sym::drop_in_place,       FunctionId;
    AllocLayout,             sym::alloc_layout,        StructId;

    Start,                   sym::start,               FunctionId;

    EhPersonality,           sym::eh_personality,      FunctionId;
    EhCatchTypeinfo,         sym::eh_catch_typeinfo,   StaticId;

    OwnedBox,                sym::owned_box,           StructId;

    PhantomData,             sym::phantom_data,        StructId;

    ManuallyDrop,            sym::manually_drop,       StructId;

    MaybeUninit,             sym::maybe_uninit,        UnionId;

    /// Align offset for stride != 1; must not panic.
    AlignOffset,             sym::align_offset,        FunctionId;

    Termination,             sym::termination,         TraitId;

    Try,                     sym::Try,                 TraitId;

    Tuple,                   sym::tuple_trait,         TraitId;

    SliceLen,                sym::slice_len_fn,        FunctionId;

    // Language items from AST lowering
    TryTraitFromResidual,    sym::from_residual,       FunctionId;
    TryTraitFromOutput,      sym::from_output,         FunctionId;
    TryTraitBranch,          sym::branch,              FunctionId;
    TryTraitFromYeet,        sym::from_yeet,           FunctionId;

    PointerLike,             sym::pointer_like,        TraitId;

    ConstParamTy,            sym::const_param_ty,      TraitId;

    Poll,                    sym::Poll,                EnumId;
    PollReady,               sym::Ready,               EnumVariantId;
    PollPending,             sym::Pending,             EnumVariantId;

    // FIXME(swatinem): the following lang items are used for async lowering and
    // should become obsolete eventually.
    ResumeTy,                sym::ResumeTy,            StructId;
    GetContext,              sym::get_context,         FunctionId;

    Context,                 sym::Context,             StructId;
    FuturePoll,              sym::poll,                FunctionId;
    FutureOutput,            sym::future_output,       TypeAliasId;

    Option,                  sym::Option,              EnumId;
    OptionSome,              sym::Some,                EnumVariantId;
    OptionNone,              sym::None,                EnumVariantId;

    ResultOk,                sym::Ok,                  EnumVariantId;
    ResultErr,               sym::Err,                 EnumVariantId;

    ControlFlowContinue,     sym::Continue,            EnumVariantId;
    ControlFlowBreak,        sym::Break,               EnumVariantId;

    IntoFutureIntoFuture,    sym::into_future,         FunctionId;
    IntoIterIntoIter,        sym::into_iter,           FunctionId;
    IteratorNext,            sym::next,                FunctionId;
    Iterator,                sym::iterator,            TraitId;

    PinNewUnchecked,         sym::new_unchecked,       FunctionId;

    RangeFrom,               sym::RangeFrom,           StructId;
    RangeFull,               sym::RangeFull,           StructId;
    RangeInclusiveStruct,    sym::RangeInclusive,      StructId;
    RangeInclusiveNew,       sym::range_inclusive_new, FunctionId;
    Range,                   sym::Range,               StructId;
    RangeToInclusive,        sym::RangeToInclusive,    StructId;
    RangeTo,                 sym::RangeTo,             StructId;

    RangeFromCopy,           sym::RangeFromCopy,           StructId;
    RangeInclusiveCopy,      sym::RangeInclusiveCopy,      StructId;
    RangeCopy,               sym::RangeCopy,               StructId;
    RangeToInclusiveCopy,    sym::RangeToInclusiveCopy,    StructId;

    String,                  sym::String,              StructId;
    CStr,                    sym::CStr,                StructId;
    Ordering,                sym::Ordering,            EnumId;

    @non_lang_core_traits:
    core::default, Default;
    core::fmt, Debug;
    core::hash, Hash;
    core::cmp, Ord;
    core::cmp, Eq;

    @non_lang_core_macros:
    core::default, Default, DefaultDerive;
    core::fmt, Debug, DebugDerive;
    core::hash, Hash, HashDerive;
    core::cmp, PartialOrd, PartialOrdDerive;
    core::cmp, Ord, OrdDerive;
    core::cmp, PartialEq, PartialEqDerive;
    core::cmp, Eq, EqDerive;
    core::marker, CoercePointee, CoercePointeeDerive;
    core::marker, Copy, CopyDerive;
    core::clone, Clone, CloneDerive;
}
