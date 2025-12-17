//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use intern::{Symbol, sym};
use stdx::impl_from;

use crate::{
    AdtId, AssocItemId, AttrDefId, Crate, EnumId, EnumVariantId, FunctionId, ImplId, ModuleDefId,
    StaticId, StructId, TraitId, TypeAliasId, UnionId,
    attrs::AttrFlags,
    db::DefDatabase,
    nameres::{assoc::TraitItems, crate_def_map, crate_local_def_map},
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

    for (_, module_data) in crate_def_map.modules() {
        for impl_def in module_data.scope.impls() {
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

pub enum GenericRequirement {
    None,
    Minimum(usize),
    Exact(usize),
}

macro_rules! language_item_table {
    (
        $LangItems:ident =>
        $( $(#[$attr:meta])* $lang_item:ident, $module:ident :: $name:ident, $method:ident, $target:ident, $generics:expr; )*
    ) => {
        #[allow(non_snake_case)] // FIXME: Should we remove this?
        #[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
        pub struct $LangItems {
            $(
                $(#[$attr])*
                pub $lang_item: Option<$target>,
            )*
        }

        impl LangItems {
            fn is_empty(&self) -> bool {
                $( self.$lang_item.is_none() )&&*
            }

            /// Merges `self` with `other`, with preference to `self` items.
            fn merge_prefer_self(&mut self, other: &Self) {
                $( self.$lang_item = self.$lang_item.or(other.$lang_item); )*
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
//  Variant name,            Name,                     Getter method name,         Target                  Generic requirements;
    Sized,                   sym::sized,               sized_trait,                TraitId,                GenericRequirement::Exact(0);
    MetaSized,               sym::meta_sized,          sized_trait,                TraitId,                GenericRequirement::Exact(0);
    PointeeSized,            sym::pointee_sized,       sized_trait,                TraitId,                GenericRequirement::Exact(0);
    Unsize,                  sym::unsize,              unsize_trait,               TraitId,                GenericRequirement::Minimum(1);
    /// Trait injected by `#[derive(PartialEq)]`, (i.e. "Partial EQ").
    StructuralPeq,           sym::structural_peq,      structural_peq_trait,       TraitId,                GenericRequirement::None;
    /// Trait injected by `#[derive(Eq)]`, (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           sym::structural_teq,      structural_teq_trait,       TraitId,                GenericRequirement::None;
    Copy,                    sym::copy,                copy_trait,                 TraitId,                GenericRequirement::Exact(0);
    Clone,                   sym::clone,               clone_trait,                TraitId,                GenericRequirement::None;
    Sync,                    sym::sync,                sync_trait,                 TraitId,                GenericRequirement::Exact(0);
    DiscriminantKind,        sym::discriminant_kind,   discriminant_kind_trait,    TraitId,                GenericRequirement::None;
    /// The associated item of the `DiscriminantKind` trait.
    Discriminant,            sym::discriminant_type,   discriminant_type,          TypeAliasId,            GenericRequirement::None;

    PointeeTrait,            sym::pointee_trait,       pointee_trait,              TraitId,                GenericRequirement::None;
    Metadata,                sym::metadata_type,       metadata_type,              TypeAliasId,            GenericRequirement::None;
    DynMetadata,             sym::dyn_metadata,        dyn_metadata,               StructId,               GenericRequirement::None;

    Freeze,                  sym::freeze,              freeze_trait,               TraitId,                GenericRequirement::Exact(0);

    FnPtrTrait,              sym::fn_ptr_trait,        fn_ptr_trait,               TraitId,                GenericRequirement::Exact(0);
    FnPtrAddr,               sym::fn_ptr_addr,         fn_ptr_addr,                FunctionId,             GenericRequirement::None;

    Drop,                    sym::drop,                drop_trait,                 TraitId,                GenericRequirement::None;
    Destruct,                sym::destruct,            destruct_trait,             TraitId,                GenericRequirement::None;

    CoerceUnsized,           sym::coerce_unsized,      coerce_unsized_trait,       TraitId,                GenericRequirement::Minimum(1);
    DispatchFromDyn,         sym::dispatch_from_dyn,   dispatch_from_dyn_trait,    TraitId,                GenericRequirement::Minimum(1);

    // language items relating to transmutability
    TransmuteOpts,           sym::transmute_opts,      transmute_opts,             StructId,               GenericRequirement::Exact(0);
    TransmuteTrait,          sym::transmute_trait,     transmute_trait,            TraitId,                GenericRequirement::Exact(3);

    Add,                     sym::add,                 add_trait,                  TraitId,                GenericRequirement::Exact(1);
    Sub,                     sym::sub,                 sub_trait,                  TraitId,                GenericRequirement::Exact(1);
    Mul,                     sym::mul,                 mul_trait,                  TraitId,                GenericRequirement::Exact(1);
    Div,                     sym::div,                 div_trait,                  TraitId,                GenericRequirement::Exact(1);
    Rem,                     sym::rem,                 rem_trait,                  TraitId,                GenericRequirement::Exact(1);
    Neg,                     sym::neg,                 neg_trait,                  TraitId,                GenericRequirement::Exact(0);
    Not,                     sym::not,                 not_trait,                  TraitId,                GenericRequirement::Exact(0);
    BitXor,                  sym::bitxor,              bitxor_trait,               TraitId,                GenericRequirement::Exact(1);
    BitAnd,                  sym::bitand,              bitand_trait,               TraitId,                GenericRequirement::Exact(1);
    BitOr,                   sym::bitor,               bitor_trait,                TraitId,                GenericRequirement::Exact(1);
    Shl,                     sym::shl,                 shl_trait,                  TraitId,                GenericRequirement::Exact(1);
    Shr,                     sym::shr,                 shr_trait,                  TraitId,                GenericRequirement::Exact(1);
    AddAssign,               sym::add_assign,          add_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    SubAssign,               sym::sub_assign,          sub_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    MulAssign,               sym::mul_assign,          mul_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    DivAssign,               sym::div_assign,          div_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    RemAssign,               sym::rem_assign,          rem_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    BitXorAssign,            sym::bitxor_assign,       bitxor_assign_trait,        TraitId,                GenericRequirement::Exact(1);
    BitAndAssign,            sym::bitand_assign,       bitand_assign_trait,        TraitId,                GenericRequirement::Exact(1);
    BitOrAssign,             sym::bitor_assign,        bitor_assign_trait,         TraitId,                GenericRequirement::Exact(1);
    ShlAssign,               sym::shl_assign,          shl_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    ShrAssign,               sym::shr_assign,          shr_assign_trait,           TraitId,                GenericRequirement::Exact(1);
    Index,                   sym::index,               index_trait,                TraitId,                GenericRequirement::Exact(1);
    IndexMut,                sym::index_mut,           index_mut_trait,            TraitId,                GenericRequirement::Exact(1);

    UnsafeCell,              sym::unsafe_cell,         unsafe_cell_type,           StructId,               GenericRequirement::None;
    UnsafePinned,            sym::unsafe_pinned,       unsafe_pinned_type,         StructId,               GenericRequirement::None;
    VaList,                  sym::va_list,             va_list,                    StructId,               GenericRequirement::None;

    Deref,                   sym::deref,               deref_trait,                TraitId,                GenericRequirement::Exact(0);
    DerefMut,                sym::deref_mut,           deref_mut_trait,            TraitId,                GenericRequirement::Exact(0);
    DerefTarget,             sym::deref_target,        deref_target,               TypeAliasId,            GenericRequirement::None;
    Receiver,                sym::receiver,            receiver_trait,             TraitId,                GenericRequirement::None;
    ReceiverTarget,           sym::receiver_target,     receiver_target,           TypeAliasId,            GenericRequirement::None;

    Fn,                      sym::fn_,                 fn_trait,                   TraitId,                GenericRequirement::Exact(1);
    FnMut,                   sym::fn_mut,              fn_mut_trait,               TraitId,                GenericRequirement::Exact(1);
    FnOnce,                  sym::fn_once,             fn_once_trait,              TraitId,                GenericRequirement::Exact(1);
    AsyncFn,                 sym::async_fn,            async_fn_trait,             TraitId,                GenericRequirement::Exact(1);
    AsyncFnMut,              sym::async_fn_mut,        async_fn_mut_trait,         TraitId,                GenericRequirement::Exact(1);
    AsyncFnOnce,             sym::async_fn_once,       async_fn_once_trait,        TraitId,                GenericRequirement::Exact(1);

    CallRefFuture,           sym::call_ref_future,     call_ref_future_ty,         TypeAliasId,            GenericRequirement::None;
    CallOnceFuture,          sym::call_once_future,    call_once_future_ty,        TypeAliasId,            GenericRequirement::None;
    AsyncFnOnceOutput,       sym::async_fn_once_output, async_fn_once_output_ty,   TypeAliasId,            GenericRequirement::None;

    FnOnceOutput,            sym::fn_once_output,      fn_once_output,             TypeAliasId,            GenericRequirement::None;

    Future,                  sym::future_trait,        future_trait,               TraitId,                GenericRequirement::Exact(0);
    CoroutineState,          sym::coroutine_state,     coroutine_state,            EnumId,                 GenericRequirement::None;
    Coroutine,               sym::coroutine,           coroutine_trait,            TraitId,                GenericRequirement::Minimum(1);
    CoroutineReturn,         sym::coroutine_return,    coroutine_return_ty,        TypeAliasId,            GenericRequirement::None;
    CoroutineYield,          sym::coroutine_yield,     coroutine_yield_ty,         TypeAliasId,            GenericRequirement::None;
    Unpin,                   sym::unpin,               unpin_trait,                TraitId,                GenericRequirement::None;
    Pin,                     sym::pin,                 pin_type,                   StructId,               GenericRequirement::None;

    PartialEq,               sym::eq,                  eq_trait,                   TraitId,                GenericRequirement::Exact(1);
    PartialOrd,              sym::partial_ord,         partial_ord_trait,          TraitId,                GenericRequirement::Exact(1);
    CVoid,                   sym::c_void,              c_void,                     EnumId,                 GenericRequirement::None;

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               panic_fn,                   FunctionId,             GenericRequirement::Exact(0);
    PanicNounwind,           sym::panic_nounwind,      panic_nounwind,             FunctionId,             GenericRequirement::Exact(0);
    PanicFmt,                sym::panic_fmt,           panic_fmt,                  FunctionId,             GenericRequirement::None;
    PanicDisplay,            sym::panic_display,       panic_display,              FunctionId,             GenericRequirement::None;
    ConstPanicFmt,           sym::const_panic_fmt,     const_panic_fmt,            FunctionId,             GenericRequirement::None;
    PanicBoundsCheck,        sym::panic_bounds_check,  panic_bounds_check_fn,      FunctionId,             GenericRequirement::Exact(0);
    PanicMisalignedPointerDereference,        sym::panic_misaligned_pointer_dereference,  panic_misaligned_pointer_dereference_fn,      FunctionId,             GenericRequirement::Exact(0);
    PanicInfo,               sym::panic_info,          panic_info,                 StructId,               GenericRequirement::None;
    PanicLocation,           sym::panic_location,      panic_location,             StructId,               GenericRequirement::None;
    PanicImpl,               sym::panic_impl,          panic_impl,                 FunctionId,             GenericRequirement::None;
    PanicCannotUnwind,       sym::panic_cannot_unwind, panic_cannot_unwind,        FunctionId,             GenericRequirement::Exact(0);
    PanicNullPointerDereference, sym::panic_null_pointer_dereference, panic_null_pointer_dereference, FunctionId, GenericRequirement::None;
    /// libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         begin_panic_fn,             FunctionId,             GenericRequirement::None;

    // Lang items needed for `format_args!()`.
    FormatAlignment,         sym::format_alignment,    format_alignment,           EnumId,                 GenericRequirement::None;
    FormatArgument,          sym::format_argument,     format_argument,            StructId,               GenericRequirement::None;
    FormatArguments,         sym::format_arguments,    format_arguments,           StructId,               GenericRequirement::None;
    FormatCount,             sym::format_count,        format_count,               EnumId,                 GenericRequirement::None;
    FormatPlaceholder,       sym::format_placeholder,  format_placeholder,         StructId,               GenericRequirement::None;
    FormatUnsafeArg,         sym::format_unsafe_arg,   format_unsafe_arg,          StructId,               GenericRequirement::None;

    ExchangeMalloc,          sym::exchange_malloc,     exchange_malloc_fn,         FunctionId,             GenericRequirement::None;
    BoxFree,                 sym::box_free,            box_free_fn,                FunctionId,             GenericRequirement::Minimum(1);
    DropInPlace,             sym::drop_in_place,       drop_in_place_fn,           FunctionId,             GenericRequirement::Minimum(1);
    AllocLayout,             sym::alloc_layout,        alloc_layout,               StructId,               GenericRequirement::None;

    Start,                   sym::start,               start_fn,                   FunctionId,             GenericRequirement::Exact(1);

    EhPersonality,           sym::eh_personality,      eh_personality,             FunctionId,             GenericRequirement::None;
    EhCatchTypeinfo,         sym::eh_catch_typeinfo,   eh_catch_typeinfo,          StaticId,               GenericRequirement::None;

    OwnedBox,                sym::owned_box,           owned_box,                  StructId,               GenericRequirement::Minimum(1);

    PhantomData,             sym::phantom_data,        phantom_data,               StructId,               GenericRequirement::Exact(1);

    ManuallyDrop,            sym::manually_drop,       manually_drop,              StructId,               GenericRequirement::None;

    MaybeUninit,             sym::maybe_uninit,        maybe_uninit,               UnionId,                GenericRequirement::None;

    /// Align offset for stride != 1; must not panic.
    AlignOffset,             sym::align_offset,        align_offset_fn,            FunctionId,             GenericRequirement::None;

    Termination,             sym::termination,         termination,                TraitId,                GenericRequirement::None;

    Try,                     sym::Try,                 try_trait,                  TraitId,                GenericRequirement::None;

    Tuple,                   sym::tuple_trait,         tuple_trait,                TraitId,                GenericRequirement::Exact(0);

    SliceLen,                sym::slice_len_fn,        slice_len_fn,               FunctionId,             GenericRequirement::None;

    // Language items from AST lowering
    TryTraitFromResidual,    sym::from_residual,       from_residual_fn,           FunctionId,             GenericRequirement::None;
    TryTraitFromOutput,      sym::from_output,         from_output_fn,             FunctionId,             GenericRequirement::None;
    TryTraitBranch,          sym::branch,              branch_fn,                  FunctionId,             GenericRequirement::None;
    TryTraitFromYeet,        sym::from_yeet,           from_yeet_fn,               FunctionId,             GenericRequirement::None;

    PointerLike,             sym::pointer_like,        pointer_like,               TraitId,                GenericRequirement::Exact(0);

    ConstParamTy,            sym::const_param_ty,      const_param_ty_trait,       TraitId,                GenericRequirement::Exact(0);

    Poll,                    sym::Poll,                poll,                       EnumId,                 GenericRequirement::None;
    PollReady,               sym::Ready,               poll_ready_variant,         EnumVariantId,          GenericRequirement::None;
    PollPending,             sym::Pending,             poll_pending_variant,       EnumVariantId,          GenericRequirement::None;

    // FIXME(swatinem): the following lang items are used for async lowering and
    // should become obsolete eventually.
    ResumeTy,                sym::ResumeTy,            resume_ty,                  StructId,               GenericRequirement::None;
    GetContext,              sym::get_context,         get_context_fn,             FunctionId,             GenericRequirement::None;

    Context,                 sym::Context,             context,                    StructId,               GenericRequirement::None;
    FuturePoll,              sym::poll,                future_poll_fn,             FunctionId,             GenericRequirement::None;
    FutureOutput,            sym::future_output,       future_output,              TypeAliasId,            GenericRequirement::None;

    Option,                  sym::Option,              option_type,                EnumId,                 GenericRequirement::None;
    OptionSome,              sym::Some,                option_some_variant,        EnumVariantId,          GenericRequirement::None;
    OptionNone,              sym::None,                option_none_variant,        EnumVariantId,          GenericRequirement::None;

    ResultOk,                sym::Ok,                  result_ok_variant,          EnumVariantId,          GenericRequirement::None;
    ResultErr,               sym::Err,                 result_err_variant,         EnumVariantId,          GenericRequirement::None;

    ControlFlowContinue,     sym::Continue,            cf_continue_variant,        EnumVariantId,          GenericRequirement::None;
    ControlFlowBreak,        sym::Break,               cf_break_variant,           EnumVariantId,          GenericRequirement::None;

    IntoFutureIntoFuture,    sym::into_future,         into_future_fn,             FunctionId,             GenericRequirement::None;
    IntoIterIntoIter,        sym::into_iter,           into_iter_fn,               FunctionId,             GenericRequirement::None;
    IteratorNext,            sym::next,                next_fn,                    FunctionId,             GenericRequirement::None;
    Iterator,                sym::iterator,            iterator,                   TraitId,                GenericRequirement::None;

    PinNewUnchecked,         sym::new_unchecked,       new_unchecked_fn,           FunctionId,             GenericRequirement::None;

    RangeFrom,               sym::RangeFrom,           range_from_struct,          StructId,               GenericRequirement::None;
    RangeFull,               sym::RangeFull,           range_full_struct,          StructId,               GenericRequirement::None;
    RangeInclusiveStruct,    sym::RangeInclusive,      range_inclusive_struct,     StructId,               GenericRequirement::None;
    RangeInclusiveNew,       sym::range_inclusive_new, range_inclusive_new_method, FunctionId,             GenericRequirement::None;
    Range,                   sym::Range,               range_struct,               StructId,               GenericRequirement::None;
    RangeToInclusive,        sym::RangeToInclusive,    range_to_inclusive_struct,  StructId,               GenericRequirement::None;
    RangeTo,                 sym::RangeTo,             range_to_struct,            StructId,               GenericRequirement::None;

    String,                  sym::String,              string,                     StructId,               GenericRequirement::None;
    CStr,                    sym::CStr,                c_str,                      StructId,               GenericRequirement::None;
    Ordering,                sym::Ordering,            ordering,                   EnumId,                 GenericRequirement::None;
}
