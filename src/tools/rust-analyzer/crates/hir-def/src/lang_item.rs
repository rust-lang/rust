//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use std::sync::Arc;

use rustc_hash::FxHashMap;
use syntax::SmolStr;

use crate::{
    db::DefDatabase, AdtId, AssocItemId, AttrDefId, CrateId, EnumId, EnumVariantId, FunctionId,
    ImplId, ModuleDefId, StaticId, StructId, TraitId, TypeAliasId, UnionId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    EnumId(EnumId),
    Function(FunctionId),
    ImplDef(ImplId),
    Static(StaticId),
    Struct(StructId),
    Union(UnionId),
    TypeAlias(TypeAliasId),
    Trait(TraitId),
    EnumVariant(EnumVariantId),
}

impl LangItemTarget {
    pub fn as_enum(self) -> Option<EnumId> {
        match self {
            LangItemTarget::EnumId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_function(self) -> Option<FunctionId> {
        match self {
            LangItemTarget::Function(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_impl_def(self) -> Option<ImplId> {
        match self {
            LangItemTarget::ImplDef(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_static(self) -> Option<StaticId> {
        match self {
            LangItemTarget::Static(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_struct(self) -> Option<StructId> {
        match self {
            LangItemTarget::Struct(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_trait(self) -> Option<TraitId> {
        match self {
            LangItemTarget::Trait(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_enum_variant(self) -> Option<EnumVariantId> {
        match self {
            LangItemTarget::EnumVariant(id) => Some(id),
            _ => None,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LangItems {
    items: FxHashMap<LangItem, LangItemTarget>,
}

impl LangItems {
    pub fn target(&self, item: LangItem) -> Option<LangItemTarget> {
        self.items.get(&item).copied()
    }

    /// Salsa query. This will look for lang items in a specific crate.
    pub(crate) fn crate_lang_items_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<LangItems> {
        let _p = profile::span("crate_lang_items_query");

        let mut lang_items = LangItems::default();

        let crate_def_map = db.crate_def_map(krate);

        for (_, module_data) in crate_def_map.modules() {
            for impl_def in module_data.scope.impls() {
                lang_items.collect_lang_item(db, impl_def, LangItemTarget::ImplDef);
                for assoc in db.impl_data(impl_def).items.iter().copied() {
                    match assoc {
                        AssocItemId::FunctionId(f) => {
                            lang_items.collect_lang_item(db, f, LangItemTarget::Function)
                        }
                        AssocItemId::TypeAliasId(t) => {
                            lang_items.collect_lang_item(db, t, LangItemTarget::TypeAlias)
                        }
                        AssocItemId::ConstId(_) => (),
                    }
                }
            }

            for def in module_data.scope.declarations() {
                match def {
                    ModuleDefId::TraitId(trait_) => {
                        lang_items.collect_lang_item(db, trait_, LangItemTarget::Trait);
                        db.trait_data(trait_).items.iter().for_each(|&(_, assoc_id)| {
                            if let AssocItemId::FunctionId(f) = assoc_id {
                                lang_items.collect_lang_item(db, f, LangItemTarget::Function);
                            }
                        });
                    }
                    ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                        lang_items.collect_lang_item(db, e, LangItemTarget::EnumId);
                        db.enum_data(e).variants.iter().for_each(|(local_id, _)| {
                            lang_items.collect_lang_item(
                                db,
                                EnumVariantId { parent: e, local_id },
                                LangItemTarget::EnumVariant,
                            );
                        });
                    }
                    ModuleDefId::AdtId(AdtId::StructId(s)) => {
                        lang_items.collect_lang_item(db, s, LangItemTarget::Struct);
                    }
                    ModuleDefId::AdtId(AdtId::UnionId(u)) => {
                        lang_items.collect_lang_item(db, u, LangItemTarget::Union);
                    }
                    ModuleDefId::FunctionId(f) => {
                        lang_items.collect_lang_item(db, f, LangItemTarget::Function);
                    }
                    ModuleDefId::StaticId(s) => {
                        lang_items.collect_lang_item(db, s, LangItemTarget::Static);
                    }
                    ModuleDefId::TypeAliasId(t) => {
                        lang_items.collect_lang_item(db, t, LangItemTarget::TypeAlias);
                    }
                    _ => {}
                }
            }
        }

        Arc::new(lang_items)
    }

    /// Salsa query. Look for a lang item, starting from the specified crate and recursively
    /// traversing its dependencies.
    pub(crate) fn lang_item_query(
        db: &dyn DefDatabase,
        start_crate: CrateId,
        item: LangItem,
    ) -> Option<LangItemTarget> {
        let _p = profile::span("lang_item_query");
        let lang_items = db.crate_lang_items(start_crate);
        let start_crate_target = lang_items.items.get(&item);
        if let Some(&target) = start_crate_target {
            return Some(target);
        }
        db.crate_graph()[start_crate]
            .dependencies
            .iter()
            .find_map(|dep| db.lang_item(dep.crate_id, item))
    }

    fn collect_lang_item<T>(
        &mut self,
        db: &dyn DefDatabase,
        item: T,
        constructor: fn(T) -> LangItemTarget,
    ) where
        T: Into<AttrDefId> + Copy,
    {
        let _p = profile::span("collect_lang_item");
        if let Some(lang_item) = lang_attr(db, item).and_then(|it| LangItem::from_str(&it)) {
            self.items.entry(lang_item).or_insert_with(|| constructor(item));
        }
    }
}

pub fn lang_attr(db: &dyn DefDatabase, item: impl Into<AttrDefId> + Copy) -> Option<SmolStr> {
    let attrs = db.attrs(item.into());
    attrs.by_key("lang").string_value().cloned()
}

pub enum GenericRequirement {
    None,
    Minimum(usize),
    Exact(usize),
}

macro_rules! language_item_table {
    (
        $( $(#[$attr:meta])* $variant:ident, $name:ident, $method:ident, $target:expr, $generics:expr; )*
    ) => {

        /// A representation of all the valid language items in Rust.
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub enum LangItem {
            $(
                #[doc = concat!("The `", stringify!($name), "` lang item.")]
                $(#[$attr])*
                $variant,
            )*
        }

        impl LangItem {
            pub fn name(self) -> SmolStr {
                match self {
                    $( LangItem::$variant => SmolStr::new(stringify!($name)), )*
                }
            }

            /// Opposite of [`LangItem::name`]
            pub fn from_name(name: &hir_expand::name::Name) -> Option<Self> {
                Self::from_str(name.as_str()?)
            }

            /// Opposite of [`LangItem::name`]
            pub fn from_str(name: &str) -> Option<Self> {
                match name {
                    $( stringify!($name) => Some(LangItem::$variant), )*
                    _ => None,
                }
            }
        }
    }
}

language_item_table! {
//  Variant name,            Name,                     Getter method name,         Target                  Generic requirements;
    Sized,                   sized,               sized_trait,                Target::Trait,          GenericRequirement::Exact(0);
    Unsize,                  unsize,              unsize_trait,               Target::Trait,          GenericRequirement::Minimum(1);
    /// Trait injected by `#[derive(PartialEq)]`, (i.e. "Partial EQ").
    StructuralPeq,           structural_peq,      structural_peq_trait,       Target::Trait,          GenericRequirement::None;
    /// Trait injected by `#[derive(Eq)]`, (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           structural_teq,      structural_teq_trait,       Target::Trait,          GenericRequirement::None;
    Copy,                    copy,                copy_trait,                 Target::Trait,          GenericRequirement::Exact(0);
    Clone,                   clone,               clone_trait,                Target::Trait,          GenericRequirement::None;
    Sync,                    sync,                sync_trait,                 Target::Trait,          GenericRequirement::Exact(0);
    DiscriminantKind,        discriminant_kind,   discriminant_kind_trait,    Target::Trait,          GenericRequirement::None;
    /// The associated item of the [`DiscriminantKind`] trait.
    Discriminant,            discriminant_type,   discriminant_type,          Target::AssocTy,        GenericRequirement::None;

    PointeeTrait,            pointee_trait,       pointee_trait,              Target::Trait,          GenericRequirement::None;
    Metadata,                metadata_type,       metadata_type,              Target::AssocTy,        GenericRequirement::None;
    DynMetadata,             dyn_metadata,        dyn_metadata,               Target::Struct,         GenericRequirement::None;

    Freeze,                  freeze,              freeze_trait,               Target::Trait,          GenericRequirement::Exact(0);

    Drop,                    drop,                drop_trait,                 Target::Trait,          GenericRequirement::None;
    Destruct,                destruct,            destruct_trait,             Target::Trait,          GenericRequirement::None;

    CoerceUnsized,           coerce_unsized,      coerce_unsized_trait,       Target::Trait,          GenericRequirement::Minimum(1);
    DispatchFromDyn,         dispatch_from_dyn,   dispatch_from_dyn_trait,    Target::Trait,          GenericRequirement::Minimum(1);

    // language items relating to transmutability
    TransmuteOpts,           transmute_opts,      transmute_opts,             Target::Struct,         GenericRequirement::Exact(0);
    TransmuteTrait,          transmute_trait,     transmute_trait,            Target::Trait,          GenericRequirement::Exact(3);

    Add,                     add,                 add_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Sub,                     sub,                 sub_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Mul,                     mul,                 mul_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Div,                     div,                 div_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Rem,                     rem,                 rem_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Neg,                     neg,                 neg_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    Not,                     not,                 not_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    BitXor,                  bitxor,              bitxor_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitAnd,                  bitand,              bitand_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitOr,                   bitor,               bitor_trait,                Target::Trait,          GenericRequirement::Exact(1);
    Shl,                     shl,                 shl_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Shr,                     shr,                 shr_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    AddAssign,               add_assign,          add_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    SubAssign,               sub_assign,          sub_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    MulAssign,               mul_assign,          mul_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    DivAssign,               div_assign,          div_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    RemAssign,               rem_assign,          rem_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    BitXorAssign,            bitxor_assign,       bitxor_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitAndAssign,            bitand_assign,       bitand_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitOrAssign,             bitor_assign,        bitor_assign_trait,         Target::Trait,          GenericRequirement::Exact(1);
    ShlAssign,               shl_assign,          shl_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    ShrAssign,               shr_assign,          shr_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    Index,                   index,               index_trait,                Target::Trait,          GenericRequirement::Exact(1);
    IndexMut,                index_mut,           index_mut_trait,            Target::Trait,          GenericRequirement::Exact(1);

    UnsafeCell,              unsafe_cell,         unsafe_cell_type,           Target::Struct,         GenericRequirement::None;
    VaList,                  va_list,             va_list,                    Target::Struct,         GenericRequirement::None;

    Deref,                   deref,               deref_trait,                Target::Trait,          GenericRequirement::Exact(0);
    DerefMut,                deref_mut,           deref_mut_trait,            Target::Trait,          GenericRequirement::Exact(0);
    DerefTarget,             deref_target,        deref_target,               Target::AssocTy,        GenericRequirement::None;
    Receiver,                receiver,            receiver_trait,             Target::Trait,          GenericRequirement::None;

    Fn,                      fn,                  fn_trait,                   Target::Trait,          GenericRequirement::Exact(1);
    FnMut,                   fn_mut,              fn_mut_trait,               Target::Trait,          GenericRequirement::Exact(1);
    FnOnce,                  fn_once,             fn_once_trait,              Target::Trait,          GenericRequirement::Exact(1);

    FnOnceOutput,            fn_once_output,      fn_once_output,             Target::AssocTy,        GenericRequirement::None;

    Future,                  future_trait,        future_trait,               Target::Trait,          GenericRequirement::Exact(0);
    GeneratorState,          generator_state,     gen_state,                  Target::Enum,           GenericRequirement::None;
    Generator,               generator,           gen_trait,                  Target::Trait,          GenericRequirement::Minimum(1);
    Unpin,                   unpin,               unpin_trait,                Target::Trait,          GenericRequirement::None;
    Pin,                     pin,                 pin_type,                   Target::Struct,         GenericRequirement::None;

    PartialEq,               eq,                  eq_trait,                   Target::Trait,          GenericRequirement::Exact(1);
    PartialOrd,              partial_ord,         partial_ord_trait,          Target::Trait,          GenericRequirement::Exact(1);

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   panic,               panic_fn,                   Target::Fn,             GenericRequirement::Exact(0);
    PanicNounwind,           panic_nounwind,      panic_nounwind,             Target::Fn,             GenericRequirement::Exact(0);
    PanicFmt,                panic_fmt,           panic_fmt,                  Target::Fn,             GenericRequirement::None;
    PanicDisplay,            panic_display,       panic_display,              Target::Fn,             GenericRequirement::None;
    ConstPanicFmt,           const_panic_fmt,     const_panic_fmt,            Target::Fn,             GenericRequirement::None;
    PanicBoundsCheck,        panic_bounds_check,  panic_bounds_check_fn,      Target::Fn,             GenericRequirement::Exact(0);
    PanicInfo,               panic_info,          panic_info,                 Target::Struct,         GenericRequirement::None;
    PanicLocation,           panic_location,      panic_location,             Target::Struct,         GenericRequirement::None;
    PanicImpl,               panic_impl,          panic_impl,                 Target::Fn,             GenericRequirement::None;
    PanicCannotUnwind,       panic_cannot_unwind, panic_cannot_unwind,        Target::Fn,             GenericRequirement::Exact(0);
    /// libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              begin_panic,         begin_panic_fn,             Target::Fn,             GenericRequirement::None;

    ExchangeMalloc,          exchange_malloc,     exchange_malloc_fn,         Target::Fn,             GenericRequirement::None;
    BoxFree,                 box_free,            box_free_fn,                Target::Fn,             GenericRequirement::Minimum(1);
    DropInPlace,             drop_in_place,       drop_in_place_fn,           Target::Fn,             GenericRequirement::Minimum(1);
    AllocLayout,             alloc_layout,        alloc_layout,               Target::Struct,         GenericRequirement::None;

    Start,                   start,               start_fn,                   Target::Fn,             GenericRequirement::Exact(1);

    EhPersonality,           eh_personality,      eh_personality,             Target::Fn,             GenericRequirement::None;
    EhCatchTypeinfo,         eh_catch_typeinfo,   eh_catch_typeinfo,          Target::Static,         GenericRequirement::None;

    OwnedBox,                owned_box,           owned_box,                  Target::Struct,         GenericRequirement::Minimum(1);

    PhantomData,             phantom_data,        phantom_data,               Target::Struct,         GenericRequirement::Exact(1);

    ManuallyDrop,            manually_drop,       manually_drop,              Target::Struct,         GenericRequirement::None;

    MaybeUninit,             maybe_uninit,        maybe_uninit,               Target::Union,          GenericRequirement::None;

    /// Align offset for stride != 1; must not panic.
    AlignOffset,             align_offset,        align_offset_fn,            Target::Fn,             GenericRequirement::None;

    Termination,             termination,         termination,                Target::Trait,          GenericRequirement::None;

    Try,                     Try,                 try_trait,                  Target::Trait,          GenericRequirement::None;

    Tuple,                   tuple_trait,         tuple_trait,                Target::Trait,          GenericRequirement::Exact(0);

    SliceLen,                slice_len_fn,        slice_len_fn,               Target::Method(MethodKind::Inherent), GenericRequirement::None;

    // Language items from AST lowering
    TryTraitFromResidual,    from_residual,       from_residual_fn,           Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitFromOutput,      from_output,         from_output_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitBranch,          branch,              branch_fn,                  Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitFromYeet,        from_yeet,           from_yeet_fn,               Target::Fn,             GenericRequirement::None;

    PointerSized,            pointer_sized,       pointer_sized,              Target::Trait,          GenericRequirement::Exact(0);

    Poll,                    Poll,                poll,                       Target::Enum,           GenericRequirement::None;
    PollReady,               Ready,               poll_ready_variant,         Target::Variant,        GenericRequirement::None;
    PollPending,             Pending,             poll_pending_variant,       Target::Variant,        GenericRequirement::None;

    // FIXME(swatinem): the following lang items are used for async lowering and
    // should become obsolete eventually.
    ResumeTy,                ResumeTy,            resume_ty,                  Target::Struct,         GenericRequirement::None;
    IdentityFuture,          identity_future,     identity_future_fn,         Target::Fn,             GenericRequirement::None;
    GetContext,              get_context,         get_context_fn,             Target::Fn,             GenericRequirement::None;

    Context,                 Context,             context,                    Target::Struct,         GenericRequirement::None;
    FuturePoll,              poll,                future_poll_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    FromFrom,                from,                from_fn,                    Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    OptionSome,              Some,                option_some_variant,        Target::Variant,        GenericRequirement::None;
    OptionNone,              None,                option_none_variant,        Target::Variant,        GenericRequirement::None;

    ResultOk,                Ok,                  result_ok_variant,          Target::Variant,        GenericRequirement::None;
    ResultErr,               Err,                 result_err_variant,         Target::Variant,        GenericRequirement::None;

    ControlFlowContinue,     Continue,            cf_continue_variant,        Target::Variant,        GenericRequirement::None;
    ControlFlowBreak,        Break,               cf_break_variant,           Target::Variant,        GenericRequirement::None;

    IntoFutureIntoFuture,    into_future,         into_future_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    IntoIterIntoIter,        into_iter,           into_iter_fn,               Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    IteratorNext,            next,                next_fn,                    Target::Method(MethodKind::Trait { body: false}), GenericRequirement::None;

    PinNewUnchecked,         new_unchecked,       new_unchecked_fn,           Target::Method(MethodKind::Inherent), GenericRequirement::None;

    RangeFrom,               RangeFrom,           range_from_struct,          Target::Struct,         GenericRequirement::None;
    RangeFull,               RangeFull,           range_full_struct,          Target::Struct,         GenericRequirement::None;
    RangeInclusiveStruct,    RangeInclusive,      range_inclusive_struct,     Target::Struct,         GenericRequirement::None;
    RangeInclusiveNew,       range_inclusive_new, range_inclusive_new_method, Target::Method(MethodKind::Inherent), GenericRequirement::None;
    Range,                   Range,               range_struct,               Target::Struct,         GenericRequirement::None;
    RangeToInclusive,        RangeToInclusive,    range_to_inclusive_struct,  Target::Struct,         GenericRequirement::None;
    RangeTo,                 RangeTo,             range_to_struct,            Target::Struct,         GenericRequirement::None;

    String,                  String,              string,                     Target::Struct,         GenericRequirement::None;
}
