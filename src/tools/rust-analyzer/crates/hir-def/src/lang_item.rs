//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use rustc_hash::FxHashMap;
use syntax::SmolStr;
use triomphe::Arc;

use crate::{
    db::DefDatabase, path::Path, AdtId, AssocItemId, AttrDefId, CrateId, EnumId, EnumVariantId,
    FunctionId, ImplId, ModuleDefId, StaticId, StructId, TraitId, TypeAliasId, UnionId,
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
        if let Some(lang_item) = db.lang_attr(item.into()) {
            self.items.entry(lang_item).or_insert_with(|| constructor(item));
        }
    }
}

pub(crate) fn lang_attr_query(db: &dyn DefDatabase, item: AttrDefId) -> Option<LangItem> {
    let attrs = db.attrs(item);
    attrs.by_key("lang").string_value().and_then(|it| LangItem::from_str(&it))
}

pub enum GenericRequirement {
    None,
    Minimum(usize),
    Exact(usize),
}

macro_rules! language_item_table {
    (
        $( $(#[$attr:meta])* $variant:ident, $module:ident :: $name:ident, $method:ident, $target:expr, $generics:expr; )*
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
            pub fn from_str(name: &str) -> Option<Self> {
                match name {
                    $( stringify!($name) => Some(LangItem::$variant), )*
                    _ => None,
                }
            }
        }
    }
}

impl LangItem {
    /// Opposite of [`LangItem::name`]
    pub fn from_name(name: &hir_expand::name::Name) -> Option<Self> {
        Self::from_str(name.as_str()?)
    }

    pub fn path(&self, db: &dyn DefDatabase, start_crate: CrateId) -> Option<Path> {
        let t = db.lang_item(start_crate, *self)?;
        Some(Path::LangItem(t))
    }
}

language_item_table! {
//  Variant name,            Name,                     Getter method name,         Target                  Generic requirements;
    Sized,                   sym::sized,               sized_trait,                Target::Trait,          GenericRequirement::Exact(0);
    Unsize,                  sym::unsize,              unsize_trait,               Target::Trait,          GenericRequirement::Minimum(1);
    /// Trait injected by `#[derive(PartialEq)]`, (i.e. "Partial EQ").
    StructuralPeq,           sym::structural_peq,      structural_peq_trait,       Target::Trait,          GenericRequirement::None;
    /// Trait injected by `#[derive(Eq)]`, (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           sym::structural_teq,      structural_teq_trait,       Target::Trait,          GenericRequirement::None;
    Copy,                    sym::copy,                copy_trait,                 Target::Trait,          GenericRequirement::Exact(0);
    Clone,                   sym::clone,               clone_trait,                Target::Trait,          GenericRequirement::None;
    Sync,                    sym::sync,                sync_trait,                 Target::Trait,          GenericRequirement::Exact(0);
    DiscriminantKind,        sym::discriminant_kind,   discriminant_kind_trait,    Target::Trait,          GenericRequirement::None;
    /// The associated item of the [`DiscriminantKind`] trait.
    Discriminant,            sym::discriminant_type,   discriminant_type,          Target::AssocTy,        GenericRequirement::None;

    PointeeTrait,            sym::pointee_trait,       pointee_trait,              Target::Trait,          GenericRequirement::None;
    Metadata,                sym::metadata_type,       metadata_type,              Target::AssocTy,        GenericRequirement::None;
    DynMetadata,             sym::dyn_metadata,        dyn_metadata,               Target::Struct,         GenericRequirement::None;

    Freeze,                  sym::freeze,              freeze_trait,               Target::Trait,          GenericRequirement::Exact(0);

    FnPtrTrait,              sym::fn_ptr_trait,        fn_ptr_trait,               Target::Trait,          GenericRequirement::Exact(0);
    FnPtrAddr,               sym::fn_ptr_addr,         fn_ptr_addr,                Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    Drop,                    sym::drop,                drop_trait,                 Target::Trait,          GenericRequirement::None;
    Destruct,                sym::destruct,            destruct_trait,             Target::Trait,          GenericRequirement::None;

    CoerceUnsized,           sym::coerce_unsized,      coerce_unsized_trait,       Target::Trait,          GenericRequirement::Minimum(1);
    DispatchFromDyn,         sym::dispatch_from_dyn,   dispatch_from_dyn_trait,    Target::Trait,          GenericRequirement::Minimum(1);

    // language items relating to transmutability
    TransmuteOpts,           sym::transmute_opts,      transmute_opts,             Target::Struct,         GenericRequirement::Exact(0);
    TransmuteTrait,          sym::transmute_trait,     transmute_trait,            Target::Trait,          GenericRequirement::Exact(3);

    Add,                     sym::add,                 add_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Sub,                     sym::sub,                 sub_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Mul,                     sym::mul,                 mul_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Div,                     sym::div,                 div_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Rem,                     sym::rem,                 rem_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Neg,                     sym::neg,                 neg_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    Not,                     sym::not,                 not_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    BitXor,                  sym::bitxor,              bitxor_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitAnd,                  sym::bitand,              bitand_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitOr,                   sym::bitor,               bitor_trait,                Target::Trait,          GenericRequirement::Exact(1);
    Shl,                     sym::shl,                 shl_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Shr,                     sym::shr,                 shr_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    AddAssign,               sym::add_assign,          add_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    SubAssign,               sym::sub_assign,          sub_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    MulAssign,               sym::mul_assign,          mul_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    DivAssign,               sym::div_assign,          div_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    RemAssign,               sym::rem_assign,          rem_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    BitXorAssign,            sym::bitxor_assign,       bitxor_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitAndAssign,            sym::bitand_assign,       bitand_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitOrAssign,             sym::bitor_assign,        bitor_assign_trait,         Target::Trait,          GenericRequirement::Exact(1);
    ShlAssign,               sym::shl_assign,          shl_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    ShrAssign,               sym::shr_assign,          shr_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    Index,                   sym::index,               index_trait,                Target::Trait,          GenericRequirement::Exact(1);
    IndexMut,                sym::index_mut,           index_mut_trait,            Target::Trait,          GenericRequirement::Exact(1);

    UnsafeCell,              sym::unsafe_cell,         unsafe_cell_type,           Target::Struct,         GenericRequirement::None;
    VaList,                  sym::va_list,             va_list,                    Target::Struct,         GenericRequirement::None;

    Deref,                   sym::deref,               deref_trait,                Target::Trait,          GenericRequirement::Exact(0);
    DerefMut,                sym::deref_mut,           deref_mut_trait,            Target::Trait,          GenericRequirement::Exact(0);
    DerefTarget,             sym::deref_target,        deref_target,               Target::AssocTy,        GenericRequirement::None;
    Receiver,                sym::receiver,            receiver_trait,             Target::Trait,          GenericRequirement::None;

    Fn,                      kw::fn,                   fn_trait,                   Target::Trait,          GenericRequirement::Exact(1);
    FnMut,                   sym::fn_mut,              fn_mut_trait,               Target::Trait,          GenericRequirement::Exact(1);
    FnOnce,                  sym::fn_once,             fn_once_trait,              Target::Trait,          GenericRequirement::Exact(1);

    FnOnceOutput,            sym::fn_once_output,      fn_once_output,             Target::AssocTy,        GenericRequirement::None;

    Future,                  sym::future_trait,        future_trait,               Target::Trait,          GenericRequirement::Exact(0);
    GeneratorState,          sym::generator_state,     gen_state,                  Target::Enum,           GenericRequirement::None;
    Generator,               sym::generator,           gen_trait,                  Target::Trait,          GenericRequirement::Minimum(1);
    Unpin,                   sym::unpin,               unpin_trait,                Target::Trait,          GenericRequirement::None;
    Pin,                     sym::pin,                 pin_type,                   Target::Struct,         GenericRequirement::None;

    PartialEq,               sym::eq,                  eq_trait,                   Target::Trait,          GenericRequirement::Exact(1);
    PartialOrd,              sym::partial_ord,         partial_ord_trait,          Target::Trait,          GenericRequirement::Exact(1);
    CVoid,                   sym::c_void,              c_void,                     Target::Enum,           GenericRequirement::None;

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               panic_fn,                   Target::Fn,             GenericRequirement::Exact(0);
    PanicNounwind,           sym::panic_nounwind,      panic_nounwind,             Target::Fn,             GenericRequirement::Exact(0);
    PanicFmt,                sym::panic_fmt,           panic_fmt,                  Target::Fn,             GenericRequirement::None;
    PanicDisplay,            sym::panic_display,       panic_display,              Target::Fn,             GenericRequirement::None;
    ConstPanicFmt,           sym::const_panic_fmt,     const_panic_fmt,            Target::Fn,             GenericRequirement::None;
    PanicBoundsCheck,        sym::panic_bounds_check,  panic_bounds_check_fn,      Target::Fn,             GenericRequirement::Exact(0);
    PanicMisalignedPointerDereference,        sym::panic_misaligned_pointer_dereference,  panic_misaligned_pointer_dereference_fn,      Target::Fn,             GenericRequirement::Exact(0);
    PanicInfo,               sym::panic_info,          panic_info,                 Target::Struct,         GenericRequirement::None;
    PanicLocation,           sym::panic_location,      panic_location,             Target::Struct,         GenericRequirement::None;
    PanicImpl,               sym::panic_impl,          panic_impl,                 Target::Fn,             GenericRequirement::None;
    PanicCannotUnwind,       sym::panic_cannot_unwind, panic_cannot_unwind,        Target::Fn,             GenericRequirement::Exact(0);
    /// libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         begin_panic_fn,             Target::Fn,             GenericRequirement::None;

    // Lang items needed for `format_args!()`.
    FormatAlignment,         sym::format_alignment,    format_alignment,           Target::Enum,           GenericRequirement::None;
    FormatArgument,          sym::format_argument,     format_argument,            Target::Struct,         GenericRequirement::None;
    FormatArguments,         sym::format_arguments,    format_arguments,           Target::Struct,         GenericRequirement::None;
    FormatCount,             sym::format_count,        format_count,               Target::Enum,           GenericRequirement::None;
    FormatPlaceholder,       sym::format_placeholder,  format_placeholder,         Target::Struct,         GenericRequirement::None;
    FormatUnsafeArg,         sym::format_unsafe_arg,   format_unsafe_arg,          Target::Struct,         GenericRequirement::None;

    ExchangeMalloc,          sym::exchange_malloc,     exchange_malloc_fn,         Target::Fn,             GenericRequirement::None;
    BoxFree,                 sym::box_free,            box_free_fn,                Target::Fn,             GenericRequirement::Minimum(1);
    DropInPlace,             sym::drop_in_place,       drop_in_place_fn,           Target::Fn,             GenericRequirement::Minimum(1);
    AllocLayout,             sym::alloc_layout,        alloc_layout,               Target::Struct,         GenericRequirement::None;

    Start,                   sym::start,               start_fn,                   Target::Fn,             GenericRequirement::Exact(1);

    EhPersonality,           sym::eh_personality,      eh_personality,             Target::Fn,             GenericRequirement::None;
    EhCatchTypeinfo,         sym::eh_catch_typeinfo,   eh_catch_typeinfo,          Target::Static,         GenericRequirement::None;

    OwnedBox,                sym::owned_box,           owned_box,                  Target::Struct,         GenericRequirement::Minimum(1);

    PhantomData,             sym::phantom_data,        phantom_data,               Target::Struct,         GenericRequirement::Exact(1);

    ManuallyDrop,            sym::manually_drop,       manually_drop,              Target::Struct,         GenericRequirement::None;

    MaybeUninit,             sym::maybe_uninit,        maybe_uninit,               Target::Union,          GenericRequirement::None;

    /// Align offset for stride != 1; must not panic.
    AlignOffset,             sym::align_offset,        align_offset_fn,            Target::Fn,             GenericRequirement::None;

    Termination,             sym::termination,         termination,                Target::Trait,          GenericRequirement::None;

    Try,                     sym::Try,                 try_trait,                  Target::Trait,          GenericRequirement::None;

    Tuple,                   sym::tuple_trait,         tuple_trait,                Target::Trait,          GenericRequirement::Exact(0);

    SliceLen,                sym::slice_len_fn,        slice_len_fn,               Target::Method(MethodKind::Inherent), GenericRequirement::None;

    // Language items from AST lowering
    TryTraitFromResidual,    sym::from_residual,       from_residual_fn,           Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitFromOutput,      sym::from_output,         from_output_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitBranch,          sym::branch,              branch_fn,                  Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitFromYeet,        sym::from_yeet,           from_yeet_fn,               Target::Fn,             GenericRequirement::None;

    PointerLike,             sym::pointer_like,        pointer_like,               Target::Trait,          GenericRequirement::Exact(0);

    ConstParamTy,            sym::const_param_ty,      const_param_ty_trait,       Target::Trait,          GenericRequirement::Exact(0);

    Poll,                    sym::Poll,                poll,                       Target::Enum,           GenericRequirement::None;
    PollReady,               sym::Ready,               poll_ready_variant,         Target::Variant,        GenericRequirement::None;
    PollPending,             sym::Pending,             poll_pending_variant,       Target::Variant,        GenericRequirement::None;

    // FIXME(swatinem): the following lang items are used for async lowering and
    // should become obsolete eventually.
    ResumeTy,                sym::ResumeTy,            resume_ty,                  Target::Struct,         GenericRequirement::None;
    GetContext,              sym::get_context,         get_context_fn,             Target::Fn,             GenericRequirement::None;

    Context,                 sym::Context,             context,                    Target::Struct,         GenericRequirement::None;
    FuturePoll,              sym::poll,                future_poll_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    Option,                  sym::Option,              option_type,                Target::Enum,           GenericRequirement::None;
    OptionSome,              sym::Some,                option_some_variant,        Target::Variant,        GenericRequirement::None;
    OptionNone,              sym::None,                option_none_variant,        Target::Variant,        GenericRequirement::None;

    ResultOk,                sym::Ok,                  result_ok_variant,          Target::Variant,        GenericRequirement::None;
    ResultErr,               sym::Err,                 result_err_variant,         Target::Variant,        GenericRequirement::None;

    ControlFlowContinue,     sym::Continue,            cf_continue_variant,        Target::Variant,        GenericRequirement::None;
    ControlFlowBreak,        sym::Break,               cf_break_variant,           Target::Variant,        GenericRequirement::None;

    IntoFutureIntoFuture,    sym::into_future,         into_future_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    IntoIterIntoIter,        sym::into_iter,           into_iter_fn,               Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    IteratorNext,            sym::next,                next_fn,                    Target::Method(MethodKind::Trait { body: false}), GenericRequirement::None;

    PinNewUnchecked,         sym::new_unchecked,       new_unchecked_fn,           Target::Method(MethodKind::Inherent), GenericRequirement::None;

    RangeFrom,               sym::RangeFrom,           range_from_struct,          Target::Struct,         GenericRequirement::None;
    RangeFull,               sym::RangeFull,           range_full_struct,          Target::Struct,         GenericRequirement::None;
    RangeInclusiveStruct,    sym::RangeInclusive,      range_inclusive_struct,     Target::Struct,         GenericRequirement::None;
    RangeInclusiveNew,       sym::range_inclusive_new, range_inclusive_new_method, Target::Method(MethodKind::Inherent), GenericRequirement::None;
    Range,                   sym::Range,               range_struct,               Target::Struct,         GenericRequirement::None;
    RangeToInclusive,        sym::RangeToInclusive,    range_to_inclusive_struct,  Target::Struct,         GenericRequirement::None;
    RangeTo,                 sym::RangeTo,             range_to_struct,            Target::Struct,         GenericRequirement::None;

    String,                  sym::String,              string,                     Target::Struct,         GenericRequirement::None;
    CStr,                    sym::CStr,                c_str,                      Target::Struct,         GenericRequirement::None;
}
