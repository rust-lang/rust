//! Defines language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

pub use self::LangItem::*;

use crate::def_id::DefId;
use crate::{MethodKind, Target};

use rustc_ast::ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable_Generic;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use lazy_static::lazy_static;

pub enum LangItemGroup {
    Op,
}

const NUM_GROUPS: usize = 1;

macro_rules! expand_group {
    () => {
        None
    };
    ($group:expr) => {
        Some($group)
    };
}

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! language_item_table {
    (
        $( $variant:ident $($group:expr)?, $name:expr, $method:ident, $target:expr; )*
    ) => {

        enum_from_u32! {
            /// A representation of all the valid language items in Rust.
            #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable)]
            pub enum LangItem {
                $($variant,)*
            }
        }

        impl LangItem {
            /// Returns the `name` symbol in `#[lang = "$name"]`.
            /// For example, `LangItem::EqTraitLangItem`,
            /// that is `#[lang = "eq"]` would result in `sym::eq`.
            pub fn name(self) -> Symbol {
                match self {
                    $( $variant => $name, )*
                }
            }

            pub fn group(self) -> Option<LangItemGroup> {
                use LangItemGroup::*;
                match self {
                    $( $variant => expand_group!($($group)*), )*
                }
            }
        }

        #[derive(HashStable_Generic)]
        pub struct LanguageItems {
            /// Mappings from lang items to their possibly found `DefId`s.
            /// The index corresponds to the order in `LangItem`.
            pub items: Vec<Option<DefId>>,
            /// Lang items that were not found during collection.
            pub missing: Vec<LangItem>,
            /// Mapping from `LangItemGroup` discriminants to all
            /// `DefId`s of lang items in that group.
            pub groups: [Vec<DefId>; NUM_GROUPS],
        }

        impl LanguageItems {
            /// Construct an empty collection of lang items and no missing ones.
            pub fn new() -> Self {
                fn init_none(_: LangItem) -> Option<DefId> { None }

                Self {
                    items: vec![$(init_none($variant)),*],
                    missing: Vec::new(),
                    groups: [vec![]; NUM_GROUPS],
                }
            }

            /// Returns the mappings to the possibly found `DefId`s for each lang item.
            pub fn items(&self) -> &[Option<DefId>] {
                &*self.items
            }

            /// Requires that a given `LangItem` was bound and returns the corresponding `DefId`.
            /// If it wasn't bound, e.g. due to a missing `#[lang = "<it.name()>"]`,
            /// returns an error message as a string.
            pub fn require(&self, it: LangItem) -> Result<DefId, String> {
                self.items[it as usize].ok_or_else(|| format!("requires `{}` lang_item", it.name()))
            }

            pub fn group(&self, group: LangItemGroup) -> &[DefId] {
                self.groups[group as usize].as_ref()
            }

            $(
                /// Returns the corresponding `DefId` for the lang item if it
                /// exists.
                #[allow(dead_code)]
                pub fn $method(&self) -> Option<DefId> {
                    self.items[$variant as usize]
                }
            )*
        }

        lazy_static! {
            /// A mapping from the name of the lang item to its order and the form it must be of.
            pub static ref ITEM_REFS: FxHashMap<Symbol, (usize, Target)> = {
                let mut item_refs = FxHashMap::default();
                $( item_refs.insert($name, ($variant as usize, $target)); )*
                item_refs
            };
        }

// End of the macro
    }
}

impl<CTX> HashStable<CTX> for LangItem {
    fn hash_stable(&self, _: &mut CTX, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

/// Extracts the first `lang = "$name"` out of a list of attributes.
/// The attributes `#[panic_handler]` and `#[alloc_error_handler]`
/// are also extracted out when found.
///
/// About the `check_name` argument: passing in a `Session` would be simpler,
/// because then we could call `Session::check_name` directly. But we want to
/// avoid the need for `librustc_hir` to depend on `librustc_session`, so we
/// use a closure instead.
pub fn extract<'a, F>(check_name: F, attrs: &'a [ast::Attribute]) -> Option<(Symbol, Span)>
where
    F: Fn(&'a ast::Attribute, Symbol) -> bool,
{
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if check_name(attr, sym::lang) => (attr.value_str()?, attr.span),
            _ if check_name(attr, sym::panic_handler) => (sym::panic_impl, attr.span),
            _ if check_name(attr, sym::alloc_error_handler) => (sym::oom, attr.span),
            _ => return None,
        })
    })
}

language_item_table! {
//  Variant name,                  Name,                    Method name,             Target;
    BoolImplItem,                  sym::bool,               bool_impl,               Target::Impl;
    CharImplItem,                  sym::char,               char_impl,               Target::Impl;
    StrImplItem,                   sym::str,                str_impl,                Target::Impl;
    ArrayImplItem,                 sym::array,              array_impl,              Target::Impl;
    SliceImplItem,                 sym::slice,              slice_impl,              Target::Impl;
    SliceU8ImplItem,               sym::slice_u8,           slice_u8_impl,           Target::Impl;
    StrAllocImplItem,              sym::str_alloc,          str_alloc_impl,          Target::Impl;
    SliceAllocImplItem,            sym::slice_alloc,        slice_alloc_impl,        Target::Impl;
    SliceU8AllocImplItem,          sym::slice_u8_alloc,     slice_u8_alloc_impl,     Target::Impl;
    ConstPtrImplItem,              sym::const_ptr,          const_ptr_impl,          Target::Impl;
    MutPtrImplItem,                sym::mut_ptr,            mut_ptr_impl,            Target::Impl;
    ConstSlicePtrImplItem,         sym::const_slice_ptr,    const_slice_ptr_impl,    Target::Impl;
    MutSlicePtrImplItem,           sym::mut_slice_ptr,      mut_slice_ptr_impl,      Target::Impl;
    I8ImplItem,                    sym::i8,                 i8_impl,                 Target::Impl;
    I16ImplItem,                   sym::i16,                i16_impl,                Target::Impl;
    I32ImplItem,                   sym::i32,                i32_impl,                Target::Impl;
    I64ImplItem,                   sym::i64,                i64_impl,                Target::Impl;
    I128ImplItem,                  sym::i128,               i128_impl,               Target::Impl;
    IsizeImplItem,                 sym::isize,              isize_impl,              Target::Impl;
    U8ImplItem,                    sym::u8,                 u8_impl,                 Target::Impl;
    U16ImplItem,                   sym::u16,                u16_impl,                Target::Impl;
    U32ImplItem,                   sym::u32,                u32_impl,                Target::Impl;
    U64ImplItem,                   sym::u64,                u64_impl,                Target::Impl;
    U128ImplItem,                  sym::u128,               u128_impl,               Target::Impl;
    UsizeImplItem,                 sym::usize,              usize_impl,              Target::Impl;
    F32ImplItem,                   sym::f32,                f32_impl,                Target::Impl;
    F64ImplItem,                   sym::f64,                f64_impl,                Target::Impl;
    F32RuntimeImplItem,            sym::f32_runtime,        f32_runtime_impl,        Target::Impl;
    F64RuntimeImplItem,            sym::f64_runtime,        f64_runtime_impl,        Target::Impl;

    SizedTraitLangItem,            sym::sized,              sized_trait,             Target::Trait;
    UnsizeTraitLangItem,           sym::unsize,             unsize_trait,            Target::Trait;
    // trait injected by #[derive(PartialEq)], (i.e. "Partial EQ").
    StructuralPeqTraitLangItem,    sym::structural_peq,     structural_peq_trait,    Target::Trait;
    // trait injected by #[derive(Eq)], (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeqTraitLangItem,    sym::structural_teq,     structural_teq_trait,    Target::Trait;
    CopyTraitLangItem,             sym::copy,               copy_trait,              Target::Trait;
    CloneTraitLangItem,            sym::clone,              clone_trait,             Target::Trait;
    SyncTraitLangItem,             sym::sync,               sync_trait,              Target::Trait;
    DiscriminantKindTraitLangItem, sym::discriminant_kind,  discriminant_kind_trait, Target::Trait;
    // The associated item of `trait DiscriminantKind`.
    DiscriminantTypeLangItem,      sym::discriminant_type,  discriminant_type,       Target::AssocTy;

    FreezeTraitLangItem,           sym::freeze,             freeze_trait,            Target::Trait;

    DropTraitLangItem,             sym::drop,               drop_trait,              Target::Trait;

    CoerceUnsizedTraitLangItem,    sym::coerce_unsized,     coerce_unsized_trait,    Target::Trait;
    DispatchFromDynTraitLangItem,  sym::dispatch_from_dyn,  dispatch_from_dyn_trait, Target::Trait;

    AddTraitLangItem(Op),          sym::add,                add_trait,               Target::Trait;
    SubTraitLangItem(Op),          sym::sub,                sub_trait,               Target::Trait;
    MulTraitLangItem(Op),          sym::mul,                mul_trait,               Target::Trait;
    DivTraitLangItem(Op),          sym::div,                div_trait,               Target::Trait;
    RemTraitLangItem(Op),          sym::rem,                rem_trait,               Target::Trait;
    NegTraitLangItem(Op),          sym::neg,                neg_trait,               Target::Trait;
    NotTraitLangItem(Op),          sym::not,                not_trait,               Target::Trait;
    BitXorTraitLangItem(Op),       sym::bitxor,             bitxor_trait,            Target::Trait;
    BitAndTraitLangItem(Op),       sym::bitand,             bitand_trait,            Target::Trait;
    BitOrTraitLangItem(Op),        sym::bitor,              bitor_trait,             Target::Trait;
    ShlTraitLangItem(Op),          sym::shl,                shl_trait,               Target::Trait;
    ShrTraitLangItem(Op),          sym::shr,                shr_trait,               Target::Trait;
    AddAssignTraitLangItem(Op),    sym::add_assign,         add_assign_trait,        Target::Trait;
    SubAssignTraitLangItem(Op),    sym::sub_assign,         sub_assign_trait,        Target::Trait;
    MulAssignTraitLangItem(Op),    sym::mul_assign,         mul_assign_trait,        Target::Trait;
    DivAssignTraitLangItem(Op),    sym::div_assign,         div_assign_trait,        Target::Trait;
    RemAssignTraitLangItem(Op),    sym::rem_assign,         rem_assign_trait,        Target::Trait;
    BitXorAssignTraitLangItem(Op), sym::bitxor_assign,      bitxor_assign_trait,     Target::Trait;
    BitAndAssignTraitLangItem(Op), sym::bitand_assign,      bitand_assign_trait,     Target::Trait;
    BitOrAssignTraitLangItem(Op),  sym::bitor_assign,       bitor_assign_trait,      Target::Trait;
    ShlAssignTraitLangItem(Op),    sym::shl_assign,         shl_assign_trait,        Target::Trait;
    ShrAssignTraitLangItem(Op),    sym::shr_assign,         shr_assign_trait,        Target::Trait;
    IndexTraitLangItem(Op),        sym::index,              index_trait,             Target::Trait;
    IndexMutTraitLangItem(Op),     sym::index_mut,          index_mut_trait,         Target::Trait;

    UnsafeCellTypeLangItem,        sym::unsafe_cell,        unsafe_cell_type,        Target::Struct;
    VaListTypeLangItem,            sym::va_list,            va_list,                 Target::Struct;

    DerefTraitLangItem,            sym::deref,              deref_trait,             Target::Trait;
    DerefMutTraitLangItem,         sym::deref_mut,          deref_mut_trait,         Target::Trait;
    ReceiverTraitLangItem,         sym::receiver,           receiver_trait,          Target::Trait;

    FnTraitLangItem,               kw::Fn,                  fn_trait,                Target::Trait;
    FnMutTraitLangItem,            sym::fn_mut,             fn_mut_trait,            Target::Trait;
    FnOnceTraitLangItem,           sym::fn_once,            fn_once_trait,           Target::Trait;

    FnOnceOutputLangItem,          sym::fn_once_output,     fn_once_output,          Target::AssocTy;

    FutureTraitLangItem,           sym::future_trait,       future_trait,            Target::Trait;
    GeneratorStateLangItem,        sym::generator_state,    gen_state,               Target::Enum;
    GeneratorTraitLangItem,        sym::generator,          gen_trait,               Target::Trait;
    UnpinTraitLangItem,            sym::unpin,              unpin_trait,             Target::Trait;
    PinTypeLangItem,               sym::pin,                pin_type,                Target::Struct;

    // Don't be fooled by the naming here: this lang item denotes `PartialEq`, not `Eq`.
    EqTraitLangItem,               sym::eq,                 eq_trait,                Target::Trait;
    PartialOrdTraitLangItem,       sym::partial_ord,        partial_ord_trait,       Target::Trait;

    // A number of panic-related lang items. The `panic` item corresponds to
    // divide-by-zero and various panic cases with `match`. The
    // `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of
    // a "weak lang item" in the sense that a crate is not required to have it
    // defined to use it, but a final product is required to define it
    // somewhere. Additionally, there are restrictions on crates that use a weak
    // lang item, but do not have it defined.
    PanicFnLangItem,               sym::panic,              panic_fn,                Target::Fn;
    PanicBoundsCheckFnLangItem,    sym::panic_bounds_check, panic_bounds_check_fn,   Target::Fn;
    PanicInfoLangItem,             sym::panic_info,         panic_info,              Target::Struct;
    PanicLocationLangItem,         sym::panic_location,     panic_location,          Target::Struct;
    PanicImplLangItem,             sym::panic_impl,         panic_impl,              Target::Fn;
    // Libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanicFnLangItem,          sym::begin_panic,        begin_panic_fn,          Target::Fn;

    ExchangeMallocFnLangItem,      sym::exchange_malloc,    exchange_malloc_fn,      Target::Fn;
    BoxFreeFnLangItem,             sym::box_free,           box_free_fn,             Target::Fn;
    DropInPlaceFnLangItem,         sym::drop_in_place,      drop_in_place_fn,        Target::Fn;
    OomLangItem,                   sym::oom,                oom,                     Target::Fn;
    AllocLayoutLangItem,           sym::alloc_layout,       alloc_layout,            Target::Struct;

    StartFnLangItem,               sym::start,              start_fn,                Target::Fn;

    EhPersonalityLangItem,         sym::eh_personality,     eh_personality,          Target::Fn;
    EhCatchTypeinfoLangItem,       sym::eh_catch_typeinfo,  eh_catch_typeinfo,       Target::Static;

    OwnedBoxLangItem,              sym::owned_box,          owned_box,               Target::Struct;

    PhantomDataItem,               sym::phantom_data,       phantom_data,            Target::Struct;

    ManuallyDropItem,              sym::manually_drop,      manually_drop,           Target::Struct;

    MaybeUninitLangItem,           sym::maybe_uninit,       maybe_uninit,            Target::Union;

    // Align offset for stride != 1; must not panic.
    AlignOffsetLangItem,           sym::align_offset,       align_offset_fn,         Target::Fn;

    TerminationTraitLangItem,      sym::termination,        termination,             Target::Trait;

    TryTraitLangItem,              kw::Try,                 try_trait,               Target::Trait;

    // language items related to source code coverage instrumentation (-Zinstrument-coverage)
    CountCodeRegionFnLangItem,         sym::count_code_region,         count_code_region_fn,         Target::Fn;
    CoverageCounterAddFnLangItem,      sym::coverage_counter_add,      coverage_counter_add_fn,      Target::Fn;
    CoverageCounterSubtractFnLangItem, sym::coverage_counter_subtract, coverage_counter_subtract_fn, Target::Fn;

    // Language items from AST lowering
    TryFromError,                  sym::from_error,         from_error_fn,           Target::Method(MethodKind::Trait { body: false });
    TryFromOk,                     sym::from_ok,            from_ok_fn,              Target::Method(MethodKind::Trait { body: false });
    TryIntoResult,                 sym::into_result,        into_result_fn,          Target::Method(MethodKind::Trait { body: false });

    PollReady,                     sym::Ready,              poll_ready_variant,      Target::Variant;
    PollPending,                   sym::Pending,            poll_pending_variant,    Target::Variant;

    FromGenerator,                 sym::from_generator,     from_generator_fn,       Target::Fn;
    GetContext,                    sym::get_context,        get_context_fn,          Target::Fn;

    FuturePoll,                    sym::poll,               future_poll_fn,          Target::Method(MethodKind::Trait { body: false });

    FromFrom,                      sym::from,               from_fn,                 Target::Method(MethodKind::Trait { body: false });

    OptionSome,                    sym::Some,               option_some_variant,     Target::Variant;
    OptionNone,                    sym::None,               option_none_variant,     Target::Variant;

    ResultOk,                      sym::Ok,                 result_ok_variant,       Target::Variant;
    ResultErr,                     sym::Err,                result_err_variant,      Target::Variant;

    IntoIterIntoIter,              sym::into_iter,          into_iter_fn,            Target::Method(MethodKind::Trait { body: false });
    IteratorNext,                  sym::next,               next_fn,                 Target::Method(MethodKind::Trait { body: false});

    PinNewUnchecked,               sym::new_unchecked,      new_unchecked_fn,        Target::Method(MethodKind::Inherent);

    RangeFrom,                     sym::RangeFrom,           range_from_struct,          Target::Struct;
    RangeFull,                     sym::RangeFull,           range_full_struct,          Target::Struct;
    RangeInclusiveStruct,          sym::RangeInclusive,      range_inclusive_struct,     Target::Struct;
    RangeInclusiveNew,             sym::range_inclusive_new, range_inclusive_new_method, Target::Method(MethodKind::Inherent);
    Range,                         sym::Range,               range_struct,               Target::Struct;
    RangeToInclusive,              sym::RangeToInclusive,    range_to_inclusive_struct,  Target::Struct;
    RangeTo,                       sym::RangeTo,             range_to_struct,            Target::Struct;
}
