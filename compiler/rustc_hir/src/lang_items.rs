//! Defines language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::def_id::DefId;
use crate::{MethodKind, Target};

use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable_Generic;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use std::lazy::SyncLazy;

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
                    $( LangItem::$variant => $name, )*
                }
            }

            pub fn group(self) -> Option<LangItemGroup> {
                use LangItemGroup::*;
                match self {
                    $( LangItem::$variant => expand_group!($($group)*), )*
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
                    items: vec![$(init_none(LangItem::$variant)),*],
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
                    self.items[LangItem::$variant as usize]
                }
            )*
        }

        /// A mapping from the name of the lang item to its order and the form it must be of.
        pub static ITEM_REFS: SyncLazy<FxHashMap<Symbol, (usize, Target)>> = SyncLazy::new(|| {
            let mut item_refs = FxHashMap::default();
            $( item_refs.insert($name, (LangItem::$variant as usize, $target)); )*
            item_refs
        });

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
//  Variant name,            Name,                    Method name,             Target;
    Bool,                    sym::bool,                bool_impl,                  Target::Impl;
    Char,                    sym::char,                char_impl,                  Target::Impl;
    Str,                     sym::str,                 str_impl,                   Target::Impl;
    Array,                   sym::array,               array_impl,                 Target::Impl;
    Slice,                   sym::slice,               slice_impl,                 Target::Impl;
    SliceU8,                 sym::slice_u8,            slice_u8_impl,              Target::Impl;
    StrAlloc,                sym::str_alloc,           str_alloc_impl,             Target::Impl;
    SliceAlloc,              sym::slice_alloc,         slice_alloc_impl,           Target::Impl;
    SliceU8Alloc,            sym::slice_u8_alloc,      slice_u8_alloc_impl,        Target::Impl;
    ConstPtr,                sym::const_ptr,           const_ptr_impl,             Target::Impl;
    MutPtr,                  sym::mut_ptr,             mut_ptr_impl,               Target::Impl;
    ConstSlicePtr,           sym::const_slice_ptr,     const_slice_ptr_impl,       Target::Impl;
    MutSlicePtr,             sym::mut_slice_ptr,       mut_slice_ptr_impl,         Target::Impl;
    I8,                      sym::i8,                  i8_impl,                    Target::Impl;
    I16,                     sym::i16,                 i16_impl,                   Target::Impl;
    I32,                     sym::i32,                 i32_impl,                   Target::Impl;
    I64,                     sym::i64,                 i64_impl,                   Target::Impl;
    I128,                    sym::i128,                i128_impl,                  Target::Impl;
    Isize,                   sym::isize,               isize_impl,                 Target::Impl;
    U8,                      sym::u8,                  u8_impl,                    Target::Impl;
    U16,                     sym::u16,                 u16_impl,                   Target::Impl;
    U32,                     sym::u32,                 u32_impl,                   Target::Impl;
    U64,                     sym::u64,                 u64_impl,                   Target::Impl;
    U128,                    sym::u128,                u128_impl,                  Target::Impl;
    Usize,                   sym::usize,               usize_impl,                 Target::Impl;
    F32,                     sym::f32,                 f32_impl,                   Target::Impl;
    F64,                     sym::f64,                 f64_impl,                   Target::Impl;
    F32Runtime,              sym::f32_runtime,         f32_runtime_impl,           Target::Impl;
    F64Runtime,              sym::f64_runtime,         f64_runtime_impl,           Target::Impl;

    Sized,                   sym::sized,               sized_trait,                Target::Trait;
    Unsize,                  sym::unsize,              unsize_trait,               Target::Trait;
    // Trait injected by #[derive(PartialEq)], (i.e. "Partial EQ").
    StructuralPeq,           sym::structural_peq,      structural_peq_trait,       Target::Trait;
    // Trait injected by #[derive(Eq)], (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeq,           sym::structural_teq,      structural_teq_trait,       Target::Trait;
    Copy,                    sym::copy,                copy_trait,                 Target::Trait;
    Clone,                   sym::clone,               clone_trait,                Target::Trait;
    Sync,                    sym::sync,                sync_trait,                 Target::Trait;
    DiscriminantKind,        sym::discriminant_kind,   discriminant_kind_trait,    Target::Trait;
    // The associated item of `trait DiscriminantKind`.
    Discriminant,            sym::discriminant_type,   discriminant_type,          Target::AssocTy;

    Freeze,                  sym::freeze,              freeze_trait,               Target::Trait;

    Drop,                    sym::drop,                drop_trait,                 Target::Trait;

    CoerceUnsized,           sym::coerce_unsized,      coerce_unsized_trait,       Target::Trait;
    DispatchFromDyn,         sym::dispatch_from_dyn,   dispatch_from_dyn_trait,    Target::Trait;

    Add(Op),                 sym::add,                 add_trait,                  Target::Trait;
    Sub(Op),                 sym::sub,                 sub_trait,                  Target::Trait;
    Mul(Op),                 sym::mul,                 mul_trait,                  Target::Trait;
    Div(Op),                 sym::div,                 div_trait,                  Target::Trait;
    Rem(Op),                 sym::rem,                 rem_trait,                  Target::Trait;
    Neg(Op),                 sym::neg,                 neg_trait,                  Target::Trait;
    Not(Op),                 sym::not,                 not_trait,                  Target::Trait;
    BitXor(Op),              sym::bitxor,              bitxor_trait,               Target::Trait;
    BitAnd(Op),              sym::bitand,              bitand_trait,               Target::Trait;
    BitOr(Op),               sym::bitor,               bitor_trait,                Target::Trait;
    Shl(Op),                 sym::shl,                 shl_trait,                  Target::Trait;
    Shr(Op),                 sym::shr,                 shr_trait,                  Target::Trait;
    AddAssign(Op),           sym::add_assign,          add_assign_trait,           Target::Trait;
    SubAssign(Op),           sym::sub_assign,          sub_assign_trait,           Target::Trait;
    MulAssign(Op),           sym::mul_assign,          mul_assign_trait,           Target::Trait;
    DivAssign(Op),           sym::div_assign,          div_assign_trait,           Target::Trait;
    RemAssign(Op),           sym::rem_assign,          rem_assign_trait,           Target::Trait;
    BitXorAssign(Op),        sym::bitxor_assign,       bitxor_assign_trait,        Target::Trait;
    BitAndAssign(Op),        sym::bitand_assign,       bitand_assign_trait,        Target::Trait;
    BitOrAssign(Op),         sym::bitor_assign,        bitor_assign_trait,         Target::Trait;
    ShlAssign(Op),           sym::shl_assign,          shl_assign_trait,           Target::Trait;
    ShrAssign(Op),           sym::shr_assign,          shr_assign_trait,           Target::Trait;
    Index(Op),               sym::index,               index_trait,                Target::Trait;
    IndexMut(Op),            sym::index_mut,           index_mut_trait,            Target::Trait;

    UnsafeCell,              sym::unsafe_cell,         unsafe_cell_type,           Target::Struct;
    VaList,                  sym::va_list,             va_list,                    Target::Struct;

    Deref,                   sym::deref,               deref_trait,                Target::Trait;
    DerefMut,                sym::deref_mut,           deref_mut_trait,            Target::Trait;
    Receiver,                sym::receiver,            receiver_trait,             Target::Trait;

    Fn,                      kw::Fn,                   fn_trait,                   Target::Trait;
    FnMut,                   sym::fn_mut,              fn_mut_trait,               Target::Trait;
    FnOnce,                  sym::fn_once,             fn_once_trait,              Target::Trait;

    FnOnceOutput,            sym::fn_once_output,      fn_once_output,             Target::AssocTy;

    Future,                  sym::future_trait,        future_trait,               Target::Trait;
    GeneratorState,          sym::generator_state,     gen_state,                  Target::Enum;
    Generator,               sym::generator,           gen_trait,                  Target::Trait;
    Unpin,                   sym::unpin,               unpin_trait,                Target::Trait;
    Pin,                     sym::pin,                 pin_type,                   Target::Struct;

    PartialEq,               sym::eq,                  eq_trait,                   Target::Trait;
    PartialOrd,              sym::partial_ord,         partial_ord_trait,          Target::Trait;

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               panic_fn,                   Target::Fn;
    PanicStr,                sym::panic_str,           panic_str,                  Target::Fn;
    PanicBoundsCheck,        sym::panic_bounds_check,  panic_bounds_check_fn,      Target::Fn;
    PanicInfo,               sym::panic_info,          panic_info,                 Target::Struct;
    PanicLocation,           sym::panic_location,      panic_location,             Target::Struct;
    PanicImpl,               sym::panic_impl,          panic_impl,                 Target::Fn;
    // libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         begin_panic_fn,             Target::Fn;

    ExchangeMalloc,          sym::exchange_malloc,     exchange_malloc_fn,         Target::Fn;
    BoxFree,                 sym::box_free,            box_free_fn,                Target::Fn;
    DropInPlace,             sym::drop_in_place,       drop_in_place_fn,           Target::Fn;
    Oom,                     sym::oom,                 oom,                        Target::Fn;
    AllocLayout,             sym::alloc_layout,        alloc_layout,               Target::Struct;

    Start,                   sym::start,               start_fn,                   Target::Fn;

    EhPersonality,           sym::eh_personality,      eh_personality,             Target::Fn;
    EhCatchTypeinfo,         sym::eh_catch_typeinfo,   eh_catch_typeinfo,          Target::Static;

    OwnedBox,                sym::owned_box,           owned_box,                  Target::Struct;

    PhantomData,             sym::phantom_data,        phantom_data,               Target::Struct;

    ManuallyDrop,            sym::manually_drop,       manually_drop,              Target::Struct;

    MaybeUninit,             sym::maybe_uninit,        maybe_uninit,               Target::Union;

    // Align offset for stride != 1; must not panic.
    AlignOffset,             sym::align_offset,        align_offset_fn,            Target::Fn;

    Termination,             sym::termination,         termination,                Target::Trait;

    Try,                     kw::Try,                  try_trait,                  Target::Trait;

    // Language items from AST lowering
    TryFromError,            sym::from_error,          from_error_fn,              Target::Method(MethodKind::Trait { body: false });
    TryFromOk,               sym::from_ok,             from_ok_fn,                 Target::Method(MethodKind::Trait { body: false });
    TryIntoResult,           sym::into_result,         into_result_fn,             Target::Method(MethodKind::Trait { body: false });

    PollReady,               sym::Ready,               poll_ready_variant,         Target::Variant;
    PollPending,             sym::Pending,             poll_pending_variant,       Target::Variant;

    FromGenerator,           sym::from_generator,      from_generator_fn,          Target::Fn;
    GetContext,              sym::get_context,         get_context_fn,             Target::Fn;

    FuturePoll,              sym::poll,                future_poll_fn,             Target::Method(MethodKind::Trait { body: false });

    FromFrom,                sym::from,                from_fn,                    Target::Method(MethodKind::Trait { body: false });

    OptionSome,              sym::Some,                option_some_variant,        Target::Variant;
    OptionNone,              sym::None,                option_none_variant,        Target::Variant;

    ResultOk,                sym::Ok,                  result_ok_variant,          Target::Variant;
    ResultErr,               sym::Err,                 result_err_variant,         Target::Variant;

    IntoIterIntoIter,        sym::into_iter,           into_iter_fn,               Target::Method(MethodKind::Trait { body: false });
    IteratorNext,            sym::next,                next_fn,                    Target::Method(MethodKind::Trait { body: false});

    PinNewUnchecked,         sym::new_unchecked,       new_unchecked_fn,           Target::Method(MethodKind::Inherent);

    RangeFrom,               sym::RangeFrom,           range_from_struct,          Target::Struct;
    RangeFull,               sym::RangeFull,           range_full_struct,          Target::Struct;
    RangeInclusiveStruct,    sym::RangeInclusive,      range_inclusive_struct,     Target::Struct;
    RangeInclusiveNew,       sym::range_inclusive_new, range_inclusive_new_method, Target::Method(MethodKind::Inherent);
    Range,                   sym::Range,               range_struct,               Target::Struct;
    RangeToInclusive,        sym::RangeToInclusive,    range_to_inclusive_struct,  Target::Struct;
    RangeTo,                 sym::RangeTo,             range_to_struct,            Target::Struct;
}
