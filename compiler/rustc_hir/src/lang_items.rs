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
        $( $(#[$attr:meta])* $variant:ident $($group:expr)?, $module:ident :: $name:ident, $method:ident, $target:expr, $generics:expr; )*
    ) => {

        enum_from_u32! {
            /// A representation of all the valid language items in Rust.
            #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable)]
            pub enum LangItem {
                $(
                    #[doc = concat!("The `", stringify!($name), "` lang item.")]
                    ///
                    $(#[$attr])*
                    $variant,
                )*
            }
        }

        impl LangItem {
            /// Returns the `name` symbol in `#[lang = "$name"]`.
            /// For example, [`LangItem::PartialEq`]`.name()`
            /// would result in [`sym::eq`] since it is `#[lang = "eq"]`.
            pub fn name(self) -> Symbol {
                match self {
                    $( LangItem::$variant => $module::$name, )*
                }
            }

            /// The [group](LangItemGroup) that this lang item belongs to,
            /// or `None` if it doesn't belong to a group.
            pub fn group(self) -> Option<LangItemGroup> {
                use LangItemGroup::*;
                match self {
                    $( LangItem::$variant => expand_group!($($group)*), )*
                }
            }

            pub fn required_generics(&self) -> GenericRequirement {
                match self {
                    $( LangItem::$variant => $generics, )*
                }
            }
        }

        /// All of the language items, defined or not.
        /// Defined lang items can come from the current crate or its dependencies.
        #[derive(HashStable_Generic, Debug)]
        pub struct LanguageItems {
            /// Mappings from lang items to their possibly found [`DefId`]s.
            /// The index corresponds to the order in [`LangItem`].
            pub items: Vec<Option<DefId>>,
            /// Lang items that were not found during collection.
            pub missing: Vec<LangItem>,
            /// Mapping from [`LangItemGroup`] discriminants to all
            /// [`DefId`]s of lang items in that group.
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

            /// Returns the [`DefId`]s of all lang items in a group.
            pub fn group(&self, group: LangItemGroup) -> &[DefId] {
                self.groups[group as usize].as_ref()
            }

            $(
                #[doc = concat!("Returns the [`DefId`] of the `", stringify!($name), "` lang item if it is defined.")]
                pub fn $method(&self) -> Option<DefId> {
                    self.items[LangItem::$variant as usize]
                }
            )*
        }

        /// A mapping from the name of the lang item to its order and the form it must be of.
        pub static ITEM_REFS: SyncLazy<FxHashMap<Symbol, (usize, Target)>> = SyncLazy::new(|| {
            let mut item_refs = FxHashMap::default();
            $( item_refs.insert($module::$name, (LangItem::$variant as usize, $target)); )*
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
/// avoid the need for `rustc_hir` to depend on `rustc_session`, so we
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
//  Variant name,            Name,                     Method name,                Target                  Generic requirements;
    Bool,                    sym::bool,                bool_impl,                  Target::Impl,           GenericRequirement::None;
    Char,                    sym::char,                char_impl,                  Target::Impl,           GenericRequirement::None;
    Str,                     sym::str,                 str_impl,                   Target::Impl,           GenericRequirement::None;
    Array,                   sym::array,               array_impl,                 Target::Impl,           GenericRequirement::None;
    Slice,                   sym::slice,               slice_impl,                 Target::Impl,           GenericRequirement::None;
    SliceU8,                 sym::slice_u8,            slice_u8_impl,              Target::Impl,           GenericRequirement::None;
    StrAlloc,                sym::str_alloc,           str_alloc_impl,             Target::Impl,           GenericRequirement::None;
    SliceAlloc,              sym::slice_alloc,         slice_alloc_impl,           Target::Impl,           GenericRequirement::None;
    SliceU8Alloc,            sym::slice_u8_alloc,      slice_u8_alloc_impl,        Target::Impl,           GenericRequirement::None;
    ConstPtr,                sym::const_ptr,           const_ptr_impl,             Target::Impl,           GenericRequirement::None;
    MutPtr,                  sym::mut_ptr,             mut_ptr_impl,               Target::Impl,           GenericRequirement::None;
    ConstSlicePtr,           sym::const_slice_ptr,     const_slice_ptr_impl,       Target::Impl,           GenericRequirement::None;
    MutSlicePtr,             sym::mut_slice_ptr,       mut_slice_ptr_impl,         Target::Impl,           GenericRequirement::None;
    I8,                      sym::i8,                  i8_impl,                    Target::Impl,           GenericRequirement::None;
    I16,                     sym::i16,                 i16_impl,                   Target::Impl,           GenericRequirement::None;
    I32,                     sym::i32,                 i32_impl,                   Target::Impl,           GenericRequirement::None;
    I64,                     sym::i64,                 i64_impl,                   Target::Impl,           GenericRequirement::None;
    I128,                    sym::i128,                i128_impl,                  Target::Impl,           GenericRequirement::None;
    Isize,                   sym::isize,               isize_impl,                 Target::Impl,           GenericRequirement::None;
    U8,                      sym::u8,                  u8_impl,                    Target::Impl,           GenericRequirement::None;
    U16,                     sym::u16,                 u16_impl,                   Target::Impl,           GenericRequirement::None;
    U32,                     sym::u32,                 u32_impl,                   Target::Impl,           GenericRequirement::None;
    U64,                     sym::u64,                 u64_impl,                   Target::Impl,           GenericRequirement::None;
    U128,                    sym::u128,                u128_impl,                  Target::Impl,           GenericRequirement::None;
    Usize,                   sym::usize,               usize_impl,                 Target::Impl,           GenericRequirement::None;
    F32,                     sym::f32,                 f32_impl,                   Target::Impl,           GenericRequirement::None;
    F64,                     sym::f64,                 f64_impl,                   Target::Impl,           GenericRequirement::None;
    F32Runtime,              sym::f32_runtime,         f32_runtime_impl,           Target::Impl,           GenericRequirement::None;
    F64Runtime,              sym::f64_runtime,         f64_runtime_impl,           Target::Impl,           GenericRequirement::None;

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

    Drop,                    sym::drop,                drop_trait,                 Target::Trait,          GenericRequirement::None;

    CoerceUnsized,           sym::coerce_unsized,      coerce_unsized_trait,       Target::Trait,          GenericRequirement::Minimum(1);
    DispatchFromDyn,         sym::dispatch_from_dyn,   dispatch_from_dyn_trait,    Target::Trait,          GenericRequirement::Minimum(1);

    Add(Op),                 sym::add,                 add_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Sub(Op),                 sym::sub,                 sub_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Mul(Op),                 sym::mul,                 mul_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Div(Op),                 sym::div,                 div_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Rem(Op),                 sym::rem,                 rem_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Neg(Op),                 sym::neg,                 neg_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    Not(Op),                 sym::not,                 not_trait,                  Target::Trait,          GenericRequirement::Exact(0);
    BitXor(Op),              sym::bitxor,              bitxor_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitAnd(Op),              sym::bitand,              bitand_trait,               Target::Trait,          GenericRequirement::Exact(1);
    BitOr(Op),               sym::bitor,               bitor_trait,                Target::Trait,          GenericRequirement::Exact(1);
    Shl(Op),                 sym::shl,                 shl_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    Shr(Op),                 sym::shr,                 shr_trait,                  Target::Trait,          GenericRequirement::Exact(1);
    AddAssign(Op),           sym::add_assign,          add_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    SubAssign(Op),           sym::sub_assign,          sub_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    MulAssign(Op),           sym::mul_assign,          mul_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    DivAssign(Op),           sym::div_assign,          div_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    RemAssign(Op),           sym::rem_assign,          rem_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    BitXorAssign(Op),        sym::bitxor_assign,       bitxor_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitAndAssign(Op),        sym::bitand_assign,       bitand_assign_trait,        Target::Trait,          GenericRequirement::Exact(1);
    BitOrAssign(Op),         sym::bitor_assign,        bitor_assign_trait,         Target::Trait,          GenericRequirement::Exact(1);
    ShlAssign(Op),           sym::shl_assign,          shl_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    ShrAssign(Op),           sym::shr_assign,          shr_assign_trait,           Target::Trait,          GenericRequirement::Exact(1);
    Index(Op),               sym::index,               index_trait,                Target::Trait,          GenericRequirement::Exact(1);
    IndexMut(Op),            sym::index_mut,           index_mut_trait,            Target::Trait,          GenericRequirement::Exact(1);

    UnsafeCell,              sym::unsafe_cell,         unsafe_cell_type,           Target::Struct,         GenericRequirement::None;
    VaList,                  sym::va_list,             va_list,                    Target::Struct,         GenericRequirement::None;

    Deref,                   sym::deref,               deref_trait,                Target::Trait,          GenericRequirement::Exact(0);
    DerefMut,                sym::deref_mut,           deref_mut_trait,            Target::Trait,          GenericRequirement::Exact(0);
    DerefTarget,             sym::deref_target,        deref_target,               Target::AssocTy,        GenericRequirement::None;
    Receiver,                sym::receiver,            receiver_trait,             Target::Trait,          GenericRequirement::None;

    Fn,                      kw::Fn,                   fn_trait,                   Target::Trait,          GenericRequirement::Exact(1);
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

    // A number of panic-related lang items. The `panic` item corresponds to divide-by-zero and
    // various panic cases with `match`. The `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of a "weak lang item"
    // in the sense that a crate is not required to have it defined to use it, but a final product
    // is required to define it somewhere. Additionally, there are restrictions on crates that use
    // a weak lang item, but do not have it defined.
    Panic,                   sym::panic,               panic_fn,                   Target::Fn,             GenericRequirement::None;
    PanicFmt,                sym::panic_fmt,           panic_fmt,                  Target::Fn,             GenericRequirement::None;
    PanicDisplay,            sym::panic_display,       panic_display,              Target::Fn,             GenericRequirement::None;
    PanicStr,                sym::panic_str,           panic_str,                  Target::Fn,             GenericRequirement::None;
    ConstPanicFmt,           sym::const_panic_fmt,     const_panic_fmt,            Target::Fn,             GenericRequirement::None;
    PanicBoundsCheck,        sym::panic_bounds_check,  panic_bounds_check_fn,      Target::Fn,             GenericRequirement::None;
    PanicInfo,               sym::panic_info,          panic_info,                 Target::Struct,         GenericRequirement::None;
    PanicLocation,           sym::panic_location,      panic_location,             Target::Struct,         GenericRequirement::None;
    PanicImpl,               sym::panic_impl,          panic_impl,                 Target::Fn,             GenericRequirement::None;
    /// libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanic,              sym::begin_panic,         begin_panic_fn,             Target::Fn,             GenericRequirement::None;
    BeginPanicFmt,           sym::begin_panic_fmt,     begin_panic_fmt,            Target::Fn,             GenericRequirement::None;

    ExchangeMalloc,          sym::exchange_malloc,     exchange_malloc_fn,         Target::Fn,             GenericRequirement::None;
    BoxFree,                 sym::box_free,            box_free_fn,                Target::Fn,             GenericRequirement::Minimum(1);
    DropInPlace,             sym::drop_in_place,       drop_in_place_fn,           Target::Fn,             GenericRequirement::Minimum(1);
    Oom,                     sym::oom,                 oom,                        Target::Fn,             GenericRequirement::None;
    AllocLayout,             sym::alloc_layout,        alloc_layout,               Target::Struct,         GenericRequirement::None;
    ConstEvalSelect,         sym::const_eval_select,   const_eval_select,          Target::Fn,             GenericRequirement::Exact(4);
    ConstConstEvalSelect,    sym::const_eval_select_ct,const_eval_select_ct,       Target::Fn,             GenericRequirement::Exact(4);

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

    SliceLen,                sym::slice_len_fn,        slice_len_fn,               Target::Method(MethodKind::Inherent), GenericRequirement::None;

    // Language items from AST lowering
    TryTraitFromResidual,    sym::from_residual,       from_residual_fn,           Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitFromOutput,      sym::from_output,         from_output_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;
    TryTraitBranch,          sym::branch,              branch_fn,                  Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    PollReady,               sym::Ready,               poll_ready_variant,         Target::Variant,        GenericRequirement::None;
    PollPending,             sym::Pending,             poll_pending_variant,       Target::Variant,        GenericRequirement::None;

    FromGenerator,           sym::from_generator,      from_generator_fn,          Target::Fn,             GenericRequirement::None;
    GetContext,              sym::get_context,         get_context_fn,             Target::Fn,             GenericRequirement::None;

    FuturePoll,              sym::poll,                future_poll_fn,             Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    FromFrom,                sym::from,                from_fn,                    Target::Method(MethodKind::Trait { body: false }), GenericRequirement::None;

    OptionSome,              sym::Some,                option_some_variant,        Target::Variant,        GenericRequirement::None;
    OptionNone,              sym::None,                option_none_variant,        Target::Variant,        GenericRequirement::None;

    ResultOk,                sym::Ok,                  result_ok_variant,          Target::Variant,        GenericRequirement::None;
    ResultErr,               sym::Err,                 result_err_variant,         Target::Variant,        GenericRequirement::None;

    ControlFlowContinue,     sym::Continue,            cf_continue_variant,        Target::Variant,        GenericRequirement::None;
    ControlFlowBreak,        sym::Break,               cf_break_variant,           Target::Variant,        GenericRequirement::None;

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
}

pub enum GenericRequirement {
    None,
    Minimum(usize),
    Exact(usize),
}
