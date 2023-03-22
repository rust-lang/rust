//! Defines language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::def_id::DefId;
use crate::errors::LangItemError;
use crate::{MethodKind, Target};

use rustc_ast as ast;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable_Generic;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

/// All of the language items, defined or not.
/// Defined lang items can come from the current crate or its dependencies.
#[derive(HashStable_Generic, Debug)]
pub struct LanguageItems {
    /// Mappings from lang items to their possibly found [`DefId`]s.
    /// The index corresponds to the order in [`LangItem`].
    items: [Option<DefId>; std::mem::variant_count::<LangItem>()],
    /// Lang items that were not found during collection.
    pub missing: Vec<LangItem>,
}

impl LanguageItems {
    /// Construct an empty collection of lang items and no missing ones.
    pub fn new() -> Self {
        Self { items: [None; std::mem::variant_count::<LangItem>()], missing: Vec::new() }
    }

    pub fn get(&self, item: LangItem) -> Option<DefId> {
        self.items[item as usize]
    }

    pub fn set(&mut self, item: LangItem, def_id: DefId) {
        self.items[item as usize] = Some(def_id);
    }

    /// Requires that a given `LangItem` was bound and returns the corresponding `DefId`.
    /// If it wasn't bound, e.g. due to a missing `#[lang = "<it.name()>"]`,
    /// returns an error encapsulating the `LangItem`.
    pub fn require(&self, it: LangItem) -> Result<DefId, LangItemError> {
        self.get(it).ok_or_else(|| LangItemError(it))
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (LangItem, DefId)> + 'a {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(i, id)| id.map(|id| (LangItem::from_u32(i as u32).unwrap(), id)))
    }
}

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! language_item_table {
    (
        $( $(#[$attr:meta])* $variant:ident, $module:ident :: $name:ident, $method:ident, $target:expr, $generics:expr; )*
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

            /// Opposite of [`LangItem::name`]
            pub fn from_name(name: Symbol) -> Option<Self> {
                match name {
                    $( $module::$name => Some(LangItem::$variant), )*
                    _ => None,
                }
            }

            /// Returns the name of the `LangItem` enum variant.
            // This method is used by Clippy for internal lints.
            pub fn variant_name(self) -> &'static str {
                match self {
                    $( LangItem::$variant => stringify!($variant), )*
                }
            }

            pub fn target(self) -> Target {
                match self {
                    $( LangItem::$variant => $target, )*
                }
            }

            pub fn required_generics(&self) -> GenericRequirement {
                match self {
                    $( LangItem::$variant => $generics, )*
                }
            }
        }

        impl LanguageItems {
            $(
                #[doc = concat!("Returns the [`DefId`] of the `", stringify!($name), "` lang item if it is defined.")]
                pub fn $method(&self) -> Option<DefId> {
                    self.items[LangItem::$variant as usize]
                }
            )*
        }
    }
}

impl<CTX> HashStable<CTX> for LangItem {
    fn hash_stable(&self, _: &mut CTX, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

/// Extracts the first `lang = "$name"` out of a list of attributes.
/// The `#[panic_handler]` attribute is also extracted out when found.
pub fn extract(attrs: &[ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if attr.has_name(sym::lang) => (attr.value_str()?, attr.span),
            _ if attr.has_name(sym::panic_handler) => (sym::panic_impl, attr.span),
            _ => return None,
        })
    })
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
    Panic,                   sym::panic,               panic_fn,                   Target::Fn,             GenericRequirement::Exact(0);
    PanicNounwind,           sym::panic_nounwind,      panic_nounwind,             Target::Fn,             GenericRequirement::Exact(0);
    PanicFmt,                sym::panic_fmt,           panic_fmt,                  Target::Fn,             GenericRequirement::None;
    PanicDisplay,            sym::panic_display,       panic_display,              Target::Fn,             GenericRequirement::None;
    ConstPanicFmt,           sym::const_panic_fmt,     const_panic_fmt,            Target::Fn,             GenericRequirement::None;
    PanicBoundsCheck,        sym::panic_bounds_check,  panic_bounds_check_fn,      Target::Fn,             GenericRequirement::Exact(0);
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
}

pub enum GenericRequirement {
    None,
    Minimum(usize),
    Exact(usize),
}

pub static FN_TRAITS: &'static [LangItem] = &[LangItem::Fn, LangItem::FnMut, LangItem::FnOnce];

pub static OPERATORS: &'static [LangItem] = &[
    LangItem::Add,
    LangItem::Sub,
    LangItem::Mul,
    LangItem::Div,
    LangItem::Rem,
    LangItem::Neg,
    LangItem::Not,
    LangItem::BitXor,
    LangItem::BitAnd,
    LangItem::BitOr,
    LangItem::Shl,
    LangItem::Shr,
    LangItem::AddAssign,
    LangItem::SubAssign,
    LangItem::MulAssign,
    LangItem::DivAssign,
    LangItem::RemAssign,
    LangItem::BitXorAssign,
    LangItem::BitAndAssign,
    LangItem::BitOrAssign,
    LangItem::ShlAssign,
    LangItem::ShrAssign,
    LangItem::Index,
    LangItem::IndexMut,
    LangItem::PartialEq,
    LangItem::PartialOrd,
];
