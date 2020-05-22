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
use crate::Target;

use rustc_ast::ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable_Generic;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use lazy_static::lazy_static;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! language_item_table {
    (
        $( $variant:ident, $name:expr, $method:ident, $target:path; )*
    ) => {

        enum_from_u32! {
            /// A representation of all the valid language items in Rust.
            #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
            pub enum LangItem {
                $($variant,)*
            }
        }

        impl LangItem {
            /// Returns the `name` in `#[lang = "$name"]`.
            /// For example, `LangItem::EqTraitLangItem`,
            /// that is `#[lang = "eq"]` would result in `"eq"`.
            pub fn name(self) -> &'static str {
                match self {
                    $( $variant => $name, )*
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
        }

        impl LanguageItems {
            /// Construct an empty collection of lang items and no missing ones.
            pub fn new() -> Self {
                fn init_none(_: LangItem) -> Option<DefId> { None }

                Self {
                    items: vec![$(init_none($variant)),*],
                    missing: Vec::new(),
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

            $(
                /// Returns the corresponding `DefId` for the lang item
                #[doc = $name]
                /// if it exists.
                #[allow(dead_code)]
                pub fn $method(&self) -> Option<DefId> {
                    self.items[$variant as usize]
                }
            )*
        }

        lazy_static! {
            /// A mapping from the name of the lang item to its order and the form it must be of.
            pub static ref ITEM_REFS: FxHashMap<&'static str, (usize, Target)> = {
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
pub fn extract(attrs: &[ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if attr.check_name(sym::lang) => (attr.value_str()?, attr.span),
            _ if attr.check_name(sym::panic_handler) => (sym::panic_impl, attr.span),
            _ if attr.check_name(sym::alloc_error_handler) => (sym::oom, attr.span),
            _ => return None,
        })
    })
}

language_item_table! {
//  Variant name,                Name,                 Method name,             Target;
    BoolImplItem,                "bool",               bool_impl,               Target::Impl;
    CharImplItem,                "char",               char_impl,               Target::Impl;
    StrImplItem,                 "str",                str_impl,                Target::Impl;
    SliceImplItem,               "slice",              slice_impl,              Target::Impl;
    SliceU8ImplItem,             "slice_u8",           slice_u8_impl,           Target::Impl;
    StrAllocImplItem,            "str_alloc",          str_alloc_impl,          Target::Impl;
    SliceAllocImplItem,          "slice_alloc",        slice_alloc_impl,        Target::Impl;
    SliceU8AllocImplItem,        "slice_u8_alloc",     slice_u8_alloc_impl,     Target::Impl;
    ConstPtrImplItem,            "const_ptr",          const_ptr_impl,          Target::Impl;
    MutPtrImplItem,              "mut_ptr",            mut_ptr_impl,            Target::Impl;
    ConstSlicePtrImplItem,       "const_slice_ptr",    const_slice_ptr_impl,    Target::Impl;
    MutSlicePtrImplItem,         "mut_slice_ptr",      mut_slice_ptr_impl,      Target::Impl;
    I8ImplItem,                  "i8",                 i8_impl,                 Target::Impl;
    I16ImplItem,                 "i16",                i16_impl,                Target::Impl;
    I32ImplItem,                 "i32",                i32_impl,                Target::Impl;
    I64ImplItem,                 "i64",                i64_impl,                Target::Impl;
    I128ImplItem,                "i128",               i128_impl,               Target::Impl;
    IsizeImplItem,               "isize",              isize_impl,              Target::Impl;
    U8ImplItem,                  "u8",                 u8_impl,                 Target::Impl;
    U16ImplItem,                 "u16",                u16_impl,                Target::Impl;
    U32ImplItem,                 "u32",                u32_impl,                Target::Impl;
    U64ImplItem,                 "u64",                u64_impl,                Target::Impl;
    U128ImplItem,                "u128",               u128_impl,               Target::Impl;
    UsizeImplItem,               "usize",              usize_impl,              Target::Impl;
    F32ImplItem,                 "f32",                f32_impl,                Target::Impl;
    F64ImplItem,                 "f64",                f64_impl,                Target::Impl;
    F32RuntimeImplItem,          "f32_runtime",        f32_runtime_impl,        Target::Impl;
    F64RuntimeImplItem,          "f64_runtime",        f64_runtime_impl,        Target::Impl;

    SizedTraitLangItem,          "sized",              sized_trait,             Target::Trait;
    UnsizeTraitLangItem,         "unsize",             unsize_trait,            Target::Trait;
    // trait injected by #[derive(PartialEq)], (i.e. "Partial EQ").
    StructuralPeqTraitLangItem,  "structural_peq",     structural_peq_trait,    Target::Trait;
    // trait injected by #[derive(Eq)], (i.e. "Total EQ"; no, I will not apologize).
    StructuralTeqTraitLangItem,  "structural_teq",     structural_teq_trait,    Target::Trait;
    CopyTraitLangItem,           "copy",               copy_trait,              Target::Trait;
    CloneTraitLangItem,          "clone",              clone_trait,             Target::Trait;
    SyncTraitLangItem,           "sync",               sync_trait,              Target::Trait;
    DiscriminantKindTraitLangItem,"discriminant_kind", discriminant_kind_trait, Target::Trait;
    FreezeTraitLangItem,         "freeze",             freeze_trait,            Target::Trait;

    DropTraitLangItem,           "drop",               drop_trait,              Target::Trait;

    CoerceUnsizedTraitLangItem,  "coerce_unsized",     coerce_unsized_trait,    Target::Trait;
    DispatchFromDynTraitLangItem,"dispatch_from_dyn",  dispatch_from_dyn_trait, Target::Trait;

    AddTraitLangItem,            "add",                add_trait,               Target::Trait;
    SubTraitLangItem,            "sub",                sub_trait,               Target::Trait;
    MulTraitLangItem,            "mul",                mul_trait,               Target::Trait;
    DivTraitLangItem,            "div",                div_trait,               Target::Trait;
    RemTraitLangItem,            "rem",                rem_trait,               Target::Trait;
    NegTraitLangItem,            "neg",                neg_trait,               Target::Trait;
    NotTraitLangItem,            "not",                not_trait,               Target::Trait;
    BitXorTraitLangItem,         "bitxor",             bitxor_trait,            Target::Trait;
    BitAndTraitLangItem,         "bitand",             bitand_trait,            Target::Trait;
    BitOrTraitLangItem,          "bitor",              bitor_trait,             Target::Trait;
    ShlTraitLangItem,            "shl",                shl_trait,               Target::Trait;
    ShrTraitLangItem,            "shr",                shr_trait,               Target::Trait;
    AddAssignTraitLangItem,      "add_assign",         add_assign_trait,        Target::Trait;
    SubAssignTraitLangItem,      "sub_assign",         sub_assign_trait,        Target::Trait;
    MulAssignTraitLangItem,      "mul_assign",         mul_assign_trait,        Target::Trait;
    DivAssignTraitLangItem,      "div_assign",         div_assign_trait,        Target::Trait;
    RemAssignTraitLangItem,      "rem_assign",         rem_assign_trait,        Target::Trait;
    BitXorAssignTraitLangItem,   "bitxor_assign",      bitxor_assign_trait,     Target::Trait;
    BitAndAssignTraitLangItem,   "bitand_assign",      bitand_assign_trait,     Target::Trait;
    BitOrAssignTraitLangItem,    "bitor_assign",       bitor_assign_trait,      Target::Trait;
    ShlAssignTraitLangItem,      "shl_assign",         shl_assign_trait,        Target::Trait;
    ShrAssignTraitLangItem,      "shr_assign",         shr_assign_trait,        Target::Trait;
    IndexTraitLangItem,          "index",              index_trait,             Target::Trait;
    IndexMutTraitLangItem,       "index_mut",          index_mut_trait,         Target::Trait;

    UnsafeCellTypeLangItem,      "unsafe_cell",        unsafe_cell_type,        Target::Struct;
    VaListTypeLangItem,          "va_list",            va_list,                 Target::Struct;

    DerefTraitLangItem,          "deref",              deref_trait,             Target::Trait;
    DerefMutTraitLangItem,       "deref_mut",          deref_mut_trait,         Target::Trait;
    ReceiverTraitLangItem,       "receiver",           receiver_trait,          Target::Trait;

    FnTraitLangItem,             "fn",                 fn_trait,                Target::Trait;
    FnMutTraitLangItem,          "fn_mut",             fn_mut_trait,            Target::Trait;
    FnOnceTraitLangItem,         "fn_once",            fn_once_trait,           Target::Trait;

    FutureTraitLangItem,         "future_trait",       future_trait,            Target::Trait;
    GeneratorStateLangItem,      "generator_state",    gen_state,               Target::Enum;
    GeneratorTraitLangItem,      "generator",          gen_trait,               Target::Trait;
    UnpinTraitLangItem,          "unpin",              unpin_trait,             Target::Trait;
    PinTypeLangItem,             "pin",                pin_type,                Target::Struct;

    // Don't be fooled by the naming here: this lang item denotes `PartialEq`, not `Eq`.
    EqTraitLangItem,             "eq",                 eq_trait,                Target::Trait;
    PartialOrdTraitLangItem,     "partial_ord",        partial_ord_trait,       Target::Trait;

    // A number of panic-related lang items. The `panic` item corresponds to
    // divide-by-zero and various panic cases with `match`. The
    // `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of
    // a "weak lang item" in the sense that a crate is not required to have it
    // defined to use it, but a final product is required to define it
    // somewhere. Additionally, there are restrictions on crates that use a weak
    // lang item, but do not have it defined.
    PanicFnLangItem,             "panic",              panic_fn,                Target::Fn;
    PanicBoundsCheckFnLangItem,  "panic_bounds_check", panic_bounds_check_fn,   Target::Fn;
    PanicInfoLangItem,           "panic_info",         panic_info,              Target::Struct;
    PanicLocationLangItem,       "panic_location",     panic_location,          Target::Struct;
    PanicImplLangItem,           "panic_impl",         panic_impl,              Target::Fn;
    // Libstd panic entry point. Necessary for const eval to be able to catch it
    BeginPanicFnLangItem,        "begin_panic",        begin_panic_fn,          Target::Fn;

    ExchangeMallocFnLangItem,    "exchange_malloc",    exchange_malloc_fn,      Target::Fn;
    BoxFreeFnLangItem,           "box_free",           box_free_fn,             Target::Fn;
    DropInPlaceFnLangItem,       "drop_in_place",      drop_in_place_fn,        Target::Fn;
    OomLangItem,                 "oom",                oom,                     Target::Fn;
    AllocLayoutLangItem,         "alloc_layout",       alloc_layout,            Target::Struct;

    StartFnLangItem,             "start",              start_fn,                Target::Fn;

    EhPersonalityLangItem,       "eh_personality",     eh_personality,          Target::Fn;
    EhCatchTypeinfoLangItem,     "eh_catch_typeinfo",  eh_catch_typeinfo,       Target::Static;

    OwnedBoxLangItem,            "owned_box",          owned_box,               Target::Struct;

    PhantomDataItem,             "phantom_data",       phantom_data,            Target::Struct;

    ManuallyDropItem,            "manually_drop",      manually_drop,           Target::Struct;

    MaybeUninitLangItem,         "maybe_uninit",       maybe_uninit,            Target::Union;

    // Align offset for stride != 1; must not panic.
    AlignOffsetLangItem,         "align_offset",       align_offset_fn,         Target::Fn;

    TerminationTraitLangItem,    "termination",        termination,             Target::Trait;
}
