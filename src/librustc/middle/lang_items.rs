//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., "Sync", "Send".
//!
//! * Traits that represent operators; e.g., "Add", "Sub", "Index".
//!
//! * Functions called by the compiler itself.

pub use self::LangItem::*;

use crate::hir::def_id::DefId;
use crate::hir::check_attr::Target;
use crate::ty::{self, TyCtxt};
use crate::middle::weak_lang_items;
use crate::util::nodemap::FxHashMap;

use syntax::ast;
use syntax::symbol::{Symbol, sym};
use syntax_pos::Span;
use rustc_macros::HashStable;
use crate::hir::itemlikevisit::ItemLikeVisitor;
use crate::hir;

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
    fn name(self) -> &'static str {
        match self {
            $( $variant => $name, )*
        }
    }
}

#[derive(HashStable)]
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

    /// Returns the kind of closure that `id`, which is one of the `Fn*` traits, corresponds to.
    /// If `id` is not one of the `Fn*` traits, `None` is returned.
    pub fn fn_trait_kind(&self, id: DefId) -> Option<ty::ClosureKind> {
        match Some(id) {
            x if x == self.fn_trait() => Some(ty::ClosureKind::Fn),
            x if x == self.fn_mut_trait() => Some(ty::ClosureKind::FnMut),
            x if x == self.fn_once_trait() => Some(ty::ClosureKind::FnOnce),
            _ => None
        }
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

struct LanguageItemCollector<'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
    /// A mapping from the name of the lang item to its order and the form it must be of.
    item_refs: FxHashMap<&'static str, (usize, Target)>,
}

impl ItemLikeVisitor<'v> for LanguageItemCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let Some((value, span)) = extract(&item.attrs) {
            let actual_target = Target::from_item(item);
            match self.item_refs.get(&*value.as_str()).cloned() {
                // Known lang item with attribute on correct target.
                Some((item_index, expected_target)) if actual_target == expected_target => {
                    let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                    self.collect_item(item_index, def_id);
                },
                // Known lang item with attribute on incorrect target.
                Some((_, expected_target)) => {
                    struct_span_err!(
                        self.tcx.sess, span, E0718,
                        "`{}` language item must be applied to a {}",
                        value, expected_target,
                    ).span_label(
                        span,
                        format!(
                            "attribute should be applied to a {}, not a {}",
                            expected_target, actual_target,
                        ),
                    ).emit();
                },
                // Unknown lang item.
                _ => {
                    struct_span_err!(
                        self.tcx.sess, span, E0522,
                        "definition of an unknown language item: `{}`",
                        value
                    ).span_label(
                        span,
                        format!("definition of unknown language item `{}`", value)
                    ).emit();
                },
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
        // at present, lang items are always items, not trait items
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
        // at present, lang items are always items, not impl items
    }
}

impl LanguageItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LanguageItemCollector<'tcx> {
        let mut item_refs = FxHashMap::default();

        $( item_refs.insert($name, ($variant as usize, $target)); )*

        LanguageItemCollector {
            tcx,
            items: LanguageItems::new(),
            item_refs,
        }
    }

    fn collect_item(&mut self, item_index: usize, item_def_id: DefId) {
        // Check for duplicates.
        if let Some(original_def_id) = self.items.items[item_index] {
            if original_def_id != item_def_id {
                let name = LangItem::from_u32(item_index as u32).unwrap().name();
                let mut err = match self.tcx.hir().span_if_local(item_def_id) {
                    Some(span) => struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0152,
                        "duplicate lang item found: `{}`.",
                        name),
                    None => self.tcx.sess.struct_err(&format!(
                            "duplicate lang item in crate `{}`: `{}`.",
                            self.tcx.crate_name(item_def_id.krate),
                            name)),
                };
                if let Some(span) = self.tcx.hir().span_if_local(original_def_id) {
                    span_note!(&mut err, span, "first defined here.");
                } else {
                    err.note(&format!("first defined in crate `{}`.",
                                      self.tcx.crate_name(original_def_id.krate)));
                }
                err.emit();
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }
}

/// Extract the first `lang = "$name"` out of a list of attributes.
/// The attributes `#[panic_handler]` and `#[alloc_error_handler]`
/// are also extracted out when found.
pub fn extract(attrs: &[ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| Some(match attr {
        _ if attr.check_name(sym::lang) => (attr.value_str()?, attr.span),
        _ if attr.check_name(sym::panic_handler) => (sym::panic_impl, attr.span),
        _ if attr.check_name(sym::alloc_error_handler) => (sym::oom, attr.span),
        _ => return None,
    }))
}

/// Traverse and collect all the lang items in all crates.
pub fn collect<'tcx>(tcx: TyCtxt<'tcx>) -> LanguageItems {
    // Initialize the collector.
    let mut collector = LanguageItemCollector::new(tcx);

    // Collect lang items in other crates.
    for &cnum in tcx.crates().iter() {
        for &(def_id, item_index) in tcx.defined_lang_items(cnum).iter() {
            collector.collect_item(item_index, def_id);
        }
    }

    // Collect lang items in this crate.
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    // Extract out the found lang items.
    let LanguageItemCollector { mut items, .. } = collector;

    // Find all required but not-yet-defined lang items.
    weak_lang_items::check_crate(tcx, &mut items);

    items
}

// End of the macro
    }
}

language_item_table! {
//  Variant name,                Name,                 Method name,             Target;
    CharImplItem,                "char",               char_impl,               Target::Impl;
    StrImplItem,                 "str",                str_impl,                Target::Impl;
    SliceImplItem,               "slice",              slice_impl,              Target::Impl;
    SliceU8ImplItem,             "slice_u8",           slice_u8_impl,           Target::Impl;
    StrAllocImplItem,            "str_alloc",          str_alloc_impl,          Target::Impl;
    SliceAllocImplItem,          "slice_alloc",        slice_alloc_impl,        Target::Impl;
    SliceU8AllocImplItem,        "slice_u8_alloc",     slice_u8_alloc_impl,     Target::Impl;
    ConstPtrImplItem,            "const_ptr",          const_ptr_impl,          Target::Impl;
    MutPtrImplItem,              "mut_ptr",            mut_ptr_impl,            Target::Impl;
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
    CopyTraitLangItem,           "copy",               copy_trait,              Target::Trait;
    CloneTraitLangItem,          "clone",              clone_trait,             Target::Trait;
    SyncTraitLangItem,           "sync",               sync_trait,              Target::Trait;
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

    EqTraitLangItem,             "eq",                 eq_trait,                Target::Trait;
    PartialOrdTraitLangItem,     "partial_ord",        partial_ord_trait,       Target::Trait;
    OrdTraitLangItem,            "ord",                ord_trait,               Target::Trait;

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
    EhUnwindResumeLangItem,      "eh_unwind_resume",   eh_unwind_resume,        Target::Fn;
    MSVCTryFilterLangItem,       "msvc_try_filter",    msvc_try_filter,         Target::Static;

    OwnedBoxLangItem,            "owned_box",          owned_box,               Target::Struct;

    PhantomDataItem,             "phantom_data",       phantom_data,            Target::Struct;

    ManuallyDropItem,            "manually_drop",      manually_drop,           Target::Struct;

    DebugTraitLangItem,          "debug_trait",        debug_trait,             Target::Trait;

    // A lang item for each of the 128-bit operators we can optionally lower.
    I128AddFnLangItem,           "i128_add",           i128_add_fn,             Target::Fn;
    U128AddFnLangItem,           "u128_add",           u128_add_fn,             Target::Fn;
    I128SubFnLangItem,           "i128_sub",           i128_sub_fn,             Target::Fn;
    U128SubFnLangItem,           "u128_sub",           u128_sub_fn,             Target::Fn;
    I128MulFnLangItem,           "i128_mul",           i128_mul_fn,             Target::Fn;
    U128MulFnLangItem,           "u128_mul",           u128_mul_fn,             Target::Fn;
    I128DivFnLangItem,           "i128_div",           i128_div_fn,             Target::Fn;
    U128DivFnLangItem,           "u128_div",           u128_div_fn,             Target::Fn;
    I128RemFnLangItem,           "i128_rem",           i128_rem_fn,             Target::Fn;
    U128RemFnLangItem,           "u128_rem",           u128_rem_fn,             Target::Fn;
    I128ShlFnLangItem,           "i128_shl",           i128_shl_fn,             Target::Fn;
    U128ShlFnLangItem,           "u128_shl",           u128_shl_fn,             Target::Fn;
    I128ShrFnLangItem,           "i128_shr",           i128_shr_fn,             Target::Fn;
    U128ShrFnLangItem,           "u128_shr",           u128_shr_fn,             Target::Fn;
    // And overflow versions for the operators that are checkable.
    // While MIR calls these Checked*, they return (T,bool), not Option<T>.
    I128AddoFnLangItem,          "i128_addo",          i128_addo_fn,            Target::Fn;
    U128AddoFnLangItem,          "u128_addo",          u128_addo_fn,            Target::Fn;
    I128SuboFnLangItem,          "i128_subo",          i128_subo_fn,            Target::Fn;
    U128SuboFnLangItem,          "u128_subo",          u128_subo_fn,            Target::Fn;
    I128MuloFnLangItem,          "i128_mulo",          i128_mulo_fn,            Target::Fn;
    U128MuloFnLangItem,          "u128_mulo",          u128_mulo_fn,            Target::Fn;
    I128ShloFnLangItem,          "i128_shlo",          i128_shlo_fn,            Target::Fn;
    U128ShloFnLangItem,          "u128_shlo",          u128_shlo_fn,            Target::Fn;
    I128ShroFnLangItem,          "i128_shro",          i128_shro_fn,            Target::Fn;
    U128ShroFnLangItem,          "u128_shro",          u128_shro_fn,            Target::Fn;

    // Align offset for stride != 1, must not panic.
    AlignOffsetLangItem,         "align_offset",       align_offset_fn,         Target::Fn;

    TerminationTraitLangItem,    "termination",        termination,             Target::Trait;

    Arc,                         "arc",                arc,                     Target::Struct;
    Rc,                          "rc",                 rc,                      Target::Struct;
}

impl<'tcx> TyCtxt<'tcx> {
    /// Returns the `DefId` for a given `LangItem`.
    /// If not found, fatally abort compilation.
    pub fn require_lang_item(&self, lang_item: LangItem) -> DefId {
        self.lang_items().require(lang_item).unwrap_or_else(|msg| {
            self.sess.fatal(&msg)
        })
    }
}
