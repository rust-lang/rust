use std::cell::RefCell;
use std::iter;
use std::ops::ControlFlow;

use bitflags::bitflags;
use rustc_abi::VariantIdx;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::DiagMessage;
use rustc_hir::def::CtorKind;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{self as hir, AmbigArg};
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Adt, AdtDef, AdtKind, Binder, FnSig, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt,
};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use tracing::debug;

use super::repr_nullable_ptr;
use crate::lints::{ImproperCTypes, ImproperCTypesLayer, UsesPowerAlignment};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    /// (In other words, declarations of items defined in foreign code.)
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe extern "C" {
    ///     static STATIC: String;
    ///     fn some_func(a:String);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in `extern`
    /// blocks are safe and follow certain rules to ensure proper
    /// compatibility with the foreign interfaces. This lint is issued when it
    /// detects a probable mistake in a definition. The lint usually should
    /// provide a description of the issue, along with possibly a hint on how
    /// to resolve it.
    pub(crate) IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint! {
    /// The `improper_c_fn_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
    /// (In other words, functions to be used by foreign code.)
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub extern "C" fn str_type(p: &str) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    pub(crate) IMPROPER_C_FN_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
}

declare_lint! {
    /// The `improper_c_var_definitions` lint detects incorrect use of
    /// [`no_mangle`] and [`export_name`] static variable definitions.
    /// (In other words, static variables accessible by name by foreign code.)
    ///
    /// [`no_mangle`]: https://doc.rust-lang.org/stable/reference/abi.html#the-no_mangle-attribute
    /// [`export_name`]: https://doc.rust-lang.org/stable/reference/abi.html#the-export_name-attribute
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #[unsafe(no_mangle)]
    /// # #[used]
    /// static mut PLUGIN_ABI_MIN_VERSION: &'static str = "0.0.5";
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in
    /// static variables exposed to foreign code are safe and follow
    /// certain rules to ensure proper compatibility with the foreign interfaces.
    /// This lint is issued when it detects a probable mistake in a definition.
    /// The lint usually should provide a description of the issue,
    /// along with possibly a hint on how to resolve it.
    pub(crate) IMPROPER_C_VAR_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign-reachable static variable definitions"
}

declare_lint! {
    /// The `improper_c_callbacks` lint detects incorrect use of
    /// [`extern` function] pointers.
    /// (In other words, function signatures for callbacks.)
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub fn str_emmiter(call_me_back: extern "C" fn(&str)) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    pub(crate) IMPROPER_C_CALLBACKS,
    Warn,
    "proper use of libc types in foreign-code-compatible callbacks"
}

declare_lint! {
    /// The `uses_power_alignment` lint detects specific `repr(C)`
    /// aggregates on AIX.
    /// In its platform C ABI, AIX uses the "power" (as in PowerPC) alignment
    /// rule (detailed in https://www.ibm.com/docs/en/xl-c-and-cpp-aix/16.1?topic=data-using-alignment-modes#alignment),
    /// which can also be set for XLC by `#pragma align(power)` or
    /// `-qalign=power`. Aggregates with a floating-point type as the
    /// recursively first field (as in "at offset 0") modify the layout of
    /// *subsequent* fields of the associated structs to use an alignment value
    /// where the floating-point type is aligned on a 4-byte boundary.
    ///
    /// Effectively, subsequent floating-point fields act as-if they are `repr(packed(4))`. This
    /// would be unsound to do in a `repr(C)` type without all the restrictions that come with
    /// `repr(packed)`. Rust instead chooses a layout that maintains soundness of Rust code, at the
    /// expense of incompatibility with C code.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-powerpc64-ibm-aix)
    /// #[repr(C)]
    /// pub struct Floats {
    ///     a: f64,
    ///     b: u8,
    ///     c: f64,
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: repr(C) does not follow the power alignment rule. This may affect platform C ABI compatibility for this type
    ///  --> <source>:5:3
    ///   |
    /// 5 |   c: f64,
    ///   |   ^^^^^^
    ///   |
    ///   = note: `#[warn(uses_power_alignment)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The power alignment rule specifies that the above struct has the
    /// following alignment:
    ///  - offset_of!(Floats, a) == 0
    ///  - offset_of!(Floats, b) == 8
    ///  - offset_of!(Floats, c) == 12
    ///
    /// However, Rust currently aligns `c` at `offset_of!(Floats, c) == 16`.
    /// Using offset 12 would be unsound since `f64` generally must be 8-aligned on this target.
    /// Thus, a warning is produced for the above struct.
    USES_POWER_ALIGNMENT,
    Warn,
    "Structs do not follow the power alignment rule under repr(C)"
}

declare_lint_pass!(ImproperCTypesLint => [
    IMPROPER_CTYPES,
    IMPROPER_C_FN_DEFINITIONS,
    IMPROPER_C_VAR_DEFINITIONS,
    IMPROPER_C_CALLBACKS,
    USES_POWER_ALIGNMENT,
]);

type Sig<'tcx> = Binder<'tcx, FnSig<'tcx>>;

// FIXME(ctypes): it seems that tests/ui/lint/opaque-ty-ffi-normalization-cycle.rs relies this:
// we consider opaque aliases that normalise to something else to be unsafe.
// ...is it the behaviour we want?
/// a modified version of cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty)
/// so that opaque types prevent normalisation once region erasure occurs
fn erase_and_maybe_normalize<'tcx>(cx: &LateContext<'tcx>, value: Ty<'tcx>) -> Ty<'tcx> {
    if (!value.has_aliases()) || value.has_opaque_types() {
        cx.tcx.erase_regions(value)
    } else {
        cx.tcx.try_normalize_erasing_regions(cx.typing_env(), value).unwrap_or(value)
        // note: the code above ^^^ would only cause a call to the commented code below vvv
        //let value = cx.tcx.erase_regions(value);
        //let mut folder = TryNormalizeAfterErasingRegionsFolder::new(cx.tcx, cx.typing_env());
        //value.try_fold_with(&mut folder).unwrap_or(value)
    }
}

/// Getting the (normalized) type out of a field (for, e.g., an enum variant or a tuple).
#[inline]
fn get_type_from_field<'tcx>(
    cx: &LateContext<'tcx>,
    field: &ty::FieldDef,
    args: GenericArgsRef<'tcx>,
) -> Ty<'tcx> {
    let field_ty = field.ty(cx.tcx, args);
    erase_and_maybe_normalize(cx, field_ty)
}

/// Check a variant of a non-exhaustive enum for improper ctypes.
/// Returns two bools: "we have FFI-unsafety due to non-exhaustive enum" and
/// "we have FFI-unsafety due to a non-exhaustive enum variant".
///
/// We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
/// This includes linting, on a best-effort basis. There are valid additions that are unlikely.
///
/// Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
/// so we don't need the lint to account for it.
/// e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
pub(crate) fn flag_non_exhaustive_variant(
    non_exhaustive_variant_list: bool,
    variant: &ty::VariantDef,
) -> (bool, bool) {
    // non_exhaustive suggests it is possible that someone might break ABI
    // see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
    // so warn on complex enums being used outside their crate
    if non_exhaustive_variant_list {
        // which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)
        if variant_has_complex_ctor(variant) {
            return (true, false);
        }
    }

    if variant.field_list_has_applicable_non_exhaustive() {
        return (false, true);
    }

    (false, false)
}

fn variant_has_complex_ctor(variant: &ty::VariantDef) -> bool {
    // CtorKind::Const means a "unit" ctor
    !matches!(variant.ctor_kind(), Some(CtorKind::Const))
}

/// A way to keep track of what we want to lint for FFI-safety.
/// In other words, the nature of the "original item" being checked, and its relation
/// to FFI boundaries.
#[derive(Clone, Copy, Debug)]
enum CItemKind {
    /// Imported items in an `extern "C"` block (function declarations, static variables) -> IMPROPER_CTYPES
    ImportedExtern,
    /// `extern "C"` function definitions, to be used elsewhere -> IMPROPER_C_FN_DEFINITIONS,
    ExportedFunction,
    /// `no_mangle`/`export_name` static variables, assumed to be used from across an FFI boundary,
    ExportedStatic,
    /// `extern "C"` function pointers -> IMPROPER_C_CALLBACKS,
    Callback,
}

#[derive(Clone, Debug)]
struct FfiUnsafeReason<'tcx> {
    ty: Ty<'tcx>,
    note: DiagMessage,
    help: Option<DiagMessage>,
    inner: Option<Box<FfiUnsafeReason<'tcx>>>,
}

/// A single explanation (out of possibly multiple)
/// telling why a given element is rendered FFI-unsafe.
/// This goes as deep as the 'core cause', but it might be located elsewhere, possibly in a different crate.
/// So, we also track the 'smallest' type in the explanation that appears in the span of the unsafe element.
/// (we call this the 'cause' or the 'local cause' of the unsafety)
#[derive(Clone, Debug)]
struct FfiUnsafeExplanation<'tcx> {
    /// A stack of incrementally "smaller" types, justifications and help messages,
    /// ending with the 'core reason' why something is FFI-unsafe, making everything around it also unsafe.
    reason: Box<FfiUnsafeReason<'tcx>>,
    /// Override the type considered the local cause of the FFI-unsafety.
    /// (e.g.: even if the lint goes into detail as to why a struct used as a function argument
    /// is unsafe, have the first lint line say that the fault lies in the use of said struct.)
    override_cause_ty: Option<Ty<'tcx>>,
}

/// The result describing the safety (or lack thereof) of a given type.
#[derive(Clone, Debug)]
enum FfiResult<'tcx> {
    /// The type is known to be safe.
    FfiSafe,
    /// The type is only a phantom annotation.
    /// (Safe in some contexts, unsafe in others.)
    FfiPhantom(Ty<'tcx>),
    /// The type is not safe.
    /// there might be any number of "explanations" as to why,
    /// each being a stack of "reasons" going from the type
    /// to a core cause of FFI-unsafety.
    FfiUnsafe(Vec<FfiUnsafeExplanation<'tcx>>),
}

impl<'tcx> FfiResult<'tcx> {
    /// Simplified creation of the FfiUnsafe variant for a single unsafety reason.
    fn new_with_reason(ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        Self::FfiUnsafe(vec![FfiUnsafeExplanation {
            override_cause_ty: None,
            reason: Box::new(FfiUnsafeReason { ty, help, note, inner: None }),
        }])
    }

    /// If the FfiUnsafe variant, 'wraps' all reasons,
    /// creating new `FfiUnsafeReason`s, putting the originals as their `inner` fields.
    /// Otherwise, keep unchanged.
    fn wrap_all(self, ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        match self {
            Self::FfiUnsafe(this) => {
                let unsafeties = this
                    .into_iter()
                    .map(|FfiUnsafeExplanation { reason, override_cause_ty }| {
                        let reason = Box::new(FfiUnsafeReason {
                            ty,
                            help: help.clone(),
                            note: note.clone(),
                            inner: Some(reason),
                        });
                        FfiUnsafeExplanation { reason, override_cause_ty }
                    })
                    .collect::<Vec<_>>();
                Self::FfiUnsafe(unsafeties)
            }
            r @ _ => r,
        }
    }
    /// If the FfiPhantom variant, turns it into a FfiUnsafe version.
    /// Otherwise, keep unchanged.
    fn forbid_phantom(self) -> Self {
        match self {
            Self::FfiPhantom(ty) => {
                Self::new_with_reason(ty, fluent::lint_improper_ctypes_only_phantomdata, None)
            }
            _ => self,
        }
    }

    /// Selectively "pluck" some explanations out of a FfiResult::FfiUnsafe,
    /// if the note at their core reason is one in a provided list.
    /// If the FfiResult is not FfiUnsafe, or if no reasons are plucked,
    /// then return FfiSafe.
    fn take_with_core_note(&mut self, notes: &[DiagMessage]) -> Self {
        match self {
            Self::FfiUnsafe(this) => {
                let mut remaining_explanations = vec![];
                std::mem::swap(this, &mut remaining_explanations);
                let mut filtered_explanations = vec![];
                let mut remaining_explanations = remaining_explanations
                    .into_iter()
                    .filter_map(|explanation| {
                        let mut reason = explanation.reason.as_ref();
                        while let Some(ref inner) = reason.inner {
                            reason = inner.as_ref();
                        }
                        let mut does_remain = true;
                        for note_match in notes {
                            if note_match == &reason.note {
                                does_remain = false;
                                break;
                            }
                        }
                        if does_remain {
                            Some(explanation)
                        } else {
                            filtered_explanations.push(explanation);
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                std::mem::swap(this, &mut remaining_explanations);
                if filtered_explanations.len() > 0 {
                    Self::FfiUnsafe(filtered_explanations)
                } else {
                    Self::FfiSafe
                }
            }
            _ => Self::FfiSafe,
        }
    }

    /// Wrap around code that generates FfiResults "from a different cause".
    /// For instance, if we have a repr(C) struct in a function's argument, FFI unsafeties inside the struct
    /// are to be blamed on the struct and not the members.
    /// This is where we use this wrapper, to tell "all FFI-unsafeties in there are caused by this `ty`"
    fn with_overrides(mut self, override_cause_ty: Option<Ty<'tcx>>) -> FfiResult<'tcx> {
        use FfiResult::*;

        if let FfiUnsafe(ref mut explanations) = self {
            explanations.iter_mut().for_each(|explanation| {
                explanation.override_cause_ty = override_cause_ty;
            });
        }
        self
    }
}

impl<'tcx> std::ops::AddAssign<FfiResult<'tcx>> for FfiResult<'tcx> {
    fn add_assign(&mut self, other: Self) {
        // note: we shouldn't really encounter FfiPhantoms here, they should be dealt with beforehand
        // still, this function deals with them in a reasonable way, I think

        match (self, other) {
            (Self::FfiUnsafe(myself), Self::FfiUnsafe(mut other_reasons)) => {
                myself.append(&mut other_reasons);
            }
            (Self::FfiUnsafe(_), _) => {
                // nothing to do
            }
            (myself, other @ Self::FfiUnsafe(_)) => {
                *myself = other;
            }
            (Self::FfiPhantom(ty1), Self::FfiPhantom(ty2)) => {
                debug!("whoops, both FfiPhantom: self({:?}) += other({:?})", ty1, ty2);
            }
            (myself @ Self::FfiSafe, other @ Self::FfiPhantom(_)) => {
                *myself = other;
            }
            (_, Self::FfiSafe) => {
                // nothing to do
            }
        }
    }
}
impl<'tcx> std::ops::Add<FfiResult<'tcx>> for FfiResult<'tcx> {
    type Output = FfiResult<'tcx>;
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

/// The result when a type has been checked but perhaps not completely. `None` indicates that
/// FFI safety/unsafety has not yet been determined, `Some(res)` indicates that the safety/unsafety
/// in the `FfiResult` is final.
type PartialFfiResult<'tcx> = Option<FfiResult<'tcx>>;

/// The type of an indirection (the way in which it points to its pointee).
#[derive(Clone, Copy)]
enum IndirectionType {
    /// Box (valid non-null pointer, owns pointee).
    Box,
    /// Ref (valid non-null pointer, borrows pointee).
    Ref,
    /// Raw pointer (not necessarily non-null or valid. no info on ownership).
    RawPtr,
}

/// The different ways a given type can have/not have a fixed size
#[derive(Clone, Copy)]
enum TypeSizedness {
    /// Type of definite size (pointers are C-compatible).
    Definite,
    /// Unsized type because it includes an opaque/foreign type (pointers are C-compatible).
    UnsizedWithExternType,
    /// Unsized type for other reasons (slice, string, dyn Trait, closure, ...) (pointers are not C-compatible).
    UnsizedWithMetadata,
    /// Not known, usually for placeholder types (Self in non-impl trait functions, type parameters, aliases, the like).
    NotYetKnown,
}

/// Determine if a type is sized or not, and whether it affects references/pointers/boxes to it.
fn get_type_sizedness<'tcx, 'a>(cx: &'a LateContext<'tcx>, ty: Ty<'tcx>) -> TypeSizedness {
    let tcx = cx.tcx;

    // note that sizedness is unrelated to inhabitedness
    if ty.is_sized(tcx, cx.typing_env()) {
        TypeSizedness::Definite
    } else {
        // the overall type is !Sized or ?Sized
        match ty.kind() {
            ty::Slice(_) => TypeSizedness::UnsizedWithMetadata,
            ty::Str => TypeSizedness::UnsizedWithMetadata,
            ty::Dynamic(..) => TypeSizedness::UnsizedWithMetadata,
            ty::Foreign(..) => TypeSizedness::UnsizedWithExternType,
            ty::Adt(def, args) => {
                // for now assume: boxes and phantoms don't mess with this
                match def.adt_kind() {
                    AdtKind::Union | AdtKind::Enum => {
                        bug!("unions and enums are necessarily sized")
                    }
                    AdtKind::Struct => {
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return TypeSizedness::UnsizedWithMetadata;
                        }

                        // note: non-exhaustive structs from other crates are not assumed to be ?Sized
                        // for the purpose of sizedness, it seems we are allowed to look at its current contents.

                        if def.non_enum_variant().fields.is_empty() {
                            bug!("an empty struct is necessarily sized");
                        }

                        let variant = def.non_enum_variant();

                        // only the last field may be !Sized (or ?Sized in the case of type params)
                        let last_field = match (&variant.fields).iter().last(){
                            Some(last_field) => last_field,
                            // even nonexhaustive-empty structs from another crate are considered Sized
                            // (eventhough one could add a !Sized field to them)
                            None => bug!("Empty struct should be Sized, right?"), //
                        };
                        let field_ty = get_type_from_field(cx, last_field, args);
                        match get_type_sizedness(cx, field_ty) {
                            s @ (TypeSizedness::UnsizedWithMetadata
                            | TypeSizedness::UnsizedWithExternType
                            | TypeSizedness::NotYetKnown) => s,
                            TypeSizedness::Definite => {
                                bug!("failed to find the reason why struct `{:?}` is unsized", ty)
                            }
                        }
                    }
                }
            }
            ty::Tuple(tuple) => {
                // only the last field may be !Sized (or ?Sized in the case of type params)
                let item_ty: Ty<'tcx> = match tuple.last() {
                    Some(item_ty) => *item_ty,
                    None => bug!("Empty tuple (AKA unit type) should be Sized, right?"),
                };
                let item_ty = erase_and_maybe_normalize(cx, item_ty);
                match get_type_sizedness(cx, item_ty) {
                    s @ (TypeSizedness::UnsizedWithMetadata
                    | TypeSizedness::UnsizedWithExternType
                    | TypeSizedness::NotYetKnown) => s,
                    TypeSizedness::Definite => {
                        bug!("failed to find the reason why tuple `{:?}` is unsized", ty)
                    }
                }
            }

            ty_kind @ (ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Array(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Pat(..) // these are (for now) numeric types with a range-based restriction
            ) => {
                // those types are all sized, right?
                bug!(
                    "This ty_kind (`{:?}`) should be sized, yet we are in a branch of code that deals with unsized types.",
                    ty_kind,
                )
            }

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach ty::Alias(ty::Opaque) here.
            ty::Param(..) | ty::Alias(ty::Opaque | ty::Projection | ty::Inherent, ..) => {
                return TypeSizedness::NotYetKnown;
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Alias(ty::Free, ..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }
}

bitflags! {
    #[allow(non_snake_case)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct VisitorState: u8 {
        /// For use in (externally-linked) static variables.
        const STATIC = 0b000001;
        /// For use in functions in general.
        const FUNC = 0b000010;
        /// For variables in function returns (implicitly: not for static variables).
        const FN_RETURN = 0b000100;
        /// For variables in functions/variables which are defined in rust.
        const DEFINED = 0b001000;
        /// For times where we are only defining the type of something
        /// (struct/enum/union definitions, FnPtrs).
        const THEORETICAL = 0b010000;

    }
}

impl VisitorState {
    // The values that can be set.

    // FIXME(const): ideally .bits() and ::from_bits() wouldn't be necessary.
    // Unfortunately, "cannot call non-const operator in constants" (bitwise or?).
    const NONE: Self = Self::empty();
    const STATIC_TY: Self = Self::STATIC;
    const EXPORTED_STATIC_TY: Self =
        Self::from_bits(Self::STATIC.bits() | Self::DEFINED.bits()).unwrap();
    const ARGUMENT_TY_IN_DEFINITION: Self =
        Self::from_bits(Self::FUNC.bits() | Self::DEFINED.bits()).unwrap();
    const RETURN_TY_IN_DEFINITION: Self =
        Self::from_bits(Self::FUNC.bits() | Self::FN_RETURN.bits() | Self::DEFINED.bits()).unwrap();
    const ARGUMENT_TY_IN_DECLARATION: Self = Self::FUNC;
    const RETURN_TY_IN_DECLARATION: Self =
        Self::from_bits(Self::FUNC.bits() | Self::FN_RETURN.bits()).unwrap();
    const ARGUMENT_TY_IN_FNPTR: Self =
        Self::from_bits(Self::FUNC.bits() | Self::THEORETICAL.bits()).unwrap();
    const RETURN_TY_IN_FNPTR: Self =
        Self::from_bits(Self::FUNC.bits() | Self::THEORETICAL.bits() | Self::FN_RETURN.bits())
            .unwrap();

    /// Get the proper visitor state for a given function's arguments.
    fn argument_from_fnmode(fn_mode: CItemKind) -> Self {
        match fn_mode {
            CItemKind::ExportedFunction => VisitorState::ARGUMENT_TY_IN_DEFINITION,
            CItemKind::ImportedExtern => VisitorState::ARGUMENT_TY_IN_DECLARATION,
            // we could also deal with CItemKind::Callback,
            // but we bake an assumption from this function's call sites here.
            _ => bug!("cannot be called with CItemKind::{:?}", fn_mode),
        }
    }

    /// Get the proper visitor state for a given function's return type.
    fn return_from_fnmode(fn_mode: CItemKind) -> Self {
        match fn_mode {
            CItemKind::ExportedFunction => VisitorState::RETURN_TY_IN_DEFINITION,
            CItemKind::ImportedExtern => VisitorState::RETURN_TY_IN_DECLARATION,
            // we could also deal with CItemKind::Callback,
            // but we bake an assumption from this function's call sites here.
            _ => bug!("cannot be called with CItemKind::{:?}", fn_mode),
        }
    }

    /// Whether the type is used in a static variable.
    fn is_in_static(self) -> bool {
        let ret = self.contains(Self::STATIC);
        if ret {
            assert!(!self.contains(Self::FUNC));
        }
        ret
    }

    /// Whether the type is used in a function.
    fn is_in_function(self) -> bool {
        let ret = self.contains(Self::FUNC);
        if ret {
            debug_assert!(!self.contains(Self::STATIC));
        }
        ret
    }
    /// Whether the type is used (directly or not) in a function, in return position.
    fn is_in_function_return(self) -> bool {
        let ret = self.contains(Self::FN_RETURN);
        if ret {
            debug_assert!(self.is_in_function());
        }
        ret
    }
    /// Whether the type is used (directly or not) in a defined function.
    /// In other words, whether or not we allow non-FFI-safe types behind a C pointer,
    /// to be treated as an opaque type on the other side of the FFI boundary.
    fn is_in_defined_function(self) -> bool {
        self.contains(Self::DEFINED) && self.is_in_function()
    }

    /// Whether we can expect type parameters and co in a given type.
    fn can_expect_ty_params(self) -> bool {
        // rust-defined functions, as well as FnPtrs
        self.contains(Self::THEORETICAL) || self.is_in_defined_function()
    }
}

/// Visitor used to recursively traverse MIR types and evaluate FFI-safety.
/// It uses ``check_*`` methods as entrypoints to be called elsewhere,
/// and ``visit_*`` methods to recurse.
struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// To prevent problems with recursive types,
    /// add a types-in-check cache and a depth counter.
    recursion_limiter: RefCell<(FxHashSet<Ty<'tcx>>, usize)>,
}

/// Structure similar to a mutex guard, allocated for each type in-check
/// to let the ImproperCTypesVisitor know the current depth of the checking process.
struct ImproperCTypesVisitorDepthGuard<'a, 'tcx, 'v>(&'v ImproperCTypesVisitor<'a, 'tcx>);

impl<'a, 'tcx, 'v> Drop for ImproperCTypesVisitorDepthGuard<'a, 'tcx, 'v> {
    fn drop(&mut self) {
        let mut limiter_guard = self.0.recursion_limiter.borrow_mut();
        let (_, ref mut depth) = *limiter_guard;
        *depth -= 1;
    }
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self { cx, recursion_limiter: RefCell::new((FxHashSet::default(), 0)) }
    }

    /// Protect against infinite recursion, for example
    /// `struct S(*mut S);`, or issue #130310.
    fn can_enter_type<'v>(
        &'v self,
        ty: Ty<'tcx>,
    ) -> Result<ImproperCTypesVisitorDepthGuard<'a, 'tcx, 'v>, FfiResult<'tcx>> {
        // panic unlikely: this non-recursive function is the only place that
        // borrows the refcell, outside of ImproperCTypesVisitorDepthGuard::drop()
        let mut limiter_guard = self.recursion_limiter.borrow_mut();
        let (ref mut cache, ref mut depth) = *limiter_guard;
        if (!cache.insert(ty)) || *depth >= 1024 {
            Err(FfiResult::FfiSafe)
        } else {
            *depth += 1;
            Ok(ImproperCTypesVisitorDepthGuard(self))
        }
    }

    /// Checks whether an `extern "ABI" fn` function pointer is indeed FFI-safe to call.
    fn visit_fnptr(
        &self,
        _state: VisitorState,
        _outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
        sig: Sig<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        debug_assert!(!sig.abi().is_rustic_abi());

        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);

        let mut all_ffires = FfiSafe;

        for arg in sig.inputs() {
            let ffi_res = self.visit_type(VisitorState::ARGUMENT_TY_IN_FNPTR, Some(ty), *arg);
            all_ffires += ffi_res.forbid_phantom().wrap_all(
                ty,
                fluent::lint_improper_ctypes_fnptr_indirect_reason,
                None,
            );
        }

        let ret_ty = sig.output();

        let ffi_res = self.visit_type(VisitorState::RETURN_TY_IN_FNPTR, Some(ty), ret_ty);
        all_ffires += ffi_res.forbid_phantom().wrap_all(
            ty,
            fluent::lint_improper_ctypes_fnptr_indirect_reason,
            None,
        );
        all_ffires
    }

    /// Checks whether an uninhabited type (one without valid values) is safe-ish to have here.
    fn visit_uninhabited(&self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        if state.is_in_function_return() {
            FfiResult::FfiSafe
        } else {
            let desc = match ty.kind() {
                ty::Adt(..) => fluent::lint_improper_ctypes_uninhabited_enum,
                ty::Never => fluent::lint_improper_ctypes_uninhabited_never,
                r @ _ => bug!("unexpected ty_kind in uninhabited type handling: {:?}", r),
            };
            FfiResult::new_with_reason(ty, desc, None)
        }
    }

    /// Checks if a simple numeric (int, float) type has an actual portable definition
    /// for the compile target.
    fn visit_numeric(&self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        // FIXME(ctypes): for now, this is very incomplete, and seems to assume a x86_64 target
        match ty.kind() {
            // note: before rust 1.77, 128-bit ints were not FFI-safe on x86_64
            // ...they probably are still unsafe on i686 and other x86_32 architectures
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => FfiResult::FfiSafe,

            ty::Char => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_char_reason,
                Some(fluent::lint_improper_ctypes_char_help),
            ),
            _ => bug!("visit_numeric is to be called with numeric (char, int, float) types"),
        }
    }

    /// Return the right help for Cstring and Cstr-linked unsafety.
    fn visit_cstr(&self, outer_ty: Option<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        debug_assert!(matches!(ty.kind(), ty::Adt(def, _)
            if matches!(
                self.cx.tcx.get_diagnostic_name(def.did()),
                Some(sym::cstring_type | sym::cstr_type)
            )
        ));

        let help = if let Some(outer_ty) = outer_ty {
            match outer_ty.kind() {
                ty::Ref(..) | ty::RawPtr(..) => {
                    if outer_ty.is_mutable_ptr() {
                        fluent::lint_improper_ctypes_cstr_help_mut
                    } else {
                        fluent::lint_improper_ctypes_cstr_help_const
                    }
                }
                ty::Adt(..) if outer_ty.boxed_ty().is_some() => {
                    fluent::lint_improper_ctypes_cstr_help_owned
                }
                _ => fluent::lint_improper_ctypes_cstr_help_unknown,
            }
        } else {
            fluent::lint_improper_ctypes_cstr_help_owned
        };

        FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_cstr_reason, Some(help))
    }

    /// Checks if the given indirection (box,ref,pointer) is "ffi-safe".
    fn visit_indirection(
        &self,
        state: VisitorState,
        ty: Ty<'tcx>,
        inner_ty: Ty<'tcx>,
        indirection_type: IndirectionType,
    ) -> FfiResult<'tcx> {
        let tcx = self.cx.tcx;

        if let ty::Adt(def, _) = inner_ty.kind() {
            if let Some(diag_name @ (sym::cstring_type | sym::cstr_type)) =
                tcx.get_diagnostic_name(def.did())
            {
                // we have better error messages when checking for C-strings directly
                let mut cstr_res = self.visit_cstr(Some(ty), inner_ty); // always unsafe with one depth-one reason.

                // Cstr pointer have metadata, CString is Sized
                if diag_name == sym::cstr_type {
                    // we need to override the "type" part of `cstr_res`'s only FfiResultReason
                    // so it says that it's the use of the indirection that is unsafe
                    match cstr_res {
                        FfiResult::FfiUnsafe(ref mut reasons) => {
                            reasons.first_mut().unwrap().reason.ty = ty;
                        }
                        _ => unreachable!(),
                    }
                    let note = match indirection_type {
                        IndirectionType::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                        IndirectionType::Ref => fluent::lint_improper_ctypes_unsized_ref,
                        IndirectionType::Box => fluent::lint_improper_ctypes_unsized_box,
                    };
                    return cstr_res.wrap_all(ty, note, None);
                } else {
                    return cstr_res;
                }
            }
        }

        // there are three remaining concerns with the pointer:
        // - is the pointer compatible with a C pointer in the first place? (if not, only send that error message)
        // - is the pointee FFI-safe? (it might not matter, see mere lines below)
        // - does the pointer type contain a non-zero assumption, but has a value given by non-rust code?
        // this block deals with the first two.
        match get_type_sizedness(self.cx, inner_ty) {
            TypeSizedness::UnsizedWithExternType | TypeSizedness::Definite => {
                // FIXME(ctypes):
                // for now, we consider this to be safe even in the case of a FFI-unsafe pointee
                // this is technically only safe if the pointer is never dereferenced on the non-rust
                // side of the FFI boundary, i.e. if the type is to be treated as opaque
                // there are techniques to flag those pointees as opaque, but not always, so we can only enforce this
                // in some cases.
                FfiResult::FfiSafe
            }
            TypeSizedness::NotYetKnown => {
                // types with sizedness NotYetKnown:
                // - Type params (with `variable: impl Trait` shorthand or not)
                //   (function definitions only, let's see how this interacts with monomorphisation)
                // - Self in trait functions/methods
                // - Opaque return types
                //   (always FFI-unsafe)
                // - non-exhaustive structs/enums/unions from other crates
                //   (always FFI-unsafe)
                // (for the three first, this is unless there is a `+Sized` bound involved)

                // whether they are FFI-safe or not does not depend on the indirections involved (&Self, &T, Box<impl Trait>),
                // so let's not wrap the current context around a potential FfiUnsafe type param.
                self.visit_type(state, Some(ty), inner_ty)
            }
            TypeSizedness::UnsizedWithMetadata => {
                let help = match inner_ty.kind() {
                    ty::Str => Some(fluent::lint_improper_ctypes_str_help),
                    ty::Slice(_) => Some(fluent::lint_improper_ctypes_slice_help),
                    _ => None,
                };
                let reason = match indirection_type {
                    IndirectionType::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                    IndirectionType::Ref => fluent::lint_improper_ctypes_unsized_ref,
                    IndirectionType::Box => fluent::lint_improper_ctypes_unsized_box,
                };
                return FfiResult::new_with_reason(ty, reason, help);
            }
        }
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn visit_variant_fields(
        &self,
        state: VisitorState,
        ty: Ty<'tcx>,
        def: AdtDef<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let mut ffires_accumulator = FfiSafe;

        let (transparent_with_all_zst_fields, field_list) =
            if !matches!(def.adt_kind(), AdtKind::Enum) && def.repr().transparent() {
                // determine if there is 0 or 1 non-1ZST field, and which it is.
                // (note: for enums, "transparent" means 1-variant)
                if ty.is_privately_uninhabited(self.cx.tcx, self.cx.typing_env()) {
                    // let's consider transparent structs to be maybe unsafe if uninhabited,
                    // even if that is because of fields otherwise ignored in FFI-safety checks
                    // FIXME(ctypes): and also maybe this should be "!is_inhabited_from" but from where?
                    ffires_accumulator += variant
                        .fields
                        .iter()
                        .map(|field| {
                            let field_ty = get_type_from_field(self.cx, field, args);
                            let mut field_res = self.visit_type(state, Some(ty), field_ty);
                            field_res.take_with_core_note(&[
                                fluent::lint_improper_ctypes_uninhabited_enum,
                                fluent::lint_improper_ctypes_uninhabited_never,
                            ])
                        })
                        .reduce(|r1, r2| r1 + r2)
                        .unwrap() // if uninhabited, then >0 fields
                }
                if let Some(field) = super::transparent_newtype_field(self.cx.tcx, variant) {
                    // Transparent newtypes have at most one non-ZST field which needs to be checked later
                    (false, vec![field])
                } else {
                    // ..or have only ZST fields, which is FFI-unsafe (unless those fields are all
                    // `PhantomData`).
                    (true, variant.fields.iter().collect::<Vec<_>>())
                }
            } else {
                (false, variant.fields.iter().collect::<Vec<_>>())
            };

        // We can't completely trust `repr(C)` markings, so make sure the fields are actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        let mut fields_ok_list = vec![true; field_list.len()];

        for (field_i, field) in field_list.into_iter().enumerate() {
            let field_ty = get_type_from_field(self.cx, field, args);
            let ffi_res = self.visit_type(state, Some(ty), field_ty);

            // checking that this is not an FfiUnsafe due to an unit type:
            // visit_type should be smart enough to not consider it unsafe if called from another ADT
            #[cfg(debug_assertions)]
            if let FfiUnsafe(ref reasons) = ffi_res {
                if let (1, Some(FfiUnsafeExplanation { reason, .. })) =
                    (reasons.len(), reasons.first())
                {
                    let FfiUnsafeReason { ty, .. } = reason.as_ref();
                    debug_assert!(!ty.is_unit());
                }
            }

            all_phantom &= match ffi_res {
                FfiPhantom(..) => true,
                FfiSafe => false,
                r @ FfiUnsafe { .. } => {
                    fields_ok_list[field_i] = false;
                    ffires_accumulator += r;
                    false
                }
            }
        }

        // if we have bad fields, also report a possible transparent_with_all_zst_fields
        // (if this combination is somehow possible)
        // otherwise, having all fields be phantoms
        // takes priority over transparent_with_all_zst_fields
        if let FfiUnsafe(explanations) = ffires_accumulator {
            // we assume the repr() of this ADT is either non-packed C or transparent.
            debug_assert!(def.repr().c() || def.repr().transparent() || def.repr().int.is_some());

            if def.repr().transparent() || matches!(def.adt_kind(), AdtKind::Enum) {
                let field_ffires = FfiUnsafe(explanations).wrap_all(
                    ty,
                    fluent::lint_improper_ctypes_struct_dueto,
                    None,
                );
                if transparent_with_all_zst_fields {
                    field_ffires
                        + FfiResult::new_with_reason(
                            ty,
                            fluent::lint_improper_ctypes_struct_zst,
                            None,
                        )
                } else {
                    field_ffires
                }
            } else {
                // since we have a repr(C) struct/union, there's a chance that we have some unsafe fields,
                // but also exactly one non-1ZST field that is FFI-safe:
                // we want to suggest repr(transparent) here.
                // (FIXME(ctypes): confirm that this makes sense for unions once #60405 / RFC2645 stabilises)
                let non_1zst_fields = super::map_non_1zst_fields(self.cx.tcx, variant);
                let (last_non_1zst, non_1zst_count) = non_1zst_fields.into_iter().enumerate().fold(
                    (None, 0_usize),
                    |(prev_nz, count), (field_i, is_nz)| {
                        if is_nz { (Some(field_i), count + 1) } else { (prev_nz, count) }
                    },
                );
                let help = if non_1zst_count == 1
                    && last_non_1zst.map(|field_i| fields_ok_list[field_i]) == Some(true)
                {
                    match def.adt_kind() {
                        AdtKind::Struct => {
                            Some(fluent::lint_improper_ctypes_struct_consider_transparent)
                        }
                        AdtKind::Union => {
                            Some(fluent::lint_improper_ctypes_union_consider_transparent)
                        }
                        AdtKind::Enum => bug!("cannot suggest an enum to be repr(transparent)"),
                    }
                } else {
                    None
                };

                FfiUnsafe(explanations).wrap_all(
                    ty,
                    fluent::lint_improper_ctypes_struct_dueto,
                    help,
                )
            }
        } else if all_phantom {
            FfiPhantom(ty)
        } else if transparent_with_all_zst_fields {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_struct_zst, None)
        } else {
            FfiSafe
        }
    }

    fn visit_struct_or_union(
        &self,
        state: VisitorState,
        ty: Ty<'tcx>,
        def: AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        debug_assert!(matches!(def.adt_kind(), AdtKind::Struct | AdtKind::Union));

        if !def.repr().c() && !def.repr().transparent() {
            return FfiResult::new_with_reason(
                ty,
                if def.is_struct() {
                    fluent::lint_improper_ctypes_struct_layout_reason
                } else {
                    fluent::lint_improper_ctypes_union_layout_reason
                },
                if def.is_struct() {
                    Some(fluent::lint_improper_ctypes_struct_layout_help)
                } else {
                    // (FIXME(ctypes): confirm that this makes sense for unions once #60405 / RFC2645 stabilises)
                    Some(fluent::lint_improper_ctypes_union_layout_help)
                },
            );
        }

        if def.non_enum_variant().field_list_has_applicable_non_exhaustive() {
            return FfiResult::new_with_reason(
                ty,
                if def.is_struct() {
                    fluent::lint_improper_ctypes_struct_non_exhaustive
                } else {
                    fluent::lint_improper_ctypes_union_non_exhaustive
                },
                None,
            );
        }

        let ffires = if def.non_enum_variant().fields.is_empty() {
            FfiResult::new_with_reason(
                ty,
                if def.is_struct() {
                    fluent::lint_improper_ctypes_struct_fieldless_reason
                } else {
                    fluent::lint_improper_ctypes_union_fieldless_reason
                },
                if def.is_struct() {
                    Some(fluent::lint_improper_ctypes_struct_fieldless_help)
                } else {
                    Some(fluent::lint_improper_ctypes_union_fieldless_help)
                },
            )
        } else {
            self.visit_variant_fields(state, ty, def, def.non_enum_variant(), args)
        };

        // from now on in the function, we lint the actual insides of the struct/union: if something is wrong,
        // then the "fault" comes from inside the struct itself.
        // even if we add more details to the lint, the initial line must specify that the FFI-unsafety is because of the struct
        // - if the struct is from the same crate, there is another warning on its definition anyway
        //   (unless it's about Boxes and references without Option<_>
        //    which is partly why we keep the details as to why that struct is FFI-unsafe)
        // - if the struct is from another crate, then there's not much that can be done anyways
        //
        // this enum is visited in the middle of another lint,
        // so we override the "cause type" of the lint
        ffires.with_overrides(Some(ty))
    }

    fn visit_enum(
        &self,
        state: VisitorState,
        ty: Ty<'tcx>,
        def: AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        debug_assert!(matches!(def.adt_kind(), AdtKind::Enum));
        use FfiResult::*;

        if def.variants().is_empty() {
            // Empty enums are implicitly handled as the never type:
            return self.visit_uninhabited(state, ty);
        }
        // Check for a repr() attribute to specify the size of the
        // discriminant.
        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none() {
            // Special-case types like `Option<extern fn()>` and `Result<extern fn(), ()>`
            if let Some(inner_ty) = repr_nullable_ptr(self.cx.tcx, self.cx.typing_env(), ty) {
                return self.visit_type(state, Some(ty), inner_ty);
            }

            return FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_enum_repr_reason,
                Some(fluent::lint_improper_ctypes_enum_repr_help),
            );
        }

        // FIXME(ctypes): connect `def.repr().int` to visit_numeric
        // (for now it's OK, `repr(char)` doesn't exist and visit_numeric doesn't warn on anything else)

        let non_exhaustive = def.variant_list_has_applicable_non_exhaustive();
        // Check the contained variants.

        let (mut nonexhaustive_flag, mut nonexhaustive_variant_flag) = (false, false);
        def.variants().iter().for_each(|variant| {
            let (nonex_enum, nonex_var) = flag_non_exhaustive_variant(non_exhaustive, variant);
            nonexhaustive_flag |= nonex_enum;
            nonexhaustive_variant_flag |= nonex_var;
        });

        // "nonexhaustive" lints only happen outside of the crate defining the enum, so no CItemKind override
        // (meaning: the fault lies in the function call, not the enum)
        if nonexhaustive_flag {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_non_exhaustive, None)
        } else if nonexhaustive_variant_flag {
            FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_non_exhaustive_variant,
                None,
            )
        } else {
            // small caveat to checking the variants: we authorise up to n-1 invariants
            // to be unsafe because uninhabited.
            // so for now let's isolate those unsafeties
            let mut variants_uninhabited_ffires = vec![FfiSafe; def.variants().len()];

            let mut ffires = def
                .variants()
                .iter()
                .enumerate()
                .map(|(variant_i, variant)| {
                    let mut variant_res = self.visit_variant_fields(state, ty, def, variant, args);
                    variants_uninhabited_ffires[variant_i] = variant_res.take_with_core_note(&[
                        fluent::lint_improper_ctypes_uninhabited_enum,
                        fluent::lint_improper_ctypes_uninhabited_never,
                    ]);
                    // FIXME(ctypes): check that enums allow any (up to all) variants to be phantoms?
                    // (previous code says no, but I don't know why? the problem with phantoms is that they're ZSTs, right?)
                    variant_res.forbid_phantom()
                })
                .reduce(|r1, r2| r1 + r2)
                .unwrap(); // always at least one variant if we hit this branch

            if variants_uninhabited_ffires.iter().all(|res| matches!(res, FfiUnsafe(..))) {
                // if the enum is uninhabited, because all its variants are uninhabited
                ffires += variants_uninhabited_ffires.into_iter().reduce(|r1, r2| r1 + r2).unwrap();
            }

            // this enum is visited in the middle of another lint,
            // so we override the "cause type" of the lint
            // (for more detail, see comment in ``visit_struct_union`` before its call to ``ffires.with_overrides``)
            ffires.with_overrides(Some(ty))
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn visit_type(
        &self,
        state: VisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let _depth_guard = match self.can_enter_type(ty) {
            Ok(guard) => guard,
            Err(ffi_res) => return ffi_res,
        };
        let tcx = self.cx.tcx;

        match *ty.kind() {
            ty::Adt(def, args) => {
                if let Some(inner_ty) = ty.boxed_ty() {
                    return self.visit_indirection(state, ty, inner_ty, IndirectionType::Box);
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        // I thought CStr (not CString) here could only be reached in non-compiling code:
                        //   - not using an indirection would cause a compile error (this lint *currently* seems to not get triggered on such non-compiling code)
                        //   - and using one would cause the lint to catch on the indirection before reaching its pointee
                        // but function *pointers* don't seem to have the same no-unsized-parameters requirement to compile
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return self.visit_cstr(outer_ty, ty);
                        }
                        self.visit_struct_or_union(state, ty, def, args)
                    }
                    AdtKind::Enum => self.visit_enum(state, ty, def, args),
                }
            }

            // It's just extra invariants on the type that you need to uphold,
            // but only the base type is relevant for being representable in FFI.
            ty::Pat(pat_ty, _) => {
                //this lint was written when pattern types could only be integers constrained to ranges
                self.visit_numeric(pat_ty)
            }

            // types which likely have a stable representation, depending on the target architecture
            ty::Char | ty::Int(..) | ty::Uint(..) | ty::Float(..) => self.visit_numeric(ty),

            // Primitive types with a stable representation.
            ty::Bool => FfiSafe,

            ty::Slice(inner_ty) => {
                // ty::Slice is used for !Sized arrays, since they are the pointee for actual slices
                let slice_is_actually_array = match outer_ty.map(|ty| ty.kind()) {
                    None => state.is_in_static(),
                    // this should have been caught a layer up, in visit_indirection
                    Some(ty::Ref(..) | ty::RawPtr(..)) => false,
                    Some(ty::Adt(..)) => ty.boxed_ty().is_none(),
                    Some(ty::Tuple(..)) => true,
                    Some(ty::FnPtr(..)) => false,
                    // this is supposed to cause a compile error that prevents this lint
                    // from being reached, but oh well
                    Some(ty::Array(..) | ty::Slice(_)) => true,
                    Some(ty @ _) => bug!("unexpected ty_kind around a slice: {:?}", ty),
                };
                if slice_is_actually_array {
                    self.visit_type(state, Some(ty), inner_ty)
                } else {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_slice_reason,
                        Some(fluent::lint_improper_ctypes_slice_help),
                    )
                }
            }

            ty::Dynamic(..) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_dyn, None)
            }

            ty::Str => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_str_reason,
                Some(fluent::lint_improper_ctypes_str_help),
            ),

            ty::Tuple(tuple) => {
                let empty_and_safe = if tuple.is_empty() {
                    match outer_ty.map(|ty| ty.kind()) {
                        // C functions can return void
                        None | Some(ty::FnPtr(..)) => state.is_in_function_return(),
                        // `()` fields are FFI-safe!
                        Some(ty::Adt(..)) => true,
                        Some(ty::RawPtr(..)) => true,
                        // most of those are not even reachable,
                        // but let's not worry about checking that here
                        _ => false,
                    }
                } else {
                    false
                };

                if empty_and_safe {
                    FfiSafe
                } else {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_tuple_reason,
                        Some(fluent::lint_improper_ctypes_tuple_help),
                    )
                }
            }

            ty::RawPtr(ty, _)
                if match ty.kind() {
                    ty::Tuple(tuple) => tuple.is_empty(),
                    _ => false,
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(inner_ty, _) => {
                return self.visit_indirection(state, ty, inner_ty, IndirectionType::RawPtr);
            }
            ty::Ref(_, inner_ty, _) => {
                return self.visit_indirection(state, ty, inner_ty, IndirectionType::Ref);
            }

            ty::Array(inner_ty, _) => {
                if state.is_in_function()
                    && matches!(outer_ty.map(|ty| ty.kind()), None | Some(ty::FnPtr(..)))
                {
                    // C doesn't really support passing arrays by value - the only way to pass an array by value
                    // is through a struct.
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_array_reason,
                        Some(fluent::lint_improper_ctypes_array_help),
                    )
                } else {
                    // let's allow phantoms to go through,
                    // since an array of 1-ZSTs is also a 1-ZST
                    self.visit_type(state, Some(ty), inner_ty)
                }
            }

            // fnptrs are a special case, they always need to be treated as
            // "the element rendered unsafe" because their unsafety doesn't affect
            // their surroundings, and their type is often declared inline
            // as a result, don't go into them when scanning for the safety of something else
            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if sig.abi().is_rustic_abi() {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_fnptr_reason,
                        Some(fluent::lint_improper_ctypes_fnptr_help),
                    )
                } else {
                    FfiSafe
                }
            }

            ty::Foreign(..) => FfiSafe,

            ty::Never => self.visit_uninhabited(state, ty),

            // This is only half of the checking-for-opaque-aliases story:
            // since they are liable to vanish on normalisation, we need a specific to find them through
            // other aliases, which is called in the next branch of this `match ty.kind()` statement
            ty::Alias(ty::Opaque, ..) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None)
            }

            // `extern "C" fn` function definitions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            // function pointers can do the same
            //
            // however, these ty_kind:s can also be encountered because the type isn't normalized yet.
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent | ty::Free, ..) => {
                if ty.has_opaque_types() {
                    // FIXME(ctypes): this is suboptimal because we give up
                    // on reporting anything *else* than the opaque part of the type
                    // but this is better than not reporting anything, or crashing
                    self.visit_for_opaque_ty(ty).unwrap()
                } else {
                    // in theory, thanks to erase_and_maybe_normalize,
                    // normalisation has already occurred
                    debug_assert_eq!(
                        self.cx
                            .tcx
                            .try_normalize_erasing_regions(self.cx.typing_env(), ty,)
                            .unwrap_or(ty),
                        ty,
                    );

                    if matches!(
                        ty.kind(),
                        ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..)
                    ) && state.can_expect_ty_params()
                    {
                        FfiSafe
                    } else {
                        // ty::Alias(ty::Free), and all params/aliases for something
                        // defined beyond the FFI boundary
                        bug!("unexpected type in foreign function: {:?}", ty)
                    }
                }
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }

    fn visit_for_opaque_ty(&self, ty: Ty<'tcx>) -> PartialFfiResult<'tcx> {
        struct ProhibitOpaqueTypes;
        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for ProhibitOpaqueTypes {
            type Result = ControlFlow<Ty<'tcx>>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if !ty.has_opaque_types() {
                    return ControlFlow::Continue(());
                }

                if let ty::Alias(ty::Opaque, ..) = ty.kind() {
                    ControlFlow::Break(ty)
                } else {
                    ty.super_visit_with(self)
                }
            }
        }

        ty.visit_with(&mut ProhibitOpaqueTypes)
            .break_value()
            .map(|ty| FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None))
    }

    fn check_type(&self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = erase_and_maybe_normalize(self.cx, ty);
        self.visit_type(state, None, ty)
    }

    /// Checks the FFI-safety of a callback (`extern "ABI"` FnPtr)
    /// that is found in a no-FFI-safety-needed context.
    fn check_fnptr(&self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = erase_and_maybe_normalize(self.cx, ty);

        match *ty.kind() {
            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if sig.abi().is_rustic_abi() {
                    bug!(
                        "expected to inspect the type of an `extern \"ABI\"` FnPtr, not an internal-ABI one"
                    )
                } else {
                    self.visit_fnptr(VisitorState::NONE, None, ty, sig)
                }
            }
            r @ _ => {
                bug!("expected to inspect the type of an `extern \"ABI\"` FnPtr, not {:?}", r,)
            }
        }
    }

    /// Per-struct-field function that checks if a struct definition follows
    /// the Power alignment Rule (see the `check_struct_for_power_alignment` method).
    fn check_arg_for_power_alignment(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
        let tcx = cx.tcx;
        assert!(tcx.sess.target.os == "aix");

        // Structs (under repr(C)) follow the power alignment rule if:
        //   - the first field of the struct is a floating-point type that
        //     is greater than 4-bytes, or
        //   - the first field of the struct is an aggregate whose
        //     recursively first field is a floating-point type greater than
        //     4 bytes.
        if ty.is_floating_point() && ty.primitive_size(tcx).bytes() > 4 {
            return true;
        } else if let Adt(adt_def, _) = ty.kind()
            && adt_def.is_struct()
            && adt_def.repr().c()
            && !adt_def.repr().packed()
            && adt_def.repr().align.is_none()
        {
            let struct_variant = adt_def.variant(VariantIdx::ZERO);
            // Within a nested struct, all fields are examined to correctly
            // report if any fields after the nested struct within the
            // original struct are misaligned.
            for struct_field in &struct_variant.fields {
                let field_ty = tcx.type_of(struct_field.did).instantiate_identity();
                if Self::check_arg_for_power_alignment(cx, field_ty) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Check a struct definition for respect of the Power alignment Rule (as in PowerPC),
    /// which should be respected in the "aix" target OS.
    /// To do so, we must follow one of the two following conditions:
    /// - The first field of the struct must be floating-point type that
    ///    is greater than 4-bytes.
    ///  - The first field of the struct must be an aggregate whose
    ///    recursively first field is a floating-point type greater than
    ///    4 bytes.
    fn check_struct_for_power_alignment(
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>,
        adt_def: AdtDef<'tcx>,
    ) {
        let tcx = cx.tcx;
        // repr(C) structs also with packed or aligned representation
        // should be ignored.
        if adt_def.repr().c()
            && !adt_def.repr().packed()
            && adt_def.repr().align.is_none()
            && tcx.sess.target.os == "aix"
            && !adt_def.all_fields().next().is_none()
        {
            let struct_variant_data = item.expect_struct().2;
            for field_def in struct_variant_data.fields().iter().skip(1) {
                // Struct fields (after the first field) are checked for the
                // power alignment rule, as fields after the first are likely
                // to be the fields that are misaligned.
                let def_id = field_def.def_id;
                let ty = tcx.type_of(def_id).instantiate_identity();
                if Self::check_arg_for_power_alignment(cx, ty) {
                    cx.emit_span_lint(USES_POWER_ALIGNMENT, field_def.span, UsesPowerAlignment);
                }
            }
        }
    }
}

impl<'tcx> ImproperCTypesLint {
    /// Find any fn-ptr types with external ABIs in `ty`, and FFI-checks them.
    /// For example, `Option<extern "C" fn()>` FFI-checks `extern "C" fn()`.
    fn check_type_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        hir_ty: &hir::Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        struct FnPtrFinder<'tcx> {
            spans: Vec<Span>,
            tys: Vec<Ty<'tcx>>,
        }

        impl<'tcx> hir::intravisit::Visitor<'_> for FnPtrFinder<'tcx> {
            fn visit_ty(&mut self, ty: &'_ hir::Ty<'_, AmbigArg>) {
                debug!(?ty);
                if let hir::TyKind::FnPtr(hir::FnPtrTy { abi, .. }) = ty.kind
                    && !abi.is_rustic_abi()
                {
                    self.spans.push(ty.span);
                }

                hir::intravisit::walk_ty(self, ty)
            }
        }

        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for FnPtrFinder<'tcx> {
            type Result = ();

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if let ty::FnPtr(_, hdr) = ty.kind()
                    && !hdr.abi.is_rustic_abi()
                {
                    self.tys.push(ty);
                }

                ty.super_visit_with(self)
            }
        }

        let mut visitor = FnPtrFinder { spans: Vec::new(), tys: Vec::new() };
        ty.visit_with(&mut visitor);
        visitor.visit_ty_unambig(hir_ty);

        let all_types = iter::zip(visitor.tys.drain(..), visitor.spans.drain(..));
        for (fn_ptr_ty, span) in all_types {
            let visitor = ImproperCTypesVisitor::new(cx);
            let ffi_res = visitor.check_fnptr(fn_ptr_ty);

            self.process_ffi_result(cx, span, ffi_res, CItemKind::Callback);
        }
    }

    /// Regardless of a function's need to be "ffi-safe", look for fn-ptr argument/return types
    /// that need to be checked for ffi-safety.
    fn check_fn_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_type_for_external_abi_fnptr(cx, input_hir, *input_ty);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            self.check_type_for_external_abi_fnptr(cx, ret_hir, sig.output());
        }
    }

    /// For a local definition of a #[repr(C)] struct/enum/union, check that it is indeed FFI-safe.
    fn check_reprc_adt(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>,
        adt_def: AdtDef<'tcx>,
    ) {
        debug_assert!(
            adt_def.repr().c() && !adt_def.repr().packed() && adt_def.repr().align.is_none()
        );

        // FIXME(ctypes): this following call is awkward.
        // is there a way to perform its logic in MIR space rather than HIR space?
        // (so that its logic can be absorbed into visitor.visit_struct_or_union)
        ImproperCTypesVisitor::check_struct_for_power_alignment(cx, item, adt_def);
    }

    /// Check that an extern "ABI" static variable is of a ffi-safe type.
    fn check_foreign_static(&self, cx: &LateContext<'tcx>, id: hir::OwnerId, span: Span) {
        let ty = cx.tcx.type_of(id).instantiate_identity();
        let visitor = ImproperCTypesVisitor::new(cx);
        let ffi_res = visitor.check_type(VisitorState::STATIC_TY, ty);
        self.process_ffi_result(cx, span, ffi_res, CItemKind::ImportedExtern);
    }

    /// Check that a `#[no_mangle]`/`#[export_name = _]` static variable is of a ffi-safe type.
    fn check_exported_static(&self, cx: &LateContext<'tcx>, id: hir::OwnerId, span: Span) {
        let ty = cx.tcx.type_of(id).instantiate_identity();
        let visitor = ImproperCTypesVisitor::new(cx);
        let ffi_res = visitor.check_type(VisitorState::EXPORTED_STATIC_TY, ty);
        self.process_ffi_result(cx, span, ffi_res, CItemKind::ExportedStatic);
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    fn check_foreign_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_mode: CItemKind,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let visit_state = VisitorState::argument_from_fnmode(fn_mode);
            let visitor = ImproperCTypesVisitor::new(cx);
            let ffi_res = visitor.check_type(visit_state, *input_ty);
            self.process_ffi_result(cx, input_hir.span, ffi_res, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let visit_state = VisitorState::return_from_fnmode(fn_mode);
            let visitor = ImproperCTypesVisitor::new(cx);
            let ffi_res = visitor.check_type(visit_state, sig.output());
            self.process_ffi_result(cx, ret_hir.span, ffi_res, fn_mode);
        }
    }

    fn process_ffi_result(
        &self,
        cx: &LateContext<'tcx>,
        sp: Span,
        res: FfiResult<'tcx>,
        fn_mode: CItemKind,
    ) {
        match res {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(
                    cx,
                    ty.clone(),
                    sp,
                    vec![ImproperCTypesLayer {
                        ty,
                        note: fluent::lint_improper_ctypes_only_phantomdata,
                        span_note: None, // filled later
                        help: None,
                        inner_ty: None,
                    }],
                    fn_mode,
                );
            }
            FfiResult::FfiUnsafe(explanations) => {
                for explanation in explanations {
                    let mut ffiresult_recursor = ControlFlow::Continue(explanation.reason.as_ref());
                    let mut cimproper_layers: Vec<ImproperCTypesLayer<'_>> = vec![];

                    // this whole while block converts the arbitrarily-deep
                    // FfiResult stack to an ImproperCTypesLayer Vec
                    while let ControlFlow::Continue(FfiUnsafeReason { ty, note, help, inner }) =
                        ffiresult_recursor
                    {
                        if let Some(layer) = cimproper_layers.last_mut() {
                            layer.inner_ty = Some(ty.clone());
                        }
                        cimproper_layers.push(ImproperCTypesLayer {
                            ty: ty.clone(),
                            inner_ty: None,
                            help: help.clone(),
                            note: note.clone(),
                            span_note: None, // filled later
                        });

                        if let Some(inner) = inner {
                            ffiresult_recursor = ControlFlow::Continue(inner.as_ref());
                        } else {
                            ffiresult_recursor = ControlFlow::Break(());
                        }
                    }
                    let cause_ty = if let Some(cause_ty) = explanation.override_cause_ty {
                        cause_ty
                    } else {
                        // should always have at least one type
                        cimproper_layers.last().unwrap().ty.clone()
                    };
                    self.emit_ffi_unsafe_type_lint(cx, cause_ty, sp, cimproper_layers, fn_mode);
                }
            }
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &self,
        cx: &LateContext<'tcx>,
        ty: Ty<'tcx>,
        sp: Span,
        mut reasons: Vec<ImproperCTypesLayer<'tcx>>,
        fn_mode: CItemKind,
    ) {
        let lint = match fn_mode {
            CItemKind::ImportedExtern => IMPROPER_CTYPES,
            CItemKind::ExportedFunction => IMPROPER_C_FN_DEFINITIONS,
            CItemKind::ExportedStatic => IMPROPER_C_VAR_DEFINITIONS,
            CItemKind::Callback => IMPROPER_C_CALLBACKS,
        };
        let desc = match fn_mode {
            CItemKind::ImportedExtern => "`extern` block",
            CItemKind::ExportedFunction => "`extern` fn",
            CItemKind::ExportedStatic => "foreign-code-reachable static",
            CItemKind::Callback => "`extern` callback",
        };
        for reason in reasons.iter_mut() {
            reason.span_note = if let ty::Adt(def, _) = reason.ty.kind()
                && let Some(sp) = cx.tcx.hir_span_if_local(def.did())
            {
                Some(sp)
            } else {
                None
            };
        }

        cx.emit_span_lint(lint, sp, ImproperCTypes { ty, desc, label: sp, reasons });
    }
}

/// IMPROPER_CTYPES checks items that are part of a header to a non-rust library
/// Namely, functions and static variables in `extern "<abi>" { }`,
/// if `<abi>` is external (e.g. "C").
///
/// `IMPROPER_C_CALLBACKS` checks for function pointers marked with an external ABI.
/// (fields of type `extern "<abi>" fn`, where e.g. `<abi>` is `C`)
/// These pointers are searched in all other items which contain types
/// (e.g.functions, struct definitions, etc)
///
/// `IMPROPER_C_FN_DEFINITIONS` checks rust-defined functions that are marked
/// to be used from the other side of a FFI boundary.
/// In other words, `extern "<abi>" fn` definitions and trait-method declarations.
/// This only matters if `<abi>` is external (e.g. `C`).
///
/// and now combinatorics for pointees
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesLint {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        let abi = cx.tcx.hir_get_foreign_abi(it.hir_id());

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                // fnptrs are a special case, they always need to be treated as
                // "the element rendered unsafe" because their unsafety doesn't affect
                // their surroundings, and their type is often declared inline
                self.check_fn_for_external_abi_fnptr(cx, it.owner_id.def_id, sig.decl);
                if !abi.is_rustic_abi() {
                    self.check_foreign_fn(
                        cx,
                        CItemKind::ImportedExtern,
                        it.owner_id.def_id,
                        sig.decl,
                    );
                }
            }
            hir::ForeignItemKind::Static(ty, _, _) if !abi.is_rustic_abi() => {
                self.check_foreign_static(cx, it.owner_id, ty.span);
            }
            hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::Static(_, _, ty, _)
            | hir::ItemKind::Const(_, _, ty, _)
            | hir::ItemKind::TyAlias(_, _, ty) => {
                self.check_type_for_external_abi_fnptr(
                    cx,
                    ty,
                    cx.tcx.type_of(item.owner_id).instantiate_identity(),
                );

                // FIXME: cx.tcx.has_attr no worky
                // if matches!(item.kind, hir::ItemKind::Static(..))
                //     && (cx.tcx.has_attr(item.owner_id, sym::no_mangle)
                //         || cx.tcx.has_attr(item.owner_id, sym::export_name))
                if matches!(item.kind, hir::ItemKind::Static(..)) {
                    let is_exported_static = cx.tcx.get_all_attrs(item.owner_id).iter().any(|x| {
                        matches!(
                            x,
                            hir::Attribute::Parsed(
                                hir::attrs::AttributeKind::NoMangle(_)
                                    | hir::attrs::AttributeKind::ExportName { .. }
                            )
                        )
                    });
                    if is_exported_static {
                        self.check_exported_static(cx, item.owner_id, ty.span);
                    }
                }
            }
            // See `check_fn` for declarations, `check_foreign_items` for definitions in extern blocks
            hir::ItemKind::Fn { .. } => {}
            hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) | hir::ItemKind::Enum(..) => {
                // looking for extern FnPtr:s is delegated to `check_field_def`.
                let adt_def: AdtDef<'tcx> = cx.tcx.adt_def(item.owner_id.to_def_id());

                if adt_def.repr().c() && !adt_def.repr().packed() && adt_def.repr().align.is_none()
                {
                    self.check_reprc_adt(cx, item, adt_def);
                }
            }

            // Doesn't define something that can contain a external type to be checked.
            hir::ItemKind::Impl(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::GlobalAsm { .. }
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::ExternCrate(..) => {}
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'tcx>) {
        self.check_type_for_external_abi_fnptr(
            cx,
            field.ty,
            cx.tcx.type_of(field.def_id).instantiate_identity(),
        );
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        _: Span,
        id: LocalDefId,
    ) {
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        // fnptrs are a special case, they always need to be treated as
        // "the element rendered unsafe" because their unsafety doesn't affect
        // their surroundings, and their type is often declared inline
        self.check_fn_for_external_abi_fnptr(cx, id, decl);
        if !abi.is_rustic_abi() {
            self.check_foreign_fn(cx, CItemKind::ExportedFunction, id, decl);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, tr_it: &hir::TraitItem<'tcx>) {
        match tr_it.kind {
            hir::TraitItemKind::Const(hir_ty, _) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
            hir::TraitItemKind::Fn(sig, trait_fn) => {
                match trait_fn {
                    // if the method is defined here,
                    // there is a matching ``LateLintPass::check_fn`` call,
                    // let's not redo that work
                    hir::TraitFn::Provided(_) => return,
                    hir::TraitFn::Required(_) => (),
                }
                let local_id = tr_it.owner_id.def_id;
                if sig.header.abi.is_rustic_abi() {
                    self.check_fn_for_external_abi_fnptr(cx, local_id, sig.decl);
                } else {
                    self.check_foreign_fn(cx, CItemKind::ExportedFunction, local_id, sig.decl);
                }
            }
            hir::TraitItemKind::Type(_, ty_maybe) => {
                if let Some(hir_ty) = ty_maybe {
                    let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                    self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
                }
            }
        }
    }
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, im_it: &hir::ImplItem<'tcx>) {
        // note: we do not skip these checks eventhough they might generate dupe warnings because:
        // - the corresponding trait might be in another crate
        // - the corresponding trait might have some templating involved, so only the impl has the full type information
        match im_it.kind {
            hir::ImplItemKind::Type(hir_ty) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
            hir::ImplItemKind::Fn(_sig, _) => {
                // see ``LateLintPass::check_fn``
            }
            hir::ImplItemKind::Const(hir_ty, _) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
        }
    }
}
