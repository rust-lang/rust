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
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{Span, sym};
use tracing::debug;

use super::repr_nullable_ptr;
use crate::lints::{ImproperCTypes, ImproperCTypesLayer, UsesPowerAlignment};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    /// (In other words, declarations of items defined in foreign code.)
    /// This also includes all [`extern` function] pointers.
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe extern "C" {
    ///     static STATIC: String;
    ///     fn some_func(a:String);
    /// }
    /// extern "C" fn register_callback(a: i32, call: extern "C" fn(char)) { /* ... */ }
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
    /// The `improper_ctypes_definitions` lint detects incorrect use of
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
    pub(crate) IMPROPER_CTYPES_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
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
    IMPROPER_CTYPES_DEFINITIONS,
    USES_POWER_ALIGNMENT,
]);

type Sig<'tcx> = Binder<'tcx, FnSig<'tcx>>;

/// Extract (binder-wrapped) FnSig object from a FnPtr's mir::Ty
fn get_fn_sig_from_mir_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Sig<'tcx> {
    let ty = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty);
    match *ty.kind() {
        ty::FnPtr(sig_tys, hdr) => {
            let sig = sig_tys.with(hdr);
            if sig.abi().is_rustic_abi() {
                bug!(
                    "expected to inspect the type of an `extern \"ABI\"` FnPtr, not an internal-ABI one"
                )
            } else {
                sig
            }
        }
        r @ _ => {
            bug!("expected to inspect the type of an `extern \"ABI\"` FnPtr, not {:?}", r,)
        }
    }
}

// FIXME(ctypes): it seems that tests/ui/lint/opaque-ty-ffi-normalization-cycle.rs relies this:
// we consider opaque aliases that normalise to something else to be unsafe.
// ...is it the behaviour we want?
/// a modified version of cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty)
/// so that opaque types prevent normalisation once region erasure occurs
fn erase_and_maybe_normalize<'tcx>(cx: &LateContext<'tcx>, value: Ty<'tcx>) -> Ty<'tcx> {
    if (!value.has_aliases()) || value.has_opaque_types() {
        cx.tcx.erase_and_anonymize_regions(value)
    } else {
        cx.tcx.try_normalize_erasing_regions(cx.typing_env(), value).unwrap_or(value)
        // note: the code above ^^^ would only cause a call to the commented code below vvv
        //let value = cx.tcx.erase_and_anonymize_regions(value);
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

fn variant_has_complex_ctor(variant: &ty::VariantDef) -> bool {
    // CtorKind::Const means a "unit" ctor
    !matches!(variant.ctor_kind(), Some(CtorKind::Const))
}

/// Per-struct-field function that checks if a struct definition follows
/// the Power alignment Rule (see the `check_struct_for_power_alignment` function).
fn check_arg_for_power_alignment<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
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
            if check_arg_for_power_alignment(cx, field_ty) {
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
fn check_struct_for_power_alignment<'tcx>(
    cx: &LateContext<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    adt_def: AdtDef<'tcx>,
) {
    let tcx = cx.tcx;

    // Only consider structs (not enums or unions) on AIX.
    if tcx.sess.target.os != "aix" || !adt_def.is_struct() {
        return;
    }

    // The struct must be repr(C), but ignore it if it explicitly specifies its alignment with
    // either `align(N)` or `packed(N)`.
    if adt_def.repr().c() && !adt_def.repr().packed() && adt_def.repr().align.is_none() {
        let struct_variant_data = item.expect_struct().2;
        for field_def in struct_variant_data.fields().iter().skip(1) {
            // Struct fields (after the first field) are checked for the
            // power alignment rule, as fields after the first are likely
            // to be the fields that are misaligned.
            let ty = tcx.type_of(field_def.def_id).instantiate_identity();
            if check_arg_for_power_alignment(cx, ty) {
                cx.emit_span_lint(USES_POWER_ALIGNMENT, field_def.span, UsesPowerAlignment);
            }
        }
    }
}

/// Annotates the nature of the "original item" being checked, and its relation
/// to FFI boundaries.
/// Mainly, whether is is something defined in rust and exported through the FFI boundary,
/// or something rust imports through the same boundary.
/// Callbacks are ultimately treated as imported items, in terms of denying/warning/ignoring FFI-unsafety
#[derive(Clone, Copy, Debug)]
enum CItemKind {
    /// Imported items in an `extern "C"` block (function declarations, static variables) -> IMPROPER_CTYPES
    ImportedExtern,
    /// `extern "C"` function definitions, to be used elsewhere -> IMPROPER_CTYPES_DEFINITIONS,
    ExportedFunction,
    /// `extern "C"` function pointers -> also IMPROPER_CTYPES,
    Callback,
}

/// Annotates whether we are in the context of a function's argument types or return type.
#[derive(Clone, Copy)]
enum FnPos {
    Arg,
    Ret,
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
enum IndirectionKind {
    /// Box (valid non-null pointer, owns pointee).
    Box,
    /// Ref (valid non-null pointer, borrows pointee).
    Ref,
    /// Raw pointer (not necessarily non-null or valid. no info on ownership).
    RawPtr,
}

/// The different ways a given type can have/not have a fixed size.
/// Relies on the vocabulary of the Hierarchy of Sized Traits change (`#![feature(sized_hierarchy)]`)
#[derive(Clone, Copy)]
enum TypeSizedness {
    /// Type of definite size (pointers are C-compatible).
    Sized,
    /// Unsized type because it includes an opaque/foreign type (pointers are C-compatible).
    /// (Relies on all Unsized types being `extern` types, and unable to be used in an array/slice)
    Unsized,
    /// MetaSized types are types whose size can be computed from pointer metadata (slice, string, dyn Trait, closure, ...)
    /// (pointers are not C-compatible).
    MetaSized,
    /// Not known, usually for placeholder types (Self in non-impl trait functions, type parameters, aliases, the like).
    NotYetKnown,
}

/// Determine if a type is sized or not, and whether it affects references/pointers/boxes to it.
fn get_type_sizedness<'tcx, 'a>(cx: &'a LateContext<'tcx>, ty: Ty<'tcx>) -> TypeSizedness {
    let tcx = cx.tcx;

    // note that sizedness is unrelated to inhabitedness
    if ty.is_sized(tcx, cx.typing_env()) {
        TypeSizedness::Sized
    } else {
        // the overall type is !Sized or ?Sized
        match ty.kind() {
            ty::Slice(_) | ty::Str | ty::Dynamic(..) => TypeSizedness::MetaSized,
            ty::Foreign(..) => TypeSizedness::Unsized,
            ty::Adt(def, args) => {
                // for now assume: boxes and phantoms don't mess with this
                match def.adt_kind() {
                    AdtKind::Union | AdtKind::Enum => {
                        bug!("unions and enums are necessarily sized")
                    }
                    AdtKind::Struct => {
                        if let Some(intermediate) =
                            def.sizedness_constraint(tcx, ty::SizedTraitKind::MetaSized)
                        {
                            let ty = intermediate.instantiate(tcx, args);
                            get_type_sizedness(cx, ty)
                        } else {
                            debug_assert!(
                                def.sizedness_constraint(tcx, ty::SizedTraitKind::Sized).is_some()
                            );
                            TypeSizedness::MetaSized
                        }

                        // if let Some(sym::cstring_type | sym::cstr_type) =
                        //     tcx.get_diagnostic_name(def.did())
                        // {
                        //     return TypeSizedness::MetaSized;
                        // }

                        // // note: non-exhaustive structs from other crates are not assumed to be ?Sized
                        // // for the purpose of sizedness, it seems we are allowed to look at its current contents.

                        // if def.non_enum_variant().fields.is_empty() {
                        //     bug!("an empty struct is necessarily sized");
                        // }

                        // let variant = def.non_enum_variant();

                        // // only the last field may be !Sized (or ?Sized in the case of type params)
                        // let last_field = match (&variant.fields).iter().last() {
                        //     Some(last_field) => last_field,
                        //     // even nonexhaustive-empty structs from another crate are considered Sized
                        //     // (eventhough one could add a !Sized field to them)
                        //     None => bug!("Empty struct should be Sized, right?"), //
                        // };
                        // let field_ty = get_type_from_field(cx, last_field, args);
                        // match get_type_sizedness(cx, field_ty) {
                        //     s @ (TypeSizedness::MetaSized
                        //     | TypeSizedness::Unsized
                        //     | TypeSizedness::NotYetKnown) => s,
                        //     TypeSizedness::Sized => {
                        //         bug!("failed to find the reason why struct `{:?}` is unsized", ty)
                        //     }
                        // }
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
                    s @ (TypeSizedness::MetaSized
                    | TypeSizedness::Unsized
                    | TypeSizedness::NotYetKnown) => s,
                    TypeSizedness::Sized => {
                        bug!("failed to find the reason why tuple `{:?}` is unsized", ty)
                    }
                }
            }

            ty::Pat(base, _) => get_type_sizedness(cx, *base),

            ty_kind @ (ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Array(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnPtr(..)
            | ty::Never) => {
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
    /// VisitorState flags that are linked with the root type's use.
    /// (These are the permanent part of the state, kept when visiting new mir::Ty.)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RootUseFlags: u16 {
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

/// Description of the relationship between current mir::Ty and
/// the type (or lack thereof) immediately containing it
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OuterTyKind {
    None,
    /// Pointee through ref, raw pointer or Box
    /// (we don't need to distinguish the ownership of Box specifically)
    Pointee {
        mutable: hir::Mutability,
        raw: bool,
    },
    /// For struct/enum/union fields
    AdtField,
    /// For arrays/slices but also tuples
    OtherItem,
}

impl OuterTyKind {
    /// Computes the relationship by providing the containing mir::Ty itself
    fn from_outer_ty<'tcx>(ty: Ty<'tcx>) -> Self {
        match ty.kind() {
            ty::FnPtr(..) => Self::None,
            k @ (ty::Ref(_, _, mutable) | ty::RawPtr(_, mutable)) => {
                Self::Pointee { raw: matches!(k, ty::RawPtr(..)), mutable: *mutable }
            }
            ty::Adt(..) => {
                if ty.boxed_ty().is_some() {
                    Self::Pointee { raw: false, mutable: hir::Mutability::Mut }
                } else {
                    Self::AdtField
                }
            }
            ty::Tuple(..) | ty::Array(..) | ty::Slice(_) => Self::OtherItem,
            k @ _ => bug!("Unexpected outer type {:?} of kind {:?}", ty, k),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct VisitorState {
    /// Flags describing both the overall context in which the current mir::Ty is,
    /// linked to how the Visitor's original mir::Ty was used.
    root_use_flags: RootUseFlags,
    /// Flags describing both the immediate context in which the current mir::Ty is,
    /// linked to how it relates to its parent mir::Ty (or lack thereof).
    outer_ty_kind: OuterTyKind,
    /// Type recursion depth, to prevent infinite recursion
    depth: usize,
}

impl RootUseFlags {
    // The values that can be set.
    const STATIC_TY: Self = Self::STATIC;
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
}

impl VisitorState {
    /// From an existing state, compute the state of any subtype of the current type.
    /// (General case.)
    fn get_next<'tcx>(&self, current_ty: Ty<'tcx>) -> Self {
        assert!(!matches!(current_ty.kind(), ty::FnPtr(..)));
        Self {
            root_use_flags: self.root_use_flags,
            outer_ty_kind: OuterTyKind::from_outer_ty(current_ty),
            depth: self.depth + 1,
        }
    }

    /// Generate the state for an "outermost" type that needs to be checked
    fn entry_point(root_use_flags: RootUseFlags) -> Self {
        Self { root_use_flags, outer_ty_kind: OuterTyKind::None, depth: 0 }
    }

    /// Get the proper visitor state for a given function's arguments or return type.
    fn entry_point_from_fnmode(fn_mode: CItemKind, fn_pos: FnPos) -> Self {
        let p_flags = match (fn_mode, fn_pos) {
            (CItemKind::ExportedFunction, FnPos::Ret) => RootUseFlags::RETURN_TY_IN_DEFINITION,
            (CItemKind::ImportedExtern, FnPos::Ret) => RootUseFlags::RETURN_TY_IN_DECLARATION,
            (CItemKind::Callback, FnPos::Ret) => RootUseFlags::RETURN_TY_IN_FNPTR,
            (CItemKind::ExportedFunction, FnPos::Arg) => RootUseFlags::ARGUMENT_TY_IN_DEFINITION,
            (CItemKind::ImportedExtern, FnPos::Arg) => RootUseFlags::ARGUMENT_TY_IN_DECLARATION,
            (CItemKind::Callback, FnPos::Arg) => RootUseFlags::ARGUMENT_TY_IN_FNPTR,
        };
        Self::entry_point(p_flags)
    }

    /// Get the proper visitor state for a static variable's type
    fn static_var() -> Self {
        Self::entry_point(RootUseFlags::STATIC_TY)
    }

    /// Whether the type is used as the type of a static variable.
    fn is_direct_in_static(&self) -> bool {
        let ret = self.root_use_flags.contains(RootUseFlags::STATIC);
        if ret {
            debug_assert!(!self.root_use_flags.contains(RootUseFlags::FUNC));
        }
        ret && matches!(self.outer_ty_kind, OuterTyKind::None)
    }

    /// Whether the type is used in a function.
    fn is_in_function(&self) -> bool {
        let ret = self.root_use_flags.contains(RootUseFlags::FUNC);
        if ret {
            debug_assert!(!self.root_use_flags.contains(RootUseFlags::STATIC));
        }
        ret
    }
    /// Whether the type is used (directly or not) in a function, in return position.
    fn is_in_function_return(&self) -> bool {
        let ret = self.root_use_flags.contains(RootUseFlags::FN_RETURN);
        if ret {
            debug_assert!(self.is_in_function());
        }
        ret
    }

    /// Whether the type is directly used in a function, in return position.
    fn is_direct_function_return(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::None) && self.is_in_function_return()
    }

    /// Whether the type itself is the type of a function argument or return type.
    fn is_direct_in_function(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::None) && self.is_in_function()
    }

    /// Whether the type is used (directly or not) in a defined function.
    /// In other words, whether or not we allow non-FFI-safe types behind a C pointer,
    /// to be treated as an opaque type on the other side of the FFI boundary.
    fn is_in_defined_function(&self) -> bool {
        self.root_use_flags.contains(RootUseFlags::DEFINED) && self.is_in_function()
    }

    /// Whether we can expect type parameters and co in a given type.
    fn can_expect_ty_params(&self) -> bool {
        // rust-defined functions, as well as FnPtrs
        self.root_use_flags.contains(RootUseFlags::THEORETICAL) || self.is_in_defined_function()
    }

    /// Whether the current type is an ADT field
    fn is_field(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::AdtField)
    }

    /// Whether the current type is behind a pointer that doesn't allow mutating this
    fn is_nonmut_pointee(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::Pointee { mutable: hir::Mutability::Not, .. })
    }

    /// Whether the current type is behind a raw pointer
    fn is_raw_pointee(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::Pointee { raw: true, .. })
    }

    /// Whether the current type directly in the memory layout of the parent ty
    fn is_memory_inlined(&self) -> bool {
        matches!(self.outer_ty_kind, OuterTyKind::AdtField | OuterTyKind::OtherItem)
    }
}

/// Visitor used to recursively traverse MIR types and evaluate FFI-safety.
/// It uses ``check_*`` methods as entrypoints to be called elsewhere,
/// and ``visit_*`` methods to recurse.
struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// The module id of the item being checked for FFI-safety
    mod_id: DefId,
    /// To prevent problems with recursive types,
    /// add a types-in-check cache.
    ty_cache: FxHashSet<Ty<'tcx>>,
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, mod_id: DefId) -> Self {
        Self { cx, mod_id, ty_cache: FxHashSet::default() }
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

    /// Return the right help for Cstring and Cstr-linked unsafety.
    fn visit_cstr(&mut self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        debug_assert!(matches!(ty.kind(), ty::Adt(def, _)
            if matches!(
                self.cx.tcx.get_diagnostic_name(def.did()),
                Some(sym::cstring_type | sym::cstr_type)
            )
        ));

        let help = if state.is_nonmut_pointee() {
            fluent::lint_improper_ctypes_cstr_help_const
        } else {
            fluent::lint_improper_ctypes_cstr_help_mut
        };

        FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_cstr_reason, Some(help))
    }

    /// Checks if the given indirection (box,ref,pointer) is "ffi-safe".
    fn visit_indirection(
        &mut self,
        state: VisitorState,
        ty: Ty<'tcx>,
        inner_ty: Ty<'tcx>,
        indirection_kind: IndirectionKind,
    ) -> FfiResult<'tcx> {
        let tcx = self.cx.tcx;

        if let ty::Adt(def, _) = inner_ty.kind() {
            if let Some(diag_name @ (sym::cstring_type | sym::cstr_type)) =
                tcx.get_diagnostic_name(def.did())
            {
                // we have better error messages when checking for C-strings directly
                let mut cstr_res = self.visit_cstr(state.get_next(ty), inner_ty); // always unsafe with one depth-one reason.

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
                    let note = match indirection_kind {
                        IndirectionKind::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                        IndirectionKind::Ref => fluent::lint_improper_ctypes_unsized_ref,
                        IndirectionKind::Box => fluent::lint_improper_ctypes_unsized_box,
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
        let type_sizedness = get_type_sizedness(self.cx, inner_ty);
        match type_sizedness {
            TypeSizedness::Unsized | TypeSizedness::Sized => {
                if matches!(
                    (type_sizedness, indirection_kind),
                    (TypeSizedness::Unsized, IndirectionKind::Box)
                ) {
                    // Box<_> means rust is capable of drop()'ing the pointee,
                    // which is impossible for `extern` types (foreign opaque types).
                    bug!(
                        "FFI-unsafeties similar to `Box<extern type>` currently cause "
                        "compilation errors that should prevent ImproperCTypes from running. "
                        "If you see this, it is likely this behaviour has changed."
                    );
                }
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
                self.visit_type(state.get_next(ty), inner_ty)
            }
            TypeSizedness::MetaSized => {
                let help = match inner_ty.kind() {
                    ty::Str => Some(fluent::lint_improper_ctypes_str_help),
                    ty::Slice(_) => Some(fluent::lint_improper_ctypes_slice_help),
                    _ => None,
                };
                let reason = match indirection_kind {
                    IndirectionKind::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                    IndirectionKind::Ref => fluent::lint_improper_ctypes_unsized_ref,
                    IndirectionKind::Box => fluent::lint_improper_ctypes_unsized_box,
                };
                return FfiResult::new_with_reason(ty, reason, help);
            }
        }
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn visit_variant_fields(
        &mut self,
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
                if !ty.is_inhabited_from(self.cx.tcx, self.mod_id, self.cx.typing_env()) {
                    // let's consider transparent structs to be maybe unsafe if uninhabited,
                    // even if that is because of fields otherwise ignored in FFI-safety checks
                    ffires_accumulator += variant
                        .fields
                        .iter()
                        .map(|field| {
                            let field_ty = get_type_from_field(self.cx, field, args);
                            let mut field_res = self.visit_type(state.get_next(ty), field_ty);
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
            let ffi_res = self.visit_type(state.get_next(ty), field_ty);

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
        &mut self,
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
                    // FIXME(#60405): confirm that this makes sense for unions once #60405 / RFC2645 stabilises
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

        // Here, if there is something wrong, then the "fault" comes from inside the struct itself.
        // Even if we add more details to the lint, the initial line must specify that
        // the FFI-unsafety is because of the struct
        // Plus, if the struct is from another crate, then there's not much that can be done anyways
        //
        // So, we override the "cause type" of the lint.
        ffires.with_overrides(Some(ty))
    }

    fn visit_enum(
        &mut self,
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
                return self.visit_type(state.get_next(ty), inner_ty);
            }

            return FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_enum_repr_reason,
                Some(fluent::lint_improper_ctypes_enum_repr_help),
            );
        }

        // FIXME(ctypes): connect `def.repr().int` to visit_numeric
        // (for now it's OK, `repr(char)` doesn't exist and visit_numeric doesn't warn on anything else)

        let enum_non_exhaustive = def.variant_list_has_applicable_non_exhaustive();
        // Check the contained variants.

        // non_exhaustive suggests it is possible that someone might break ABI
        // See: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
        // so warn on complex enums being used outside their crate.
        //
        // We treat `#[non_exhaustive]` enum variants as unsafe if the enum is passed by-value,
        // as additions it will change it size.
        //
        // We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
        // This includes linting, on a best-effort basis. There are valid additions that are unlikely.
        //
        // Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
        // so we don't need the lint to account for it.
        // e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
        // Which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)

        let (mut improper_on_nonexhaustive_flag, mut nonexhaustive_variant_flag) = (false, false);
        def.variants().iter().for_each(|variant| {
            improper_on_nonexhaustive_flag |=
                enum_non_exhaustive && variant_has_complex_ctor(variant);
            nonexhaustive_variant_flag |= variant.field_list_has_applicable_non_exhaustive();
        });

        if improper_on_nonexhaustive_flag {
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
    fn visit_type(&mut self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        if !(self.ty_cache.insert(ty)
            && self.cx.tcx.recursion_limit().value_within_limit(state.depth))
        {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, args) => {
                if let Some(inner_ty) = ty.boxed_ty() {
                    return self.visit_indirection(state, ty, inner_ty, IndirectionKind::Box);
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        // There are two ways to encounter cstr here (since pointees are treated elsewhere):
                        // - Cstr used as an argument of a FnPtr (!Sized structs are in fact allowed there)
                        // - Cstr as the last field of a struct
                        // This excludes non-compiling code where a CStr is used where !Sized is not allowed
                        // (currently those mistakes prevent this lint from running)
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return self.visit_cstr(state, ty);
                        }
                        self.visit_struct_or_union(state, ty, def, args)
                    }
                    AdtKind::Enum => self.visit_enum(state, ty, def, args),
                }
            }

            // Pattern types are just extra invariants on the type that you need to uphold,
            // but only the base type is relevant for being representable in FFI.
            // (note: this lint was written when pattern types could only be integers constrained to ranges)
            // (also note: the lack of ".get_next(ty)" on the state is on purpose)
            ty::Pat(pat_ty, _) => self.visit_type(state, pat_ty),

            // types which likely have a stable representation, if the target architecture defines those
            // note: before rust 1.77, 128-bit ints were not FFI-safe on x86_64
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => FfiResult::FfiSafe,

            ty::Bool => FfiResult::FfiSafe,

            ty::Char => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_char_reason,
                Some(fluent::lint_improper_ctypes_char_help),
            ),

            ty::Slice(inner_ty) => {
                // ty::Slice is used for !Sized arrays, since they are the pointee for actual slices
                let slice_is_actually_array =
                    state.is_memory_inlined() || state.is_direct_in_static();

                if slice_is_actually_array {
                    self.visit_type(state.get_next(ty), inner_ty)
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
                if tuple.is_empty()
                    && (
                        // C functions can return void
                        state.is_direct_function_return()
                    // `()` fields are safe
                    || state.is_field()
                        // this serves as a "void*"
                        || state.is_raw_pointee()
                    )
                {
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
                return self.visit_indirection(state, ty, inner_ty, IndirectionKind::RawPtr);
            }
            ty::Ref(_, inner_ty, _) => {
                return self.visit_indirection(state, ty, inner_ty, IndirectionKind::Ref);
            }

            ty::Array(inner_ty, _) => {
                if state.is_direct_in_function() {
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
                    self.visit_type(state.get_next(ty), inner_ty)
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

    fn visit_for_opaque_ty(&mut self, ty: Ty<'tcx>) -> PartialFfiResult<'tcx> {
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

    fn check_type(&mut self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = erase_and_maybe_normalize(self.cx, ty);
        self.visit_type(state, ty)
    }
}

impl<'tcx> ImproperCTypesLint {
    /// Find any fn-ptr types with external ABIs in `ty`, and FFI-checks them.
    /// For example, `Option<extern "C" fn()>` FFI-checks `extern "C" fn()`.
    fn check_type_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        hir_ty: &'tcx hir::Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        struct FnPtrFinder<'tcx> {
            current_depth: usize,
            depths: Vec<usize>,
            decls: Vec<&'tcx hir::FnDecl<'tcx>>,
            hir_ids: Vec<hir::HirId>,
            tys: Vec<Ty<'tcx>>,
        }

        impl<'tcx> hir::intravisit::Visitor<'tcx> for FnPtrFinder<'tcx> {
            fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
                debug!(?ty);
                self.current_depth += 1;
                if let hir::TyKind::FnPtr(hir::FnPtrTy { abi, decl, .. }) = ty.kind
                    && !abi.is_rustic_abi()
                {
                    self.decls.push(*decl);
                    self.depths.push(self.current_depth);
                    self.hir_ids.push(ty.hir_id);
                }

                hir::intravisit::walk_ty(self, ty);
                self.current_depth -= 1;
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

        let mut visitor = FnPtrFinder {
            hir_ids: Vec::new(),
            tys: Vec::new(),
            decls: Vec::new(),
            depths: Vec::new(),
            current_depth: 0,
        };
        ty.visit_with(&mut visitor);
        visitor.visit_ty_unambig(hir_ty);

        let all_types = iter::zip(
            iter::zip(visitor.depths.drain(..), visitor.hir_ids.drain(..)),
            iter::zip(visitor.tys.drain(..), visitor.decls.drain(..)),
        );

        for ((depth, hir_id), (fn_ptr_ty, decl)) in all_types {
            let mir_sig = get_fn_sig_from_mir_ty(cx, fn_ptr_ty);
            let mod_id = cx.tcx.parent_module(hir_id).to_def_id();

            self.check_foreign_fn(cx, CItemKind::Callback, mir_sig, decl, mod_id, depth);
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
        check_struct_for_power_alignment(cx, item, adt_def);
    }

    /// Check that an extern "ABI" static variable is of a ffi-safe type.
    fn check_foreign_static(&mut self, cx: &LateContext<'tcx>, id: hir::HirId, span: Span) {
        let ty = cx.tcx.type_of(id.owner).instantiate_identity();
        let mod_id = cx.tcx.parent_module(id).to_def_id();
        let mut visitor = ImproperCTypesVisitor::new(cx, mod_id);
        let ffi_res = visitor.check_type(VisitorState::static_var(), ty);
        self.process_ffi_result(cx, span, ffi_res, CItemKind::ImportedExtern);
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    fn check_foreign_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_mode: CItemKind,
        sig: Sig<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        mod_id: DefId,
        depth: usize,
    ) {
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let mut state = VisitorState::entry_point_from_fnmode(fn_mode, FnPos::Arg);
            state.depth = depth;
            let mut visitor = ImproperCTypesVisitor::new(cx, mod_id);
            let ffi_res = visitor.check_type(state, *input_ty);
            self.process_ffi_result(cx, input_hir.span, ffi_res, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let mut state = VisitorState::entry_point_from_fnmode(fn_mode, FnPos::Ret);
            state.depth = depth;
            let mut visitor = ImproperCTypesVisitor::new(cx, mod_id);
            let ffi_res = visitor.check_type(state, sig.output());
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
            CItemKind::ExportedFunction => IMPROPER_CTYPES_DEFINITIONS,
            // Internally, we treat this differently, but at the end of the day
            // their linting needs to be enabled/disabled alongside that of "FFI-imported" items.
            CItemKind::Callback => IMPROPER_CTYPES,
        };
        let desc = match fn_mode {
            CItemKind::ImportedExtern => "`extern` block",
            CItemKind::ExportedFunction => "`extern` fn",
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
/// it also checks for function pointers marked with an external ABI.
/// (fields of type `extern "<abi>" fn`, where e.g. `<abi>` is `C`)
/// These pointers are searched in all other items which contain types
/// (e.g.functions, struct definitions, etc)
///
/// `IMPROPER_CTYPES_DEFINITIONS` checks rust-defined functions that are marked
/// to be used from the other side of a FFI boundary.
/// In other words, `extern "<abi>" fn` definitions and trait-method declarations.
/// This only matters if `<abi>` is external (e.g. `C`).
///
/// maybe later: specialised lints for pointees
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesLint {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        let abi = cx.tcx.hir_get_foreign_abi(it.hir_id());

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                // fnptrs are a special case, they always need to be treated as
                // "the element rendered unsafe" because their unsafety doesn't affect
                // their surroundings, and their type is often declared inline
                self.check_fn_for_external_abi_fnptr(cx, it.owner_id.def_id, sig.decl);
                let mir_sig = cx.tcx.fn_sig(it.owner_id.def_id).instantiate_identity();
                let mod_id = cx.tcx.parent_module_from_def_id(it.owner_id.def_id).to_def_id();
                if !abi.is_rustic_abi() {
                    self.check_foreign_fn(cx, CItemKind::ImportedExtern, mir_sig, sig.decl, mod_id, 0);
                }
            }
            hir::ForeignItemKind::Static(ty, _, _) if !abi.is_rustic_abi() => {
                self.check_foreign_static(cx, it.hir_id(), ty.span);
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
            let mir_sig = cx.tcx.fn_sig(id).instantiate_identity();
            let mod_id = cx.tcx.parent_module_from_def_id(id).to_def_id();
            self.check_foreign_fn(cx, CItemKind::ExportedFunction, mir_sig, decl, mod_id, 0);
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

                self.check_fn_for_external_abi_fnptr(cx, local_id, sig.decl);
                if !sig.header.abi.is_rustic_abi() {
                    let mir_sig = cx.tcx.fn_sig(local_id).instantiate_identity();
                    let mod_id = cx.tcx.parent_module_from_def_id(local_id).to_def_id();
                    self.check_foreign_fn(cx, CItemKind::ExportedFunction, mir_sig, sig.decl, mod_id, 0);
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
