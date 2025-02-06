use std::iter;
use std::ops::ControlFlow;

use rustc_abi::{ExternAbi, VariantIdx};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::DiagMessage;
use rustc_hir as hir;
use rustc_hir::AmbigArg;
use rustc_hir::def::CtorKind;
use rustc_hir::intravisit::VisitorExt;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Adt, AdtKind, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt,
};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use rustc_type_ir::{Binder, FnSig};
use tracing::debug;

use super::{
    CItemKind, IMPROPER_CTYPES, IMPROPER_CTYPES_DEFINITIONS, ImproperCTypesDeclarations,
    ImproperCTypesDefinitions, USES_POWER_ALIGNMENT, repr_nullable_ptr,
};
use crate::lints::{ImproperCTypes, ImproperCTypesLayer, UsesPowerAlignment};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

type Sig<'tcx> = Binder<TyCtxt<'tcx>, FnSig<TyCtxt<'tcx>>>;

/// for a given `extern "ABI"`, tell wether that ABI is *not* considered a FFI boundary
fn fn_abi_is_internal(abi: ExternAbi) -> bool {
    matches!(
        abi,
        ExternAbi::Rust | ExternAbi::RustCall | ExternAbi::RustCold | ExternAbi::RustIntrinsic
    )
}

// a shorthand for an often used lifetime-region normalisation step
#[inline]
fn normalize_if_possible<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty)
}

// getting the (normalized) type out of a field (for, e.g., an enum variant or a tuple)
#[inline]
fn get_type_from_field<'tcx>(
    cx: &LateContext<'tcx>,
    field: &ty::FieldDef,
    args: GenericArgsRef<'tcx>,
) -> Ty<'tcx> {
    let field_ty = field.ty(cx.tcx, args);
    normalize_if_possible(cx, field_ty)
}

/// Check a variant of a non-exhaustive enum for improper ctypes
/// returns two bools: "we have FFI-unsafety due to non-exhaustive enum" and
/// "we have FFI-unsafety due to a non-exhaustive enum variant"
///
/// We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
/// This includes linting, on a best-effort basis. There are valid additions that are unlikely.
///
/// Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
/// so we don't need the lint to account for it.
/// e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
pub(crate) fn flag_non_exhaustive_variant(
    non_local_def: bool,
    variant: &ty::VariantDef,
) -> (bool, bool) {
    // non_exhaustive suggests it is possible that someone might break ABI
    // see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
    // so warn on complex enums being used outside their crate
    if non_local_def {
        // which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)
        if variant_has_complex_ctor(variant) {
            return (true, false);
        }
    }

    let non_exhaustive_variant_fields = variant.is_field_list_non_exhaustive();
    if non_exhaustive_variant_fields && !variant.def_id.is_local() {
        return (false, true);
    }

    (false, false)
}

fn variant_has_complex_ctor(variant: &ty::VariantDef) -> bool {
    // CtorKind::Const means a "unit" ctor
    !matches!(variant.ctor_kind(), Some(CtorKind::Const))
}

// non_exhaustive suggests it is possible that someone might break ABI
// see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
// so warn on complex enums being used outside their crate
pub(crate) fn non_local_and_non_exhaustive(def: ty::AdtDef<'_>) -> bool {
    def.is_variant_list_non_exhaustive() && !def.did().is_local()
}

struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    cache: FxHashSet<Ty<'tcx>>,
}

#[derive(Clone, Debug)]
struct FfiUnsafeReason<'tcx> {
    ty: Ty<'tcx>,
    reason: DiagMessage,
    help: Option<DiagMessage>,
    inner: Option<Box<FfiUnsafeReason<'tcx>>>,
}

#[derive(Clone, Debug)]
enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe(Vec<FfiUnsafeReason<'tcx>>),
}

impl<'tcx> FfiResult<'tcx> {
    /// Simplified creation of the FfiUnsafe variant for a single unsafety reason
    fn new_with_reason(ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        Self::FfiUnsafe(vec![FfiUnsafeReason { ty, help, reason: note, inner: None }])
    }

    /// If the FfiUnsafe variant, 'wraps' all reasons,
    /// creating new `FfiUnsafeReason`s, putting the originals as their `inner` fields.
    /// Otherwise, keep unchanged
    fn wrap_all(self, ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        match self {
            Self::FfiUnsafe(this) => {
                let unsafeties = this
                    .into_iter()
                    .map(|reason| FfiUnsafeReason {
                        ty,
                        help: help.clone(),
                        reason: note.clone(),
                        inner: Some(Box::new(reason)),
                    })
                    .collect();
                Self::FfiUnsafe(unsafeties)
            }
            r @ _ => r,
        }
    }
    /// If the FfiPhantom variant, turns it into a FfiUnsafe version.
    /// Otherwise, keep unchanged.
    fn forbid_phantom(self) -> Self {
        match self {
            Self::FfiSafe | Self::FfiUnsafe(..) => self,
            Self::FfiPhantom(ty) => Self::FfiUnsafe(vec![FfiUnsafeReason {
                ty,
                reason: fluent::lint_improper_ctypes_only_phantomdata,
                help: None,
                inner: None,
            }]),
        }
    }
}

impl<'tcx> std::ops::AddAssign<FfiResult<'tcx>> for FfiResult<'tcx> {
    fn add_assign(&mut self, mut other: Self) {
        // note: we shouldn't really encounter FfiPhantoms here, they should be dealt with beforehand
        // still, this function deals with them in a reasonable way, I think

        // this function is awful to look but that's because matching mutable references consumes them (?!)
        // the function itself imitates the following piece of non-compiling code:

        // match (self, other) {
        //     (Self::FfiUnsafe(_), _) => {
        //         // nothing to do
        //     },
        //     (_, Self::FfiUnsafe(_)) => {
        //         *self = other;
        //     },
        //     (Self::FfiPhantom(ref ty1),Self::FfiPhantom(ty2)) => {
        //         println!("whoops, both FfiPhantom: self({:?}) += other({:?})", ty1, ty2);
        //     },
        //     (Self::FfiSafe,Self::FfiPhantom(_)) => {
        //         *self = other;
        //     },
        //     (_, Self::FfiSafe) => {
        //         // nothing to do
        //     },
        // }

        let s_disc = std::mem::discriminant(self);
        let o_disc = std::mem::discriminant(&other);
        if s_disc == o_disc {
            match (self, &mut other) {
                (Self::FfiUnsafe(ref mut s_inner), Self::FfiUnsafe(ref mut o_inner)) => {
                    s_inner.append(o_inner);
                }
                (Self::FfiPhantom(ref ty1), Self::FfiPhantom(ty2)) => {
                    debug!("whoops: both FfiPhantom, self({:?}) += other({:?})", ty1, ty2);
                }
                (Self::FfiSafe, Self::FfiSafe) => {}
                _ => unreachable!(),
            }
        } else {
            if let Self::FfiUnsafe(_) = self {
                return;
            }
            match other {
                Self::FfiUnsafe(o_inner) => {
                    // self is Safe or Phantom: Unsafe wins
                    *self = Self::FfiUnsafe(o_inner);
                }
                Self::FfiSafe => {
                    // self is always "wins"
                    return;
                }
                Self::FfiPhantom(o_inner) => {
                    // self is Safe: Phantom wins
                    *self = Self::FfiPhantom(o_inner);
                }
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

/// Determine if a type is sized or not, and wether it affects references/pointers/boxes to it
#[derive(Clone, Copy)]
enum TypeSizedness {
    /// type of definite size (pointers are C-compatible)
    Definite,
    /// unsized type because it includes an opaque/foreign type (pointers are C-compatible)
    UnsizedWithExternType,
    /// unsized type for other reasons (slice, string, dyn Trait, closure, ...) (pointers are not C-compatible)
    UnsizedWithMetadata,
    /// not known, usually for placeholder types (Self in non-impl trait functions, type parameters, aliases, the like)
    NotYetKnown,
}

/// what type indirection points to a given type
#[derive(Clone, Copy)]
enum IndirectionType {
    /// box (valid non-null pointer, owns pointee)
    Box,
    /// ref (valid non-null pointer, borrows pointee)
    Ref,
    /// raw pointer (not necessarily non-null or valid. no info on ownership)
    RawPtr,
}

/// Is this type unsized because it contains (or is) a foreign type?
/// (Returns Err if the type happens to be sized after all)
fn get_type_sizedness<'tcx, 'a>(cx: &'a LateContext<'tcx>, ty: Ty<'tcx>) -> TypeSizedness {
    let tcx = cx.tcx;

    // note that sizedness is unrelated to inhabitedness
    if ty.is_sized(tcx, cx.typing_env()) {
        //let is_inh = ty.is_privately_uninhabited(tcx, cx.typing_env());
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
                let item_ty = normalize_if_possible(cx, item_ty);
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

            ty::Alias(ty::Weak, ..)
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

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum CTypesVisitorState {
    // bitflags:
    // 0010: static
    // 0100: function return
    // 1000: used in declared function
    StaticTy = 0b0010,
    ArgumentTyInDefinition = 0b1000,
    ReturnTyInDefinition = 0b1100,
    ArgumentTyInDeclaration = 0b0000,
    ReturnTyInDeclaration = 0b0100,
}

impl CTypesVisitorState {
    /// wether the type is used (directly or not) in a static variable
    fn is_in_static(self) -> bool {
        ((self as u8) & 0b0010) != 0
    }
    /// wether the type is used (directly or not) in a function, in return position
    fn is_in_function_return(self) -> bool {
        let ret = ((self as u8) & 0b0100) != 0;
        #[cfg(debug_assertions)]
        if ret {
            assert!(!self.is_in_static());
        }
        ret
    }
    /// wether the type is used (directly or not) in a defined function
    /// in other words, wether or not we allow non-FFI-safe types behind a C pointer,
    /// to be treated as an opaque type on the other side of the FFI boundary
    fn is_in_defined_function(self) -> bool {
        let ret = ((self as u8) & 0b1000) != 0;
        #[cfg(debug_assertions)]
        if ret {
            assert!(!self.is_in_static());
        }
        ret
    }

    /// wether the value for that type might come from the non-rust side of a FFI boundary
    fn value_may_be_unchecked(self) -> bool {
        // function declarations are assumed to be rust-caller, non-rust-callee
        // function definitions are assumed to be maybe-not-rust-caller, rust-callee
        // FnPtrs are... well, nothing's certain about anything. (FIXME need more flags in enum?)
        // Same with statics.
        if self.is_in_static() {
            true
        } else if self.is_in_defined_function() {
            !self.is_in_function_return()
        } else {
            self.is_in_function_return()
        }
    }
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Checks wether an `extern "ABI" fn` function pointer is indeed FFI-safe to call
    fn visit_fnptr(&mut self, mode: CItemKind, ty: Ty<'tcx>, sig: Sig<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;
        debug_assert!(!fn_abi_is_internal(sig.abi()));
        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);
        let state = match mode {
            CItemKind::Declaration => CTypesVisitorState::ArgumentTyInDeclaration,
            CItemKind::Definition => CTypesVisitorState::ArgumentTyInDefinition,
        };

        let mut all_ffires = FfiSafe;

        for arg in sig.inputs() {
            let ffi_res = self.visit_type(state, None, *arg);
            all_ffires += ffi_res.forbid_phantom().wrap_all(
                ty,
                fluent::lint_improper_ctypes_fnptr_indirect_reason,
                None,
            );
        }

        let ret_ty = sig.output();
        let state = match mode {
            CItemKind::Declaration => CTypesVisitorState::ReturnTyInDeclaration,
            CItemKind::Definition => CTypesVisitorState::ReturnTyInDefinition,
        };

        let ffi_res = self.visit_type(state, None, ret_ty);
        all_ffires += ffi_res.forbid_phantom().wrap_all(
            ty,
            fluent::lint_improper_ctypes_fnptr_indirect_reason,
            None,
        );

        all_ffires
    }

    /// Checks if a simple numeric (int, float) type has an actual portable definition
    /// for the compile target
    fn visit_numeric(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        // FIXME: for now, this is very incomplete, and seems to assume a x86_64 target
        match ty.kind() {
            ty::Int(ty::IntTy::I128) | ty::Uint(ty::UintTy::U128) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_128bit, None)
            }
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => FfiResult::FfiSafe,
            _ => bug!("visit_numeric is to be called with numeric (int, float) types"),
        }
    }

    /// Return the right help for Cstring and Cstr-linked unsafety
    fn visit_cstr(&mut self, outer_ty: Option<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
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

    /// Checks if the given indirection (box,ref,pointer) is "ffi-safe"
    fn visit_indirection(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
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
                            reasons.first_mut().unwrap().ty = ty;
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
        // - does the pointer type contain a non-zero assumption, but a value given by non-rust code?
        // this block deals with the first two.
        let mut ffi_res = match get_type_sizedness(self.cx, inner_ty) {
            TypeSizedness::UnsizedWithExternType | TypeSizedness::Definite => {
                // there's a nuance on what this lint should do for
                // function definitions (`extern "C" fn fn_name(...) {...}`)
                // versus declarations (`unsafe extern "C" {fn fn_name(...);}`).
                // This is touched upon in https://github.com/rust-lang/rust/issues/66220
                // and https://github.com/rust-lang/rust/pull/72700
                //
                // The big question is: what does "ABI safety" mean? if you have something translated to a C pointer
                // (which has a stable layout) but points to FFI-unsafe type, is it safe?
                // On one hand, the function's ABI will match that of a similar C-declared function API,
                // on the other, dereferencing the pointer on the other side of the FFI boundary will be painful.
                // In this code, the opinion on is split between function declarations and function definitions,
                // with the idea that at least one side of the FFI boundary needs to treat the pointee as an opaque type.
                // For declarations, we see this as unsafe, but for definitions, we see this as safe.
                //
                // For extern function declarations, the actual definition of the function is written somewhere else,
                // meaning the declaration is free to express this opaqueness with an extern type (opaque caller-side) or a std::ffi::c_void (opaque callee-side)
                // (or other possibly better tricks, see https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs)
                // For extern function definitions, however, in the case where the type is opaque caller-side, it is not opaque callee-side,
                // and having the full type information is necessary to compile the function.
                if state.is_in_defined_function() {
                    FfiResult::FfiSafe
                } else {
                    return self.visit_type(state, Some(ty), inner_ty).forbid_phantom().wrap_all(
                        ty,
                        fluent::lint_improper_ctypes_sized_ptr_to_unsafe_type,
                        None,
                    );
                }
            }
            TypeSizedness::NotYetKnown => {
                // types with sizedness NotYetKnown:
                // - Type params (with `variable: impl Trait` shorthand or not)
                //   (function definitions only, let's see how this interacts with monomorphisation)
                // - Self in trait functions/methods
                //   (FIXME note: function 'declarations' there should be treated as definitions)
                // - Opaque return types
                //   (always FFI-unsafe)
                // - non-exhaustive structs/enums/unions from other crates
                //   (always FFI-unsafe)
                // (for the three first, this is unless there is a `+Sized` bound involved)
                //
                // FIXME: on a side note, we should separate 'true' declarations (non-rust code),
                // 'fake' declarations (in traits, needed to be implemented elsewhere), and definitions.
                // (for instance, definitions should worry about &self with Self:?Sized, but fake declarations shouldn't)

                // wether they are FFI-safe or not does not depend on the indirections involved (&Self, &T, Box<impl Trait>),
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
        };

        // and now the third concern (does the pointer type contain a non-zero assumption, and is the value given by non-rust code?)
        ffi_res += if state.value_may_be_unchecked() {
            let has_nonnull_assumption = match indirection_type {
                IndirectionType::RawPtr => false,
                IndirectionType::Ref | IndirectionType::Box => true,
            };
            let has_optionlike_wrapper = if let Some(outer_ty) = outer_ty {
                super::is_outer_optionlike_around_ty(self.cx, outer_ty, ty)
            } else {
                false
            };

            if has_nonnull_assumption && !has_optionlike_wrapper {
                FfiResult::new_with_reason(
                    ty,
                    fluent::lint_improper_ctypes_ptr_validity_reason,
                    Some(fluent::lint_improper_ctypes_ptr_validity_help),
                )
            } else {
                FfiResult::FfiSafe
            }
        } else {
            FfiResult::FfiSafe
        };

        ffi_res
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn visit_variant_fields(
        &mut self,
        state: CTypesVisitorState,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        let transparent_with_all_zst_fields = if def.repr().transparent() {
            if let Some(field) = super::transparent_newtype_field(self.cx.tcx, variant) {
                // Transparent newtypes have at most one non-ZST field which needs to be checked..
                let field_ty = get_type_from_field(self.cx, field, args);
                let ffi_res = self.visit_type(state, Some(ty), field_ty);
                debug_assert!(!matches!(
                    // checking that this is not an FfiUnsafe due to an unit type:
                    // visit_type should be smart enough to not consider it unsafe if called from here
                    ffi_res,
                    FfiUnsafe(ref reasons)
                    if matches!(
                        (reasons.len(),reasons.first()),
                        (1,Some(FfiUnsafeReason{ty,..})) if ty.is_unit()
                    )
                ));
                return ffi_res;
            } else {
                // ..or have only ZST fields, which is FFI-unsafe (unless those fields are all
                // `PhantomData`).
                true
            }
        } else {
            false
        };

        let mut all_ffires = FfiSafe;
        // We can't completely trust `repr(C)` markings, so make sure the fields are actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        for field in &variant.fields {
            let field_ty = get_type_from_field(self.cx, field, args);
            all_phantom &= match self.visit_type(state, Some(ty), field_ty) {
                FfiPhantom(..) => true,
                r @ (FfiUnsafe { .. } | FfiSafe) => {
                    all_ffires += r;
                    false
                }
            }
        }

        if matches!(all_ffires, FfiUnsafe(..)) {
            all_ffires.wrap_all(ty, fluent::lint_improper_ctypes_struct_dueto, None)
        } else if all_phantom {
            FfiPhantom(ty)
        } else if transparent_with_all_zst_fields {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_struct_zst, None)
        } else {
            FfiSafe
        }
    }

    fn visit_struct_union(
        &mut self,
        state: CTypesVisitorState,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
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
                    Some(fluent::lint_improper_ctypes_union_layout_help)
                },
            );
        }

        let is_non_exhaustive = def.non_enum_variant().is_field_list_non_exhaustive();
        if is_non_exhaustive && !def.did().is_local() {
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

        if def.non_enum_variant().fields.is_empty() {
            return FfiResult::new_with_reason(
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
            );
        }

        self.visit_variant_fields(state, ty, def, def.non_enum_variant(), args)
    }

    fn visit_enum(
        &mut self,
        state: CTypesVisitorState,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        debug_assert!(matches!(def.adt_kind(), AdtKind::Enum));
        use FfiResult::*;

        if def.variants().is_empty() {
            // Empty enums are implicitely handled as the never type:
            // FIXME think about the FFI-safety of functions that use that
            return FfiSafe;
        }
        // Check for a repr() attribute to specify the size of the
        // discriminant.
        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none() {
            // Special-case types like `Option<extern fn()>` and `Result<extern fn(), ()>`
            if let Some(inner_ty) = repr_nullable_ptr(
                self.cx.tcx,
                self.cx.typing_env(),
                ty,
                if state.is_in_defined_function() {
                    CItemKind::Definition
                } else {
                    CItemKind::Declaration
                },
            ) {
                return self.visit_type(state, Some(ty), inner_ty);
            }

            return FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_enum_repr_reason,
                Some(fluent::lint_improper_ctypes_enum_repr_help),
            );
        }

        let non_local_def = non_local_and_non_exhaustive(def);
        // Check the contained variants.

        let (mut nonexhaustive_flag, mut nonexhaustive_variant_flag) = (false, false);
        def.variants().iter().for_each(|variant| {
            let (nonex_enum, nonex_var) = flag_non_exhaustive_variant(non_local_def, variant);
            nonexhaustive_flag |= nonex_enum;
            nonexhaustive_variant_flag |= nonex_var;
        });

        if nonexhaustive_flag {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_non_exhaustive, None)
        } else if nonexhaustive_variant_flag {
            FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_non_exhaustive_variant,
                None,
            )
        } else {
            def.variants()
                .iter()
                .map(|variant| {
                    self.visit_variant_fields(state, ty, def, variant, args)
                        // FIXME: check that enums allow any (up to all) variants to be phantoms?
                        // (previous code says no, but I don't know why? the problem with phantoms is that they're ZSTs, right?)
                        .forbid_phantom()
                })
                .reduce(|r1, r2| r1 + r2)
                .unwrap() // always at least one variant if we hit this branch
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn visit_type(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !self.cache.insert(ty) {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, args) => {
                if let Some(inner_ty) = ty.boxed_ty() {
                    return self.visit_indirection(
                        state,
                        outer_ty,
                        ty,
                        inner_ty,
                        IndirectionType::Box,
                    );
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        // I thought CStr (not CString) could not be reached here:
                        //   - not using an indirection would cause a compile error prior to this lint
                        //   - and using one would cause the lint to catch on the indirection before reaching its pointee
                        // but for some reason one can just go and write function *pointers* like that:
                        // `type Foo = extern "C" fn(::std::ffi::CStr);`
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return self.visit_cstr(outer_ty, ty);
                        }
                        self.visit_struct_union(state, ty, def, args)
                    }
                    AdtKind::Enum => self.visit_enum(state, ty, def, args),
                }
            }

            ty::Char => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_char_reason,
                Some(fluent::lint_improper_ctypes_char_help),
            ),

            ty::Pat(pat_ty, _) => {
                if state.value_may_be_unchecked() {
                    // you would think that int-range pattern types that exclude 0 would have Option layout optimisation
                    // they don't (see tests/ui/type/pattern_types/range_patterns.stderr)
                    // so there's no need to allow Option<pattern_type!(u32 in 1..)>.
                    debug_assert!(matches!(
                        pat_ty.kind(),
                        ty::Int(..) | ty::Uint(..) | ty::Float(..)
                    ));
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_pat_intrange_reason,
                        Some(fluent::lint_improper_ctypes_pat_intrange_help),
                    )
                } else if let ty::Int(_) | ty::Uint(_) = pat_ty.kind() {
                    self.visit_numeric(pat_ty)
                } else {
                    bug!(
                        "this lint was written when pattern types could only be integers constrained to ranges"
                    )
                }
            }

            // types which likely have a stable representation, depending on the target architecture
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => self.visit_numeric(ty),

            // Primitive types with a stable representation.
            ty::Bool | ty::Never => FfiSafe,

            ty::Slice(_) => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_slice_reason,
                Some(fluent::lint_improper_ctypes_slice_help),
            ),

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
                    if let Some(outer_ty) = outer_ty {
                        match outer_ty.kind() {
                            // `()` fields are FFI-safe!
                            ty::Adt(..) => true,
                            ty::RawPtr(..) => true,
                            // most of those are not even reachable,
                            // but let's not worry about checking that here
                            _ => false,
                        }
                    } else {
                        // C functions can return void
                        state.is_in_function_return()
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
                return self.visit_indirection(
                    state,
                    outer_ty,
                    ty,
                    inner_ty,
                    IndirectionType::RawPtr,
                );
            }
            ty::Ref(_, inner_ty, _) => {
                return self.visit_indirection(state, outer_ty, ty, inner_ty, IndirectionType::Ref);
            }

            ty::Array(inner_ty, _) => {
                if outer_ty.is_none() && !state.is_in_static() {
                    // C doesn't really support passing arrays by value - the only way to pass an array by value
                    // is through a struct.
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_array_reason,
                        Some(fluent::lint_improper_ctypes_array_help),
                    )
                } else {
                    self.visit_type(state, Some(ty), inner_ty)
                }
            }

            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if fn_abi_is_internal(sig.abi()) {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_fnptr_reason,
                        Some(fluent::lint_improper_ctypes_fnptr_help),
                    )
                } else {
                    let mode = if state.is_in_defined_function() {
                        CItemKind::Definition
                    } else {
                        CItemKind::Declaration
                    };
                    self.visit_fnptr(mode, ty, sig)
                }
            }

            ty::Foreign(..) => FfiSafe,

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            ty::Alias(ty::Opaque, ..) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None)
            }

            // `extern "C" fn` functions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..)
                if state.is_in_defined_function() =>
            {
                FfiSafe
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Param(..)
            | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
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

    fn check_for_opaque_ty(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        struct ProhibitOpaqueTypes;
        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for ProhibitOpaqueTypes {
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

        if let Some(ty) = ty.visit_with(&mut ProhibitOpaqueTypes).break_value() {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None)
        } else {
            FfiResult::FfiSafe
        }
    }

    fn check_for_type(&mut self, state: CTypesVisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = normalize_if_possible(self.cx, ty);

        match self.check_for_opaque_ty(ty) {
            FfiResult::FfiSafe => (),
            ffi_res @ _ => return ffi_res,
        }
        self.visit_type(state, None, ty)
    }

    fn check_for_fnptr(&mut self, mode: CItemKind, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = normalize_if_possible(self.cx, ty);

        match self.check_for_opaque_ty(ty) {
            FfiResult::FfiSafe => (),
            ffi_res @ _ => return ffi_res,
        }

        match *ty.kind() {
            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if fn_abi_is_internal(sig.abi()) {
                    bug!(
                        "expected to inspect the type of an `extern \"ABI\"` FnPtr, not an internal-ABI one"
                    )
                } else {
                    self.visit_fnptr(mode, ty, sig)
                }
            }
            _ => bug!(
                "expected to inspect the type of an `extern \"ABI\"` FnPtr, not whtaever this is"
            ),
        }
    }
}

/// common structure for functionality that is shared
/// between ImproperCTypesDeclarations and ImproperCTypesDefinitions
struct ImproperCTypesLint<'c, 'tcx> {
    cx: &'c LateContext<'tcx>,
}

impl<'c, 'tcx> ImproperCTypesLint<'c, 'tcx> {
    fn check_arg_for_power_alignment(&mut self, ty: Ty<'tcx>) -> bool {
        // Structs (under repr(C)) follow the power alignment rule if:
        //   - the first field of the struct is a floating-point type that
        //     is greater than 4-bytes, or
        //   - the first field of the struct is an aggregate whose
        //     recursively first field is a floating-point type greater than
        //     4 bytes.
        let tcx = self.cx.tcx;
        if tcx.sess.target.os != "aix" {
            return false;
        }
        if ty.is_floating_point() && ty.primitive_size(tcx).bytes() > 4 {
            return true;
        } else if let Adt(adt_def, _) = ty.kind()
            && adt_def.is_struct()
        {
            let struct_variant = adt_def.variant(VariantIdx::ZERO);
            // Within a nested struct, all fields are examined to correctly
            // report if any fields after the nested struct within the
            // original struct are misaligned.
            for struct_field in &struct_variant.fields {
                let field_ty = tcx.type_of(struct_field.did).instantiate_identity();
                if self.check_arg_for_power_alignment(field_ty) {
                    return true;
                }
            }
        }
        return false;
    }

    fn check_struct_for_power_alignment(&mut self, item: &'tcx hir::Item<'tcx>) {
        let tcx = self.cx.tcx;
        let adt_def = tcx.adt_def(item.owner_id.to_def_id());
        if adt_def.repr().c()
            && !adt_def.repr().packed()
            && tcx.sess.target.os == "aix"
            && !adt_def.all_fields().next().is_none()
        {
            let struct_variant_data = item.expect_struct().0;
            for (index, ..) in struct_variant_data.fields().iter().enumerate() {
                // Struct fields (after the first field) are checked for the
                // power alignment rule, as fields after the first are likely
                // to be the fields that are misaligned.
                if index != 0 {
                    let first_field_def = struct_variant_data.fields()[index];
                    let def_id = first_field_def.def_id;
                    let ty = tcx.type_of(def_id).instantiate_identity();
                    if self.check_arg_for_power_alignment(ty) {
                        self.cx.emit_span_lint(
                            USES_POWER_ALIGNMENT,
                            first_field_def.span,
                            UsesPowerAlignment,
                        );
                    }
                }
            }
        }
    }

    /// Find any fn-ptr types with external ABIs in `ty`.
    ///
    /// For example, `Option<extern "C" fn()>` returns `extern "C" fn()`
    fn check_type_for_external_abi_fnptr(
        &self,
        fn_mode: CItemKind,
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
                if let hir::TyKind::BareFn(hir::BareFnTy { abi, .. }) = ty.kind
                    && !fn_abi_is_internal(*abi)
                {
                    self.spans.push(ty.span);
                }

                hir::intravisit::walk_ty(self, ty)
            }
        }

        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for FnPtrFinder<'tcx> {
            type Result = ControlFlow<Ty<'tcx>>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if let ty::FnPtr(_, hdr) = ty.kind()
                    && !fn_abi_is_internal(hdr.abi)
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
        all_types
            .map(|(fn_ptr_ty, span)| {
                // FIXME this will probably lead to error deduplication: fix this
                let mut visitor =
                    ImproperCTypesVisitor { cx: self.cx, cache: FxHashSet::default() };
                let ffi_res = visitor.check_for_fnptr(fn_mode, fn_ptr_ty);
                (span, ffi_res)
            })
            // even in function *definitions*, `FnPtr`s are always function declarations ...right?
            // (FIXME: we can't do that yet because one of rustc's crates can't compile if we do)
            .for_each(|(span, ffi_res)| self.process_ffi_result(span, ffi_res, fn_mode));
        //.drain();
    }

    /// For a function that doesn't need to be "ffi-safe", look for fn-ptr argument/return types
    /// that need to be checked for ffi-safety
    fn check_fn_for_external_abi_fnptr(
        &self,
        fn_mode: CItemKind,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = self.cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_type_for_external_abi_fnptr(fn_mode, input_hir, *input_ty);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            self.check_type_for_external_abi_fnptr(fn_mode, ret_hir, sig.output());
        }
    }

    /// Check that an extern "ABI" static variable is of a ffi-safe type
    fn check_foreign_static(&self, id: hir::OwnerId, span: Span) {
        let ty = self.cx.tcx.type_of(id).instantiate_identity();
        let mut visitor = ImproperCTypesVisitor { cx: self.cx, cache: FxHashSet::default() };
        let ffi_res = visitor.check_for_type(CTypesVisitorState::StaticTy, ty);
        self.process_ffi_result(span, ffi_res, CItemKind::Declaration);
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    fn check_foreign_fn(
        &self,
        fn_mode: CItemKind,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = self.cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let mut visitor = ImproperCTypesVisitor { cx: self.cx, cache: FxHashSet::default() };
            let visit_state = match fn_mode {
                CItemKind::Definition => CTypesVisitorState::ArgumentTyInDefinition,
                CItemKind::Declaration => CTypesVisitorState::ArgumentTyInDeclaration,
            };
            let ffi_res = visitor.check_for_type(visit_state, *input_ty);
            self.process_ffi_result(input_hir.span, ffi_res, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let mut visitor = ImproperCTypesVisitor { cx: self.cx, cache: FxHashSet::default() };
            let visit_state = match fn_mode {
                CItemKind::Definition => CTypesVisitorState::ReturnTyInDefinition,
                CItemKind::Declaration => CTypesVisitorState::ReturnTyInDeclaration,
            };
            let ffi_res = visitor.check_for_type(visit_state, sig.output());
            self.process_ffi_result(ret_hir.span, ffi_res, fn_mode);
        }
    }

    fn process_ffi_result(&self, sp: Span, res: FfiResult<'tcx>, fn_mode: CItemKind) {
        match res {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(
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
            FfiResult::FfiUnsafe(reasons) => {
                for reason in reasons {
                    let mut ffiresult_recursor = ControlFlow::Continue(&reason);
                    let mut cimproper_layers: Vec<ImproperCTypesLayer<'_>> = vec![];

                    // this whole while block converts the arbitrarily-deep
                    // FfiResult stack to an ImproperCTypesLayer Vec
                    while let ControlFlow::Continue(FfiUnsafeReason {
                        ref ty,
                        ref reason,
                        ref help,
                        ref inner,
                    }) = ffiresult_recursor
                    {
                        if let Some(layer) = cimproper_layers.last_mut() {
                            layer.inner_ty = Some(ty.clone());
                        }
                        cimproper_layers.push(ImproperCTypesLayer {
                            ty: ty.clone(),
                            inner_ty: None,
                            help: help.clone(),
                            note: reason.clone(),
                            span_note: None, // filled later
                        });

                        if let Some(inner) = inner {
                            ffiresult_recursor = ControlFlow::Continue(inner.as_ref());
                        } else {
                            ffiresult_recursor = ControlFlow::Break(());
                        }
                    }
                    // should always have at least one type
                    let last_ty = cimproper_layers.last().unwrap().ty.clone();
                    self.emit_ffi_unsafe_type_lint(last_ty, sp, cimproper_layers, fn_mode);
                }
            }
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &self,
        ty: Ty<'tcx>,
        sp: Span,
        mut reasons: Vec<ImproperCTypesLayer<'tcx>>,
        fn_mode: CItemKind,
    ) {
        let lint = match fn_mode {
            CItemKind::Declaration => IMPROPER_CTYPES,
            CItemKind::Definition => IMPROPER_CTYPES_DEFINITIONS,
        };
        let desc = match fn_mode {
            CItemKind::Declaration => "block",
            CItemKind::Definition => "fn",
        };
        for reason in reasons.iter_mut() {
            reason.span_note = if let ty::Adt(def, _) = reason.ty.kind()
                && let Some(sp) = self.cx.tcx.hir().span_if_local(def.did())
            {
                Some(sp)
            } else {
                None
            };
        }

        self.cx.emit_span_lint(lint, sp, ImproperCTypes { ty, desc, label: sp, reasons });
    }
}

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDeclarations {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        let abi = cx.tcx.hir().get_foreign_abi(it.hir_id());
        let lint = ImproperCTypesLint { cx };

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                if fn_abi_is_internal(abi) {
                    lint.check_fn_for_external_abi_fnptr(
                        CItemKind::Declaration,
                        it.owner_id.def_id,
                        sig.decl,
                    )
                } else {
                    lint.check_foreign_fn(CItemKind::Declaration, it.owner_id.def_id, sig.decl);
                }
            }
            hir::ForeignItemKind::Static(ty, _, _) if !fn_abi_is_internal(abi) => {
                lint.check_foreign_static(it.owner_id, ty.span);
            }
            hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => (),
        }
    }
}

/// `ImproperCTypesDefinitions` checks items outside of foreign items (e.g. stuff that isn't in
/// `extern "C" { }` blocks):
///
/// - `extern "<abi>" fn` definitions are checked in the same way as the
///   `ImproperCtypesDeclarations` visitor checks functions if `<abi>` is external (e.g. "C").
/// - All other items which contain types (e.g. other functions, struct definitions, etc) are
///   checked for extern fn-ptrs with external ABIs.
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDefinitions {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::Static(ty, ..)
            | hir::ItemKind::Const(ty, ..)
            | hir::ItemKind::TyAlias(ty, ..) => {
                ImproperCTypesLint { cx }.check_type_for_external_abi_fnptr(
                    CItemKind::Definition,
                    ty,
                    cx.tcx.type_of(item.owner_id).instantiate_identity(),
                );
            }
            // See `check_fn`..
            hir::ItemKind::Fn { .. } => {}
            // Structs are checked based on if they follow the power alignment
            // rule (under repr(C)).
            hir::ItemKind::Struct(..) => {
                ImproperCTypesLint { cx }.check_struct_for_power_alignment(item);
            }
            // See `check_field_def`..
            hir::ItemKind::Union(..) | hir::ItemKind::Enum(..) => {}
            // Doesn't define something that can contain a external type to be checked.
            hir::ItemKind::Impl(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::GlobalAsm(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::ExternCrate(..) => {}
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'tcx>) {
        ImproperCTypesLint { cx }.check_type_for_external_abi_fnptr(
            CItemKind::Definition,
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

        let lint = ImproperCTypesLint { cx };
        if fn_abi_is_internal(abi) {
            lint.check_fn_for_external_abi_fnptr(CItemKind::Definition, id, decl);
        } else {
            lint.check_foreign_fn(CItemKind::Definition, id, decl);
        }
    }
}
