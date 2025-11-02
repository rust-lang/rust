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
    self, Adt, AdtDef, AdtKind, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt,
};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use tracing::debug;

use super::repr_nullable_ptr;
use crate::lints::{ImproperCTypes, UsesPowerAlignment};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe extern "C" {
    ///     static STATIC: String;
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
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint! {
    /// The `improper_ctypes_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
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
    IMPROPER_CTYPES_DEFINITIONS,
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
    USES_POWER_ALIGNMENT
]);

/// Check a variant of a non-exhaustive enum for improper ctypes
///
/// We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
/// This includes linting, on a best-effort basis. There are valid additions that are unlikely.
///
/// Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
/// so we don't need the lint to account for it.
/// e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
pub(crate) fn check_non_exhaustive_variant(
    non_exhaustive_variant_list: bool,
    variant: &ty::VariantDef,
) -> ControlFlow<DiagMessage, ()> {
    // non_exhaustive suggests it is possible that someone might break ABI
    // see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
    // so warn on complex enums being used outside their crate
    if non_exhaustive_variant_list {
        // which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)
        if variant_has_complex_ctor(variant) {
            return ControlFlow::Break(fluent::lint_improper_ctypes_non_exhaustive);
        }
    }

    if variant.field_list_has_applicable_non_exhaustive() {
        return ControlFlow::Break(fluent::lint_improper_ctypes_non_exhaustive_variant);
    }

    ControlFlow::Continue(())
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

#[derive(Clone, Copy)]
enum CItemKind {
    Declaration,
    Definition,
}

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe { ty: Ty<'tcx>, reason: DiagMessage, help: Option<DiagMessage> },
}

/// The result when a type has been checked but perhaps not completely. `None` indicates that
/// FFI safety/unsafety has not yet been determined, `Some(res)` indicates that the safety/unsafety
/// in the `FfiResult` is final.
type PartialFfiResult<'tcx> = Option<FfiResult<'tcx>>;

bitflags! {
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

    /// Get the proper visitor state for a given function's arguments.
    fn argument_from_fnmode(fn_mode: CItemKind) -> Self {
        match fn_mode {
            CItemKind::Definition => VisitorState::ARGUMENT_TY_IN_DEFINITION,
            CItemKind::Declaration => VisitorState::ARGUMENT_TY_IN_DECLARATION,
        }
    }

    /// Get the proper visitor state for a given function's return type.
    fn return_from_fnmode(fn_mode: CItemKind) -> Self {
        match fn_mode {
            CItemKind::Definition => VisitorState::RETURN_TY_IN_DEFINITION,
            CItemKind::Declaration => VisitorState::RETURN_TY_IN_DECLARATION,
        }
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

    /// Whether the type is used (directly or not) in a function pointer type.
    /// Here, we also allow non-FFI-safe types behind a C pointer,
    /// to be treated as an opaque type on the other side of the FFI boundary.
    fn is_in_fnptr(self) -> bool {
        self.contains(Self::THEORETICAL) && self.is_in_function()
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
    /// add a types-in-check cache.
    cache: FxHashSet<Ty<'tcx>>,
    /// The original type being checked, before we recursed
    /// to any other types it contains.
    base_ty: Ty<'tcx>,
    base_fn_mode: CItemKind,
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, base_ty: Ty<'tcx>, base_fn_mode: CItemKind) -> Self {
        Self { cx, base_ty, base_fn_mode, cache: FxHashSet::default() }
    }

    /// Checks if the given field's type is "ffi-safe".
    fn check_field_type_for_ffi(
        &mut self,
        state: VisitorState,
        field: &ty::FieldDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        let field_ty = field.ty(self.cx.tcx, args);
        let field_ty = self
            .cx
            .tcx
            .try_normalize_erasing_regions(self.cx.typing_env(), field_ty)
            .unwrap_or(field_ty);
        self.visit_type(state, field_ty)
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn check_variant_for_ffi(
        &mut self,
        state: VisitorState,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        let transparent_with_all_zst_fields = if def.repr().transparent() {
            if let Some(field) = super::transparent_newtype_field(self.cx.tcx, variant) {
                // Transparent newtypes have at most one non-ZST field which needs to be checked..
                match self.check_field_type_for_ffi(state, field, args) {
                    FfiUnsafe { ty, .. } if ty.is_unit() => (),
                    r => return r,
                }

                false
            } else {
                // ..or have only ZST fields, which is FFI-unsafe (unless those fields are all
                // `PhantomData`).
                true
            }
        } else {
            false
        };

        // We can't completely trust `repr(C)` markings, so make sure the fields are actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        for field in &variant.fields {
            all_phantom &= match self.check_field_type_for_ffi(state, field, args) {
                FfiSafe => false,
                // `()` fields are FFI-safe!
                FfiUnsafe { ty, .. } if ty.is_unit() => false,
                FfiPhantom(..) => true,
                r @ FfiUnsafe { .. } => return r,
            }
        }

        if all_phantom {
            FfiPhantom(ty)
        } else if transparent_with_all_zst_fields {
            FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_struct_zst, help: None }
        } else {
            FfiSafe
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn visit_type(&mut self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
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
                if let Some(boxed) = ty.boxed_ty()
                    && (
                        // FIXME(ctypes): this logic is broken, but it still fits the current tests
                        state.is_in_defined_function()
                            || (state.is_in_fnptr()
                                && matches!(self.base_fn_mode, CItemKind::Definition))
                    )
                {
                    if boxed.is_sized(tcx, self.cx.typing_env()) {
                        return FfiSafe;
                    } else {
                        return FfiUnsafe {
                            ty,
                            reason: fluent::lint_improper_ctypes_box,
                            help: None,
                        };
                    }
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                            && !self.base_ty.is_mutable_ptr()
                        {
                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_cstr_reason,
                                help: Some(fluent::lint_improper_ctypes_cstr_help),
                            };
                        }

                        if !def.repr().c() && !def.repr().transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_layout_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_layout_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_layout_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_layout_help)
                                },
                            };
                        }

                        if def.non_enum_variant().field_list_has_applicable_non_exhaustive() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_non_exhaustive
                                } else {
                                    fluent::lint_improper_ctypes_union_non_exhaustive
                                },
                                help: None,
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_fieldless_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_fieldless_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_fieldless_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_fieldless_help)
                                },
                            };
                        }

                        self.check_variant_for_ffi(state, ty, def, def.non_enum_variant(), args)
                    }
                    AdtKind::Enum => {
                        if def.variants().is_empty() {
                            // Empty enums are okay... although sort of useless.
                            return FfiSafe;
                        }
                        // Check for a repr() attribute to specify the size of the
                        // discriminant.
                        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none()
                        {
                            // Special-case types like `Option<extern fn()>` and `Result<extern fn(), ()>`
                            if let Some(ty) =
                                repr_nullable_ptr(self.cx.tcx, self.cx.typing_env(), ty)
                            {
                                return self.visit_type(state, ty);
                            }

                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_enum_repr_reason,
                                help: Some(fluent::lint_improper_ctypes_enum_repr_help),
                            };
                        }

                        let non_exhaustive = def.variant_list_has_applicable_non_exhaustive();
                        // Check the contained variants.
                        let ret = def.variants().iter().try_for_each(|variant| {
                            check_non_exhaustive_variant(non_exhaustive, variant)
                                .map_break(|reason| FfiUnsafe { ty, reason, help: None })?;

                            match self.check_variant_for_ffi(state, ty, def, variant, args) {
                                FfiSafe => ControlFlow::Continue(()),
                                r => ControlFlow::Break(r),
                            }
                        });
                        if let ControlFlow::Break(result) = ret {
                            return result;
                        }

                        FfiSafe
                    }
                }
            }

            ty::Char => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_char_reason,
                help: Some(fluent::lint_improper_ctypes_char_help),
            },

            // It's just extra invariants on the type that you need to uphold,
            // but only the base type is relevant for being representable in FFI.
            ty::Pat(base, ..) => self.visit_type(state, base),

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_slice_reason,
                help: Some(fluent::lint_improper_ctypes_slice_help),
            },

            ty::Dynamic(..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_dyn, help: None }
            }

            ty::Str => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_str_reason,
                help: Some(fluent::lint_improper_ctypes_str_help),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_tuple_reason,
                help: Some(fluent::lint_improper_ctypes_tuple_help),
            },

            ty::RawPtr(ty, _) | ty::Ref(_, ty, _)
                if {
                    (state.is_in_defined_function() || state.is_in_fnptr())
                        && ty.is_sized(self.cx.tcx, self.cx.typing_env())
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty, _)
                if match ty.kind() {
                    ty::Tuple(tuple) => tuple.is_empty(),
                    _ => false,
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty, _) | ty::Ref(_, ty, _) => self.visit_type(state, ty),

            ty::Array(inner_ty, _) => self.visit_type(state, inner_ty),

            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if sig.abi().is_rustic_abi() {
                    return FfiUnsafe {
                        ty,
                        reason: fluent::lint_improper_ctypes_fnptr_reason,
                        help: Some(fluent::lint_improper_ctypes_fnptr_help),
                    };
                }

                let sig = tcx.instantiate_bound_regions_with_erased(sig);
                for arg in sig.inputs() {
                    match self.visit_type(VisitorState::ARGUMENT_TY_IN_FNPTR, *arg) {
                        FfiSafe => {}
                        r => return r,
                    }
                }

                let ret_ty = sig.output();
                if ret_ty.is_unit() {
                    return FfiSafe;
                }

                self.visit_type(VisitorState::RETURN_TY_IN_FNPTR, ret_ty)
            }

            ty::Foreign(..) => FfiSafe,

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            ty::Alias(ty::Opaque, ..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_opaque, help: None }
            }

            // `extern "C" fn` functions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..)
                if state.can_expect_ty_params() =>
            {
                FfiSafe
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Param(..)
            | ty::Alias(ty::Projection | ty::Inherent | ty::Free, ..)
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

        if let Some(ty) = self
            .cx
            .tcx
            .try_normalize_erasing_regions(self.cx.typing_env(), ty)
            .unwrap_or(ty)
            .visit_with(&mut ProhibitOpaqueTypes)
            .break_value()
        {
            Some(FfiResult::FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_opaque,
                help: None,
            })
        } else {
            None
        }
    }

    /// Check if the type is array and emit an unsafe type lint.
    fn check_for_array_ty(&mut self, ty: Ty<'tcx>) -> PartialFfiResult<'tcx> {
        if let ty::Array(..) = ty.kind() {
            Some(FfiResult::FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_array_reason,
                help: Some(fluent::lint_improper_ctypes_array_help),
            })
        } else {
            None
        }
    }

    /// Determine the FFI-safety of a single (MIR) type, given the context of how it is used.
    fn check_type(&mut self, state: VisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        if let Some(res) = self.visit_for_opaque_ty(ty) {
            return res;
        }

        let ty = self.cx.tcx.try_normalize_erasing_regions(self.cx.typing_env(), ty).unwrap_or(ty);

        // C doesn't really support passing arrays by value - the only way to pass an array by value
        // is through a struct. So, first test that the top level isn't an array, and then
        // recursively check the types inside.
        if state.is_in_function() {
            if let Some(res) = self.check_for_array_ty(ty) {
                return res;
            }
        }

        // Don't report FFI errors for unit return types. This check exists here, and not in
        // the caller (where it would make more sense) so that normalization has definitely
        // happened.
        if state.is_in_function_return() && ty.is_unit() {
            return FfiResult::FfiSafe;
        }

        self.visit_type(state, ty)
    }
}

impl<'tcx> ImproperCTypesLint {
    /// Find any fn-ptr types with external ABIs in `ty`, and FFI-checks them.
    /// For example, `Option<extern "C" fn()>` FFI-checks `extern "C" fn()`.
    fn check_type_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        state: VisitorState,
        hir_ty: &hir::Ty<'tcx>,
        ty: Ty<'tcx>,
        fn_mode: CItemKind,
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
            let mut visitor = ImproperCTypesVisitor::new(cx, fn_ptr_ty, fn_mode);
            // FIXME(ctypes): make a check_for_fnptr
            let ffi_res = visitor.check_type(state, fn_ptr_ty);

            self.process_ffi_result(cx, span, ffi_res, fn_mode);
        }
    }

    /// Regardless of a function's need to be "ffi-safe", look for fn-ptr argument/return types
    /// that need to be checked for ffi-safety.
    fn check_fn_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_mode: CItemKind,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let state = VisitorState::argument_from_fnmode(fn_mode);
            self.check_type_for_external_abi_fnptr(cx, state, input_hir, *input_ty, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let state = VisitorState::return_from_fnmode(fn_mode);
            self.check_type_for_external_abi_fnptr(cx, state, ret_hir, sig.output(), fn_mode);
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

    fn check_foreign_static(&mut self, cx: &LateContext<'tcx>, id: hir::OwnerId, span: Span) {
        let ty = cx.tcx.type_of(id).instantiate_identity();
        let mut visitor = ImproperCTypesVisitor::new(cx, ty, CItemKind::Declaration);
        let ffi_res = visitor.check_type(VisitorState::STATIC_TY, ty);
        self.process_ffi_result(cx, span, ffi_res, CItemKind::Declaration);
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
            let state = VisitorState::argument_from_fnmode(fn_mode);
            let mut visitor = ImproperCTypesVisitor::new(cx, *input_ty, fn_mode);
            let ffi_res = visitor.check_type(state, *input_ty);
            self.process_ffi_result(cx, input_hir.span, ffi_res, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let state = VisitorState::return_from_fnmode(fn_mode);
            let mut visitor = ImproperCTypesVisitor::new(cx, sig.output(), fn_mode);
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
                    ty,
                    sp,
                    fluent::lint_improper_ctypes_only_phantomdata,
                    None,
                    fn_mode,
                );
            }
            FfiResult::FfiUnsafe { ty, reason, help } => {
                self.emit_ffi_unsafe_type_lint(cx, ty, sp, reason, help, fn_mode);
            }
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &self,
        cx: &LateContext<'tcx>,
        ty: Ty<'tcx>,
        sp: Span,
        note: DiagMessage,
        help: Option<DiagMessage>,
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
        let span_note = if let ty::Adt(def, _) = ty.kind()
            && let Some(sp) = cx.tcx.hir_span_if_local(def.did())
        {
            Some(sp)
        } else {
            None
        };
        cx.emit_span_lint(lint, sp, ImproperCTypes { ty, desc, label: sp, help, note, span_note });
    }
}

/// `ImproperCTypesDefinitions` checks items outside of foreign items (e.g. stuff that isn't in
/// `extern "C" { }` blocks):
///
/// - `extern "<abi>" fn` definitions are checked in the same way as the
///   `ImproperCtypesDeclarations` visitor checks functions if `<abi>` is external (e.g. "C").
/// - All other items which contain types (e.g. other functions, struct definitions, etc) are
///   checked for extern fn-ptrs with external ABIs.
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesLint {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        let abi = cx.tcx.hir_get_foreign_abi(it.hir_id());

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                // fnptrs are a special case, they always need to be treated as
                // "the element rendered unsafe" because their unsafety doesn't affect
                // their surroundings, and their type is often declared inline
                if !abi.is_rustic_abi() {
                    self.check_foreign_fn(cx, CItemKind::Declaration, it.owner_id.def_id, sig.decl);
                } else {
                    self.check_fn_for_external_abi_fnptr(
                        cx,
                        CItemKind::Declaration,
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
                    VisitorState::STATIC_TY,
                    ty,
                    cx.tcx.type_of(item.owner_id).instantiate_identity(),
                    CItemKind::Definition,
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
            VisitorState::STATIC_TY,
            field.ty,
            cx.tcx.type_of(field.def_id).instantiate_identity(),
            CItemKind::Definition,
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
        if !abi.is_rustic_abi() {
            self.check_foreign_fn(cx, CItemKind::Definition, id, decl);
        } else {
            self.check_fn_for_external_abi_fnptr(cx, CItemKind::Definition, id, decl);
        }
    }
}
