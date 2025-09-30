use rustc_abi::{BackendRepr, FieldsShape, Scalar, Variants};
use rustc_middle::ty::layout::{
    HasTyCtxt, LayoutCx, LayoutError, LayoutOf, TyAndLayout, ValidityRequirement,
};
use rustc_middle::ty::{PseudoCanonicalInput, ScalarInt, Ty, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_span::DUMMY_SP;

use crate::const_eval::{CanAccessMutGlobal, CheckAlignment, CompileTimeMachine};
use crate::interpret::{InterpCx, MemoryKind};

/// Determines if this type permits "raw" initialization by just transmuting some memory into an
/// instance of `T`.
///
/// `init_kind` indicates if the memory is zero-initialized or left uninitialized. We assume
/// uninitialized memory is mitigated by filling it with 0x01, which reduces the chance of causing
/// LLVM UB.
///
/// By default we check whether that operation would cause *LLVM UB*, i.e., whether the LLVM IR we
/// generate has UB or not. This is a mitigation strategy, which is why we are okay with accepting
/// Rust UB as long as there is no risk of miscompilations. The `strict_init_checks` can be set to
/// do a full check against Rust UB instead (in which case we will also ignore the 0x01-filling and
/// to the full uninit check).
pub fn check_validity_requirement<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: ValidityRequirement,
    input: PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> Result<bool, &'tcx LayoutError<'tcx>> {
    let layout = tcx.layout_of(input)?;

    // There is nothing strict or lax about inhabitedness.
    if kind == ValidityRequirement::Inhabited {
        return Ok(!layout.is_uninhabited());
    }

    let layout_cx = LayoutCx::new(tcx, input.typing_env);
    if kind == ValidityRequirement::Uninit || tcx.sess.opts.unstable_opts.strict_init_checks {
        Ok(check_validity_requirement_strict(layout, &layout_cx, kind))
    } else {
        check_validity_requirement_lax(layout, &layout_cx, kind)
    }
}

/// Implements the 'strict' version of the [`check_validity_requirement`] checks; see that function
/// for details.
fn check_validity_requirement_strict<'tcx>(
    ty: TyAndLayout<'tcx>,
    cx: &LayoutCx<'tcx>,
    kind: ValidityRequirement,
) -> bool {
    let machine = CompileTimeMachine::new(CanAccessMutGlobal::No, CheckAlignment::Error);

    let mut cx = InterpCx::new(cx.tcx(), DUMMY_SP, cx.typing_env, machine);

    // It doesn't really matter which `MemoryKind` we use here, `Stack` is the least wrong.
    let allocated =
        cx.allocate(ty, MemoryKind::Stack).expect("OOM: failed to allocate for uninit check");

    if kind == ValidityRequirement::Zero {
        cx.write_bytes_ptr(
            allocated.ptr(),
            std::iter::repeat(0_u8).take(ty.layout.size().bytes_usize()),
        )
        .expect("failed to write bytes for zero valid check");
    }

    // Assume that if it failed, it's a validation failure.
    // This does *not* actually check that references are dereferenceable, but since all types that
    // require dereferenceability also require non-null, we don't actually get any false negatives
    // due to this.
    // The value we are validating is temporary and discarded at the end of this function, so
    // there is no point in resetting provenance and padding.
    cx.validate_operand(
        &allocated.into(),
        /*recursive*/ false,
        /*reset_provenance_and_padding*/ false,
    )
    .discard_err()
    .is_some()
}

/// Implements the 'lax' (default) version of the [`check_validity_requirement`] checks; see that
/// function for details.
fn check_validity_requirement_lax<'tcx>(
    this: TyAndLayout<'tcx>,
    cx: &LayoutCx<'tcx>,
    init_kind: ValidityRequirement,
) -> Result<bool, &'tcx LayoutError<'tcx>> {
    let scalar_allows_raw_init = move |s: Scalar| -> bool {
        match init_kind {
            ValidityRequirement::Inhabited => {
                bug!("ValidityRequirement::Inhabited should have been handled above")
            }
            ValidityRequirement::Zero => {
                // The range must contain 0.
                s.valid_range(cx).contains(0)
            }
            ValidityRequirement::UninitMitigated0x01Fill => {
                // The range must include an 0x01-filled buffer.
                let mut val: u128 = 0x01;
                for _ in 1..s.size(cx).bytes() {
                    // For sizes >1, repeat the 0x01.
                    val = (val << 8) | 0x01;
                }
                s.valid_range(cx).contains(val)
            }
            ValidityRequirement::Uninit => {
                bug!("ValidityRequirement::Uninit should have been handled above")
            }
        }
    };

    // Check the ABI.
    let valid = !this.is_uninhabited() // definitely UB if uninhabited
        && match this.backend_repr {
            BackendRepr::Scalar(s) => scalar_allows_raw_init(s),
            BackendRepr::ScalarPair(s1, s2) => {
                scalar_allows_raw_init(s1) && scalar_allows_raw_init(s2)
            }
            BackendRepr::SimdVector { element: s, count } => count == 0 || scalar_allows_raw_init(s),
            BackendRepr::Memory { .. } => true, // Fields are checked below.
        };
    if !valid {
        // This is definitely not okay.
        return Ok(false);
    }

    // Special magic check for references and boxes (i.e., special pointer types).
    if let Some(pointee) = this.ty.builtin_deref(false) {
        let pointee = cx.layout_of(pointee)?;
        // We need to ensure that the LLVM attributes `aligned` and `dereferenceable(size)` are satisfied.
        if pointee.align.bytes() > 1 {
            // 0x01-filling is not aligned.
            return Ok(false);
        }
        if pointee.size.bytes() > 0 {
            // A 'fake' integer pointer is not sufficiently dereferenceable.
            return Ok(false);
        }
    }

    // If we have not found an error yet, we need to recursively descend into fields.
    match &this.fields {
        FieldsShape::Primitive | FieldsShape::Union { .. } => {}
        FieldsShape::Array { .. } => {
            // Arrays never have scalar layout in LLVM, so if the array is not actually
            // accessed, there is no LLVM UB -- therefore we can skip this.
        }
        FieldsShape::Arbitrary { offsets, .. } => {
            for idx in 0..offsets.len() {
                if !check_validity_requirement_lax(this.field(cx, idx), cx, init_kind)? {
                    // We found a field that is unhappy with this kind of initialization.
                    return Ok(false);
                }
            }
        }
    }

    match &this.variants {
        Variants::Empty => return Ok(false),
        Variants::Single { .. } => {
            // All fields of this single variant have already been checked above, there is nothing
            // else to do.
        }
        Variants::Multiple { .. } => {
            // We cannot tell LLVM anything about the details of this multi-variant layout, so
            // invalid values "hidden" inside the variant cannot cause LLVM trouble.
        }
    }

    Ok(true)
}

pub(crate) fn validate_scalar_in_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    scalar: ScalarInt,
    ty: Ty<'tcx>,
) -> bool {
    let machine = CompileTimeMachine::new(CanAccessMutGlobal::No, CheckAlignment::Error);

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let mut cx = InterpCx::new(tcx, DUMMY_SP, typing_env, machine);

    let Ok(layout) = cx.layout_of(ty) else {
        bug!("could not compute layout of {scalar:?}:{ty:?}")
    };

    // It doesn't really matter which `MemoryKind` we use here, `Stack` is the least wrong.
    let allocated =
        cx.allocate(layout, MemoryKind::Stack).expect("OOM: failed to allocate for uninit check");

    cx.write_scalar(scalar, &allocated).unwrap();

    cx.validate_operand(
        &allocated.into(),
        /*recursive*/ false,
        /*reset_provenance_and_padding*/ false,
    )
    .discard_err()
    .is_some()
}
