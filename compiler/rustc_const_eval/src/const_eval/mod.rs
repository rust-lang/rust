// Not in interpret to make sure we do not use private implementation details

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::DUMMY_SP;
use tracing::instrument;

use crate::interpret::InterpCx;

mod dummy_machine;
mod error;
mod eval_queries;
mod fn_queries;
mod machine;
mod valtrees;

pub use self::dummy_machine::*;
pub use self::error::*;
pub use self::eval_queries::*;
pub use self::fn_queries::*;
pub use self::machine::*;
pub(crate) use self::valtrees::{eval_to_valtree, valtree_to_const_value};

// We forbid type-level constants that contain more than `VALTREE_MAX_NODES` nodes.
const VALTREE_MAX_NODES: usize = 100000;

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn try_destructure_mir_constant_for_user_output<'tcx>(
    tcx: TyCtxt<'tcx>,
    val: mir::ConstValue,
    ty: Ty<'tcx>,
) -> Option<mir::DestructuredConstant<'tcx>> {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    // FIXME: use a proper span here?
    let (ecx, op) = mk_eval_cx_for_const_val(tcx.at(rustc_span::DUMMY_SP), typing_env, val, ty)?;

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match ty.kind() {
        ty::Array(_, len) => (len.try_to_target_usize(tcx)? as usize, None, op),
        ty::Adt(def, _) if def.variants().is_empty() => {
            return None;
        }
        ty::Adt(def, _) => {
            let variant = ecx.read_discriminant(&op).discard_err()?;
            let down = ecx.project_downcast(&op, variant).discard_err()?;
            (def.variants()[variant].fields.len(), Some(variant), down)
        }
        ty::Tuple(args) => (args.len(), None, op),
        _ => bug!("cannot destructure mir constant {:?}", val),
    };

    let fields_iter = (0..field_count)
        .map(|i| {
            let field_op = ecx.project_field(&down, FieldIdx::from_usize(i)).discard_err()?;
            let val = op_to_const(&ecx, &field_op, /* for diagnostics */ true);
            Some((val, field_op.layout.ty))
        })
        .collect::<Option<Vec<_>>>()?;
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    Some(mir::DestructuredConstant { variant, fields })
}

/// Computes the tag (if any) for a given type and variant.
#[instrument(skip(tcx), level = "debug")]
pub fn tag_for_variant_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, (Ty<'tcx>, VariantIdx)>,
) -> Option<ty::ScalarInt> {
    let (ty, variant_index) = key.value;
    assert!(ty.is_enum());

    let ecx = InterpCx::new(tcx, DUMMY_SP, key.typing_env, crate::const_eval::DummyMachine);

    let layout = ecx.layout_of(ty).unwrap();
    ecx.tag_for_variant(layout, variant_index).unwrap().map(|(tag, _tag_field)| tag)
}
