// Not in interpret to make sure we do not use private implementation details

use crate::interpret::InterpCx;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{InterpError, InterpErrorInfo};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::{self, Ty};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;
mod valtrees;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
pub use machine::*;
pub(crate) use valtrees::{eval_to_valtree, valtree_to_const_value};

// We forbid type-level constants that contain more than `VALTREE_MAX_NODES` nodes.
const VALTREE_MAX_NODES: usize = 100000;

pub(crate) enum ValTreeCreationError {
    NodesOverflow,
    /// Values of this type, or this particular value, are not supported as valtrees.
    NonSupportedType,
    /// The value pointed to non-read-only memory, so we cannot make it a valtree.
    NotReadOnly,
    Other,
}
pub(crate) type ValTreeCreationResult<'tcx> = Result<ty::ValTree<'tcx>, ValTreeCreationError>;

impl From<InterpErrorInfo<'_>> for ValTreeCreationError {
    fn from(err: InterpErrorInfo<'_>) -> Self {
        match err.kind() {
            InterpError::MachineStop(err) => {
                let err = err.downcast_ref::<ConstEvalErrKind>().unwrap();
                match err {
                    ConstEvalErrKind::ConstAccessesMutGlobal => ValTreeCreationError::NotReadOnly,
                    _ => ValTreeCreationError::Other,
                }
            }
            _ => ValTreeCreationError::Other,
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn try_destructure_mir_constant_for_user_output<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    val: mir::ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> Option<mir::DestructuredConstant<'tcx>> {
    let param_env = ty::ParamEnv::reveal_all();
    let ecx = mk_eval_cx(tcx.tcx, tcx.span, param_env, CanAccessMutGlobal::No);
    let op = ecx.const_val_to_op(val, ty, None).ok()?;

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match ty.kind() {
        ty::Array(_, len) => (len.eval_target_usize(tcx.tcx, param_env) as usize, None, op),
        ty::Adt(def, _) if def.variants().is_empty() => {
            return None;
        }
        ty::Adt(def, _) => {
            let variant = ecx.read_discriminant(&op).ok()?;
            let down = ecx.project_downcast(&op, variant).ok()?;
            (def.variants()[variant].fields.len(), Some(variant), down)
        }
        ty::Tuple(args) => (args.len(), None, op),
        _ => bug!("cannot destructure mir constant {:?}", val),
    };

    let fields_iter = (0..field_count)
        .map(|i| {
            let field_op = ecx.project_field(&down, i).ok()?;
            let val = op_to_const(&ecx, &field_op, /* for diagnostics */ true);
            Some((val, field_op.layout.ty))
        })
        .collect::<Option<Vec<_>>>()?;
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    Some(mir::DestructuredConstant { variant, fields })
}
