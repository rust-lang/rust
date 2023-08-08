// Not in interpret to make sure we do not use private implementation details

use crate::errors::MaxNumNodesInConstErr;
use crate::interpret::{intern_const_alloc_recursive, ConstValue, InternKind, InterpCx, Scalar};
use rustc_middle::mir;
use rustc_middle::mir::interpret::{EvalToValTreeResult, GlobalId};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;
mod valtrees;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
pub use machine::*;
pub(crate) use valtrees::{const_to_valtree_inner, valtree_to_const_value};

pub(crate) fn const_caller_location(
    tcx: TyCtxt<'_>,
    (file, line, col): (Symbol, u32, u32),
) -> ConstValue<'_> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), CanAccessStatics::No);

    let loc_place = ecx.alloc_caller_location(file, line, col);
    if intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &loc_place).is_err() {
        bug!("intern_const_alloc_recursive should not error in this case")
    }
    ConstValue::Scalar(Scalar::from_maybe_pointer(loc_place.ptr, &tcx))
}

// We forbid type-level constants that contain more than `VALTREE_MAX_NODES` nodes.
const VALTREE_MAX_NODES: usize = 100000;

pub(crate) enum ValTreeCreationError {
    NodesOverflow,
    NonSupportedType,
    Other,
}
pub(crate) type ValTreeCreationResult<'tcx> = Result<ty::ValTree<'tcx>, ValTreeCreationError>;

/// Evaluates a constant and turns it into a type-level constant value.
pub(crate) fn eval_to_valtree<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cid: GlobalId<'tcx>,
) -> EvalToValTreeResult<'tcx> {
    let const_alloc = tcx.eval_to_allocation_raw(param_env.and(cid))?;

    // FIXME Need to provide a span to `eval_to_valtree`
    let ecx = mk_eval_cx(
        tcx,
        DUMMY_SP,
        param_env,
        // It is absolutely crucial for soundness that
        // we do not read from static items or other mutable memory.
        CanAccessStatics::No,
    );
    let place = ecx.raw_const_to_mplace(const_alloc).unwrap();
    debug!(?place);

    let mut num_nodes = 0;
    let valtree_result = const_to_valtree_inner(&ecx, &place, &mut num_nodes);

    match valtree_result {
        Ok(valtree) => Ok(Some(valtree)),
        Err(err) => {
            let did = cid.instance.def_id();
            let global_const_id = cid.display(tcx);
            match err {
                ValTreeCreationError::NodesOverflow => {
                    let span = tcx.hir().span_if_local(did);
                    tcx.sess.emit_err(MaxNumNodesInConstErr { span, global_const_id });

                    Ok(None)
                }
                ValTreeCreationError::NonSupportedType | ValTreeCreationError::Other => Ok(None),
            }
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn try_destructure_mir_constant_for_diagnostics<'tcx>(
    tcx: TyCtxt<'tcx>,
    val: ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> Option<mir::DestructuredConstant<'tcx>> {
    let param_env = ty::ParamEnv::reveal_all();
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, CanAccessStatics::No);
    let op = ecx.const_val_to_op(val, ty, None).ok()?;

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match ty.kind() {
        ty::Array(_, len) => (len.eval_target_usize(tcx, param_env) as usize, None, op),
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
            let val = op_to_const(&ecx, &field_op);
            Some((val, field_op.layout.ty))
        })
        .collect::<Option<Vec<_>>>()?;
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    Some(mir::DestructuredConstant { variant, fields })
}
