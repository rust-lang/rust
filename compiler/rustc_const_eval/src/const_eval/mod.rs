// Not in interpret to make sure we do not use private implementation details

use std::convert::TryFrom;

use rustc_hir::Mutability;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{
    mir::{self, interpret::ConstAlloc},
    ty::ScalarInt,
};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

use crate::interpret::{
    intern_const_alloc_recursive, ConstValue, InternKind, InterpCx, MPlaceTy, MemPlaceMeta, Scalar,
};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
pub use machine::*;

pub(crate) fn const_caller_location(
    tcx: TyCtxt<'_>,
    (file, line, col): (Symbol, u32, u32),
) -> ConstValue<'_> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), false);

    let loc_place = ecx.alloc_caller_location(file, line, col);
    if intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &loc_place).is_err() {
        bug!("intern_const_alloc_recursive should not error in this case")
    }
    ConstValue::Scalar(Scalar::from_pointer(loc_place.ptr.into_pointer_or_addr().unwrap(), &tcx))
}

/// Convert an evaluated constant to a type level constant
pub(crate) fn const_to_valtree<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    raw: ConstAlloc<'tcx>,
) -> Option<ty::ValTree<'tcx>> {
    let ecx = mk_eval_cx(
        tcx, DUMMY_SP, param_env,
        // It is absolutely crucial for soundness that
        // we do not read from static items or other mutable memory.
        false,
    );
    let place = ecx.raw_const_to_mplace(raw).unwrap();
    const_to_valtree_inner(&ecx, &place)
}

fn const_to_valtree_inner<'tcx>(
    ecx: &CompileTimeEvalContext<'tcx, 'tcx>,
    place: &MPlaceTy<'tcx>,
) -> Option<ty::ValTree<'tcx>> {
    let branches = |n, variant| {
        let place = match variant {
            Some(variant) => ecx.mplace_downcast(&place, variant).unwrap(),
            None => *place,
        };
        let variant =
            variant.map(|variant| Some(ty::ValTree::Leaf(ScalarInt::from(variant.as_u32()))));
        let fields = (0..n).map(|i| {
            let field = ecx.mplace_field(&place, i).unwrap();
            const_to_valtree_inner(ecx, &field)
        });
        // For enums, we preped their variant index before the variant's fields so we can figure out
        // the variant again when just seeing a valtree.
        let branches = variant.into_iter().chain(fields);
        Some(ty::ValTree::Branch(
            ecx.tcx.arena.alloc_from_iter(branches.collect::<Option<Vec<_>>>()?),
        ))
    };
    match place.layout.ty.kind() {
        ty::FnDef(..) => Some(ty::ValTree::zst()),
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char => {
            let val = ecx.read_immediate(&place.into()).unwrap();
            let val = val.to_scalar().unwrap();
            Some(ty::ValTree::Leaf(val.assert_int()))
        }

        // Raw pointers are not allowed in type level constants, as we cannot properly test them for
        // equality at compile-time (see `ptr_guaranteed_eq`/`_ne`).
        // Technically we could allow function pointers (represented as `ty::Instance`), but this is not guaranteed to
        // agree with runtime equality tests.
        ty::FnPtr(_) | ty::RawPtr(_) => None,
        ty::Ref(..) => unimplemented!("need to use deref_const"),

        // Trait objects are not allowed in type level constants, as we have no concept for
        // resolving their backing type, even if we can do that at const eval time. We may
        // hypothetically be able to allow `dyn StructuralEq` trait objects in the future,
        // but it is unclear if this is useful.
        ty::Dynamic(..) => None,

        ty::Slice(_) | ty::Str => {
            unimplemented!("need to find the backing data of the slice/str and recurse on that")
        }
        ty::Tuple(substs) => branches(substs.len(), None),
        ty::Array(_, len) => branches(usize::try_from(len.eval_usize(ecx.tcx.tcx, ecx.param_env)).unwrap(), None),

        ty::Adt(def, _) => {
            if def.variants.is_empty() {
                bug!("uninhabited types should have errored and never gotten converted to valtree")
            }

            let variant = ecx.read_discriminant(&place.into()).unwrap().1;

            branches(def.variants[variant].fields.len(), def.is_enum().then_some(variant))
        }

        ty::Never
        | ty::Error(_)
        | ty::Foreign(..)
        | ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        // FIXME(oli-obk): we could look behind opaque types
        | ty::Opaque(..)
        | ty::Infer(_)
        // FIXME(oli-obk): we can probably encode closures just like structs
        | ty::Closure(..)
        | ty::Generator(..)
        | ty::GeneratorWitness(..) => None,
    }
}

/// This function uses `unwrap` copiously, because an already validated constant
/// must have valid fields and can thus never fail outside of compiler bugs. However, it is
/// invoked from the pretty printer, where it can receive enums with no variants and e.g.
/// `read_discriminant` needs to be able to handle that.
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> mir::DestructuredConst<'tcx> {
    trace!("destructure_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None).unwrap();

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match val.ty.kind() {
        ty::Array(_, len) => (usize::try_from(len.eval_usize(tcx, param_env)).unwrap(), None, op),
        ty::Adt(def, _) if def.variants.is_empty() => {
            return mir::DestructuredConst { variant: None, fields: &[] };
        }
        ty::Adt(def, _) => {
            let variant = ecx.read_discriminant(&op).unwrap().1;
            let down = ecx.operand_downcast(&op, variant).unwrap();
            (def.variants[variant].fields.len(), Some(variant), down)
        }
        ty::Tuple(substs) => (substs.len(), None, op),
        _ => bug!("cannot destructure constant {:?}", val),
    };

    let fields_iter = (0..field_count).map(|i| {
        let field_op = ecx.operand_field(&down, i).unwrap();
        let val = op_to_const(&ecx, &field_op);
        ty::Const::from_value(tcx, val, field_op.layout.ty)
    });
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    mir::DestructuredConst { variant, fields }
}

pub(crate) fn deref_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    trace!("deref_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None).unwrap();
    let mplace = ecx.deref_operand(&op).unwrap();
    if let Some(alloc_id) = mplace.ptr.provenance {
        assert_eq!(
            tcx.get_global_alloc(alloc_id).unwrap().unwrap_memory().mutability,
            Mutability::Not,
            "deref_const cannot be used with mutable allocations as \
            that could allow pattern matching to observe mutable statics",
        );
    }

    let ty = match mplace.meta {
        MemPlaceMeta::None => mplace.layout.ty,
        MemPlaceMeta::Poison => bug!("poison metadata in `deref_const`: {:#?}", mplace),
        // In case of unsized types, figure out the real type behind.
        MemPlaceMeta::Meta(scalar) => match mplace.layout.ty.kind() {
            ty::Str => bug!("there's no sized equivalent of a `str`"),
            ty::Slice(elem_ty) => tcx.mk_array(elem_ty, scalar.to_machine_usize(&tcx).unwrap()),
            _ => bug!(
                "type {} should not have metadata, but had {:?}",
                mplace.layout.ty,
                mplace.meta
            ),
        },
    };

    tcx.mk_const(ty::Const { val: ty::ConstKind::Value(op_to_const(&ecx, &mplace.into())), ty })
}
