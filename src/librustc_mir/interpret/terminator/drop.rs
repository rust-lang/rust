use rustc::mir::BasicBlock;
use rustc::ty::{self, Ty};
use syntax::codemap::Span;

use rustc::mir::interpret::{EvalResult, PrimVal, Value};
use interpret::{Machine, ValTy, EvalContext, Place, PlaceExtra};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub(crate) fn drop_place(
        &mut self,
        place: Place,
        instance: ty::Instance<'tcx>,
        ty: Ty<'tcx>,
        span: Span,
        target: BasicBlock,
    ) -> EvalResult<'tcx> {
        trace!("drop_place: {:#?}", place);
        // We take the address of the object.  This may well be unaligned, which is fine for us here.
        // However, unaligned accesses will probably make the actual drop implementation fail -- a problem shared
        // by rustc.
        let val = match self.force_allocation(place)? {
            Place::Ptr {
                ptr,
                align: _,
                extra: PlaceExtra::Vtable(vtable),
            } => ptr.to_value_with_vtable(vtable),
            Place::Ptr {
                ptr,
                align: _,
                extra: PlaceExtra::Length(len),
            } => ptr.to_value_with_len(len),
            Place::Ptr {
                ptr,
                align: _,
                extra: PlaceExtra::None,
            } => ptr.to_value(),
            _ => bug!("force_allocation broken"),
        };
        self.drop(val, instance, ty, span, target)
    }

    fn drop(
        &mut self,
        arg: Value,
        instance: ty::Instance<'tcx>,
        ty: Ty<'tcx>,
        span: Span,
        target: BasicBlock,
    ) -> EvalResult<'tcx> {
        trace!("drop: {:#?}, {:?}, {:?}", arg, ty.sty, instance.def);

        let instance = match ty.sty {
            ty::TyDynamic(..) => {
                let vtable = match arg {
                    Value::ByValPair(_, PrimVal::Ptr(vtable)) => vtable,
                    _ => bug!("expected fat ptr, got {:?}", arg),
                };
                match self.read_drop_type_from_vtable(vtable)? {
                    Some(func) => func,
                    // no drop fn -> bail out
                    None => {
                        self.goto_block(target);
                        return Ok(())
                    },
                }
            }
            _ => instance,
        };

        // the drop function expects a reference to the value
        let valty = ValTy {
            value: arg,
            ty: self.tcx.mk_mut_ptr(ty),
        };

        let fn_sig = self.tcx.fn_sig(instance.def_id()).skip_binder().clone();

        self.eval_fn_call(
            instance,
            Some((Place::undef(), target)),
            &vec![valty],
            span,
            fn_sig,
        )
    }
}
