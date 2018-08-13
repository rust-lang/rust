use rustc::mir::BasicBlock;
use rustc::ty::{self, Ty, layout::LayoutOf};
use syntax::source_map::Span;

use rustc::mir::interpret::{EvalResult};
use interpret::{Machine, EvalContext, PlaceTy, Value, OpTy, Operand};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub(crate) fn drop_place(
        &mut self,
        place: PlaceTy<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Span,
        target: BasicBlock,
    ) -> EvalResult<'tcx> {
        trace!("drop_place: {:#?}", place);
        // We take the address of the object.  This may well be unaligned, which is fine for us here.
        // However, unaligned accesses will probably make the actual drop implementation fail -- a problem shared
        // by rustc.
        let val = self.force_allocation(place)?.to_ref(&self);
        self.drop(val, instance, place.layout.ty, span, target)
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
                if let Value::ScalarPair(_, vtable) = arg {
                    self.read_drop_type_from_vtable(vtable.to_ptr()?)?
                } else {
                    bug!("expected fat ptr, got {:?}", arg);
                }
            }
            _ => instance,
        };

        // the drop function expects a reference to the value
        let arg = OpTy {
            op: Operand::Immediate(arg),
            layout: self.layout_of(self.tcx.mk_mut_ptr(ty))?,
        };

        let fn_sig = self.tcx.fn_sig(instance.def_id()).skip_binder().clone();
        // This should always be (), but getting it from the sig seems
        // easier than creating a layout of ().
        let dest = PlaceTy::null(&self, self.layout_of(fn_sig.output())?);

        self.eval_fn_call(
            instance,
            Some((dest, target)),
            &[arg],
            span,
            fn_sig,
        )
    }
}
