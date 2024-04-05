use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Location, MentionedItem};
use rustc_middle::ty::{self, adjustment::PointerCoercion, TyCtxt};
use rustc_span::source_map::Spanned;

pub struct MentionedItemsVisitor<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub body: &'a mir::Body<'tcx>,
    pub mentioned_items: &'a mut Vec<Spanned<MentionedItem<'tcx>>>,
}

// This visitor is carefully in sync with the one in `rustc_monomorphize::collector`. We are
// visiting the exact same places but then instead of monomorphizing and creating `MonoItems`, we
// have to remain generic and just recording the relevant information in `mentioned_items`, where it
// will then be monomorphized later during "mentioned items" collection.
impl<'tcx> Visitor<'tcx> for MentionedItemsVisitor<'_, 'tcx> {
    #[instrument(skip(self))]
    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        let span = || self.body.source_info(location).span;
        match &terminator.kind {
            mir::TerminatorKind::Call { func, .. } => {
                let callee_ty = func.ty(self.body, self.tcx);
                self.mentioned_items
                    .push(Spanned { node: MentionedItem::Fn(callee_ty), span: span() });
            }
            mir::TerminatorKind::Drop { place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                self.mentioned_items.push(Spanned { node: MentionedItem::Drop(ty), span: span() });
            }
            mir::TerminatorKind::InlineAsm { ref operands, .. } => {
                for op in operands {
                    match *op {
                        mir::InlineAsmOperand::SymFn { ref value } => {
                            self.mentioned_items.push(Spanned {
                                node: MentionedItem::Fn(value.const_.ty()),
                                span: span(),
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        let span = || self.body.source_info(location).span;
        match *rvalue {
            // We need to detect unsizing casts that required vtables.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize),
                ref operand,
                target_ty,
            )
            | mir::Rvalue::Cast(mir::CastKind::DynStar, ref operand, target_ty) => {
                // This isn't monomorphized yet so we can't tell what the actual types are -- just
                // add everything that may involve a vtable.
                let source_ty = operand.ty(self.body, self.tcx);
                let may_involve_vtable = match (
                    source_ty.builtin_deref(true).map(|t| t.kind()),
                    target_ty.builtin_deref(true).map(|t| t.kind()),
                ) {
                    (Some(ty::Array(..)), Some(ty::Str | ty::Slice(..))) => false, // &str/&[T] unsizing
                    _ => true,
                };
                if may_involve_vtable {
                    self.mentioned_items.push(Spanned {
                        node: MentionedItem::UnsizeCast { source_ty, target_ty },
                        span: span(),
                    });
                }
            }
            // Similarly, record closures that are turned into function pointers.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_)),
                ref operand,
                _,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                self.mentioned_items
                    .push(Spanned { node: MentionedItem::Closure(source_ty), span: span() });
            }
            // And finally, function pointer reification casts.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer),
                ref operand,
                _,
            ) => {
                let fn_ty = operand.ty(self.body, self.tcx);
                self.mentioned_items.push(Spanned { node: MentionedItem::Fn(fn_ty), span: span() });
            }
            _ => {}
        }
    }
}
