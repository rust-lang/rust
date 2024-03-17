use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, ConstOperand, Location, MentionedItem, MirPass};
use rustc_middle::ty::{self, adjustment::PointerCoercion, TyCtxt};
use rustc_session::Session;
use rustc_span::source_map::Spanned;

pub struct MentionedItems;

struct MentionedItemsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    mentioned_items: &'a mut Vec<Spanned<MentionedItem<'tcx>>>,
}

impl<'tcx> MirPass<'tcx> for MentionedItems {
    fn is_enabled(&self, _sess: &Session) -> bool {
        // If this pass is skipped the collector assume that nothing got mentioned! We could
        // potentially skip it in opt-level 0 if we are sure that opt-level will never *remove* uses
        // of anything, but that still seems fragile. Furthermore, even debug builds use level 1, so
        // special-casing level 0 is just not worth it.
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        debug_assert!(body.mentioned_items.is_empty());
        let mut mentioned_items = Vec::new();
        MentionedItemsVisitor { tcx, body, mentioned_items: &mut mentioned_items }.visit_body(body);
        body.mentioned_items = mentioned_items;
    }
}

impl<'tcx> Visitor<'tcx> for MentionedItemsVisitor<'_, 'tcx> {
    fn visit_constant(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
        let const_ = constant.const_;
        // This is how function items get referenced: via constants of `FnDef` type. This handles
        // both functions that are called and those that are just turned to function pointers.
        if let ty::FnDef(def_id, args) = const_.ty().kind() {
            debug!("adding to required_items: {def_id:?}");
            self.mentioned_items
                .push(Spanned { node: MentionedItem::Fn(*def_id, args), span: constant.span });
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        match terminator.kind {
            // We don't need to handle `Call` as we already handled all function type operands in
            // `visit_constant`. But we do need to handle `Drop`.
            mir::TerminatorKind::Drop { place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                let span = self.body.source_info(location).span;
                self.mentioned_items.push(Spanned { node: MentionedItem::Drop(ty), span });
            }
            _ => {}
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        match *rvalue {
            // We need to detect unsizing casts that required vtables.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize),
                ref operand,
                target_ty,
            )
            | mir::Rvalue::Cast(mir::CastKind::DynStar, ref operand, target_ty) => {
                let span = self.body.source_info(location).span;
                self.mentioned_items.push(Spanned {
                    node: MentionedItem::UnsizeCast {
                        source_ty: operand.ty(self.body, self.tcx),
                        target_ty,
                    },
                    span,
                });
            }
            // Similarly, record closures that are turned into function pointers.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_)),
                ref operand,
                _,
            ) => {
                let span = self.body.source_info(location).span;
                let source_ty = operand.ty(self.body, self.tcx);
                match *source_ty.kind() {
                    ty::Closure(def_id, args) => {
                        self.mentioned_items
                            .push(Spanned { node: MentionedItem::Closure(def_id, args), span });
                    }
                    _ => bug!(),
                }
            }
            // Function pointer casts are already handled by `visit_constant` above.
            _ => {}
        }
    }
}
