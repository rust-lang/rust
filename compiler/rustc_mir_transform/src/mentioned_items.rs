use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Location, MentionedItem};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_span::source_map::Spanned;

pub(super) struct MentionedItems;

struct MentionedItemsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    mentioned_items: Vec<Spanned<MentionedItem<'tcx>>>,
}

impl<'tcx> crate::MirPass<'tcx> for MentionedItems {
    fn is_enabled(&self, _sess: &Session) -> bool {
        // If this pass is skipped the collector assume that nothing got mentioned! We could
        // potentially skip it in opt-level 0 if we are sure that opt-level will never *remove* uses
        // of anything, but that still seems fragile. Furthermore, even debug builds use level 1, so
        // special-casing level 0 is just not worth it.
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        let mut visitor = MentionedItemsVisitor { tcx, body, mentioned_items: Vec::new() };
        visitor.visit_body(body);
        body.set_mentioned_items(visitor.mentioned_items);
    }

    fn is_required(&self) -> bool {
        true
    }
}

// This visitor is carefully in sync with the one in `rustc_monomorphize::collector`. We are
// visiting the exact same places but then instead of monomorphizing and creating `MonoItems`, we
// have to remain generic and just recording the relevant information in `mentioned_items`, where it
// will then be monomorphized later during "mentioned items" collection.
impl<'tcx> Visitor<'tcx> for MentionedItemsVisitor<'_, 'tcx> {
    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        let span = || self.body.source_info(location).span;
        match &terminator.kind {
            mir::TerminatorKind::Call { func, .. } | mir::TerminatorKind::TailCall { func, .. } => {
                let callee_ty = func.ty(self.body, self.tcx);
                self.mentioned_items
                    .push(Spanned { node: MentionedItem::Fn(callee_ty), span: span() });
            }
            mir::TerminatorKind::Drop { place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                self.mentioned_items.push(Spanned { node: MentionedItem::Drop(ty), span: span() });
            }
            mir::TerminatorKind::InlineAsm { operands, .. } => {
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
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                ref operand,
                target_ty,
            ) => {
                // This isn't monomorphized yet so we can't tell what the actual types are -- just
                // add everything that may involve a vtable.
                let source_ty = operand.ty(self.body, self.tcx);
                let may_involve_vtable = match (
                    source_ty.builtin_deref(true).map(|t| t.kind()),
                    target_ty.builtin_deref(true).map(|t| t.kind()),
                ) {
                    // &str/&[T] unsizing
                    (Some(ty::Array(..)), Some(ty::Str | ty::Slice(..))) => false,

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
                mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _),
                ref operand,
                _,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                self.mentioned_items
                    .push(Spanned { node: MentionedItem::Closure(source_ty), span: span() });
            }
            // And finally, function pointer reification casts.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _),
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
