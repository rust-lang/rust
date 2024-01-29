//! This pass rewrites the receiver of the function to a thin-pointer compatible representation of
//! the provided type.

use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

// This is a layering violation - this is replicating work that occurs when computing an ABI.
//
// We need to have a `dyn` receiver type in order to allow a type match at the vtable call. However,
// we want to match the existing ABI for vtable methods, which passes a *thin* pointer
// Existing shims make this cast implicit at the callsite, so they don't need to get this type
// correct. Without this, we will get a local type mismatch when we actually try to use nontrivial
// receiver (e.g. `Arc<Self>`). This means that we must use a thin self, because otherwise codegen
// will assume an argument is present for the vtable. Unfortunately, unwrapping the receiver type
// currently involves replicating ABI / layout work.
//
// Perhaps in the future we could avoid the thin-self hack with an explicit existentials, e.g.
// * `∃T: Foo. *const T`
// * `∃T: Foo. Arc<T>`
// but until then, we need to unwrap receivers down to `* dyn Foo` of some variant to use the
// existing codegen paths.
fn unwrap_receiver<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    use ty::layout::{LayoutCx, LayoutOf, MaybeResult, TyAndLayout};
    let cx = LayoutCx { tcx, param_env: ty::ParamEnv::reveal_all() };
    let mut receiver_layout: TyAndLayout<'_> =
        cx.layout_of(ty).to_result().expect("unable to compute layout of receiver type");
    // The VTableShim should have already done any `dyn Foo` -> `*const dyn Foo` coercions
    assert!(!receiver_layout.is_unsized());
    // If we aren't a pointer or a ref already, we better be a no-padding wrapper around one
    while !receiver_layout.ty.is_unsafe_ptr() && !receiver_layout.ty.is_ref() {
        receiver_layout = receiver_layout
            .non_1zst_field(&cx)
            .expect("not exactly one non-1-ZST field in a CFI shim receiver")
            .1
    }
    receiver_layout.ty
}

// Visitor to rewrite all uses of a given local to another
struct RewriteLocal<'tcx> {
    tcx: TyCtxt<'tcx>,
    source: Local,
    target: Local,
}

impl<'tcx> visit::MutVisitor<'tcx> for RewriteLocal<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn visit_local(
        &mut self,
        local: &mut Local,
        _context: visit::PlaceContext,
        _location: Location,
    ) {
        if self.source == *local {
            *local = self.target;
        }
    }
}

pub struct RewriteReceiver<'tcx> {
    receiver: Ty<'tcx>,
}

impl<'tcx> RewriteReceiver<'tcx> {
    pub fn new(receiver: Ty<'tcx>) -> Self {
        Self { receiver }
    }
}

impl<'tcx> MirPass<'tcx> for RewriteReceiver<'tcx> {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.cfi_shims()
    }
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        use visit::MutVisitor;
        let source_info = SourceInfo::outermost(body.span);
        let receiver =
            body.args_iter().next().expect("RewriteReceiver pass on function with no arguments?");
        let cast_receiver = body.local_decls.push(body.local_decls[receiver].clone());
        body.local_decls[receiver].ty = unwrap_receiver(tcx, self.receiver);
        body.local_decls[receiver].mutability = Mutability::Not;
        RewriteLocal { tcx, source: receiver, target: cast_receiver }.visit_body(body);
        body.basic_blocks.as_mut_preserves_cfg()[START_BLOCK].statements.insert(
            0,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(cast_receiver),
                    Rvalue::Cast(
                        CastKind::Transmute,
                        Operand::Move(Place::from(receiver)),
                        body.local_decls[cast_receiver].ty,
                    ),
                ))),
            },
        );
    }
}
