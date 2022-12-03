use crate::MirPass;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct Fold;

impl<'tcx> MirPass<'tcx> for Fold {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        while did_optimization(body, tcx) {}
    }
}

fn did_optimization<'tcx>(body: &mut Body<'tcx>, _tcx: TyCtxt<'tcx>) -> bool {
    let mut visitor = AnalysisVisitor {
        analysis: IndexVec::from_elem_n(Analysis::default(), body.local_decls.len()),
    };
    visitor.visit_body(body);
    let analysis = visitor.analysis;

    for result in analysis.iter() {
        let write = result.write.as_assign(body);
        let read = result.read.as_assign(body);
        let Some(((temp_loc, (_temp_place, temp_rvalue)), (final_loc, (final_place, final_rvalue)))) = write.zip(read) else {
                continue;
            };
        if temp_loc.block != final_loc.block {
            continue;
        }

        if let Rvalue::UnaryOp(UnOp::Not, _op) = final_rvalue {
            match temp_rvalue {
                Rvalue::BinaryOp(BinOp::Eq, op) => {
                    let new = Rvalue::BinaryOp(BinOp::Ne, op.clone());
                    let new = StatementKind::Assign(Box::new((*final_place, new)));
                    let local = final_place.local;
                    let statements = &mut body.basic_blocks_mut()[temp_loc.block].statements;
                    statements[temp_loc.statement_index].kind = new;
                    statements[final_loc.statement_index].make_nop();
                    expand_live_range(body, final_loc.block, local);
                    continue;
                }
                Rvalue::BinaryOp(BinOp::Ne, op) => {
                    let new = Rvalue::BinaryOp(BinOp::Eq, op.clone());
                    let new = StatementKind::Assign(Box::new((*final_place, new)));
                    let local = final_place.local;
                    let statements = &mut body.basic_blocks_mut()[temp_loc.block].statements;
                    statements[temp_loc.statement_index].kind = new;
                    statements[final_loc.statement_index].make_nop();
                    expand_live_range(body, final_loc.block, local);
                    continue;
                }
                _ => {}
            }
        }
    }
    false
}

fn expand_live_range<'tcx>(body: &mut Body<'tcx>, block: BasicBlock, r#final: Local) {
    use StatementKind::{StorageDead, StorageLive};

    let mut deleted_live = false;
    let mut deleted_dead = false;

    for statement in &mut body.basic_blocks_mut()[block].statements {
        if statement.kind == StorageLive(r#final) {
            statement.make_nop();
            deleted_live = true;
        } else if statement.kind == StorageDead(r#final) {
            statement.make_nop();
            deleted_dead = true;
        }
    }

    let source_info = body.local_decls[r#final].source_info;
    let block = &mut body.basic_blocks_mut()[block].statements;
    if deleted_live {
        block.insert(0, Statement { source_info, kind: StorageLive(r#final) });
    }
    if deleted_dead {
        block.push(Statement { source_info, kind: StorageDead(r#final) });
    }
}

#[derive(Debug, Clone, Copy)]
enum Status {
    None,
    Once(Location),
    Multiple,
}

impl Status {
    fn incr(&mut self, location: Location) {
        *self = match self {
            Status::None => Self::Once(location),
            Status::Once(_) | Status::Multiple => Self::Multiple,
        };
    }

    fn as_assign<'a, 'tcx>(
        self,
        body: &'a Body<'tcx>,
    ) -> Option<(Location, &'a (Place<'tcx>, Rvalue<'tcx>))> {
        let location = match self {
            Status::Once(location) => location,
            Status::None | Status::Multiple => return None,
        };

        body.stmt_at(location).left()?.kind.as_assign().map(|res| (location, res))
    }
}

#[derive(Debug, Clone)]
struct Analysis {
    read: Status,
    write: Status,
}

impl Default for Analysis {
    fn default() -> Self {
        Self { read: Status::None, write: Status::None }
    }
}

struct AnalysisVisitor {
    analysis: IndexVec<Local, Analysis>,
}

impl<'tcx> Visitor<'tcx> for AnalysisVisitor {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        use rustc_middle::mir::visit::MutatingUseContext;
        match context {
            PlaceContext::NonUse(_) => {}
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                self.analysis[place.local].write.incr(location);
            }
            PlaceContext::MutatingUse(
                MutatingUseContext::Borrow
                | MutatingUseContext::Projection
                | MutatingUseContext::AddressOf,
            ) => {
                self.analysis[place.local].read.incr(location);
            }
            PlaceContext::MutatingUse(_) => {
                self.analysis[place.local].write.incr(location);
            }
            PlaceContext::NonMutatingUse(_) => {
                self.analysis[place.local].read.incr(location);
            }
        }

        self.super_place(place, context, location);
    }
}
