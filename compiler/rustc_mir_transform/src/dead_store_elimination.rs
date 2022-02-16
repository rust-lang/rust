use rustc_middle::{
    mir::*,
    ty::{List, TyCtxt},
};
use smallvec::SmallVec;

/// Loops over basic blocks and calls [`simple_local_dse`] for each, see there for more.
pub struct SimpleLocalDse;

impl<'tcx> MirPass<'tcx> for SimpleLocalDse {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for bb in body.basic_blocks_mut() {
            simple_local_dse(bb, tcx);
        }
    }
}

/// This performs a very basic form of dead store elimintation on the basic block.
///
/// Essentially, we loop over the statements in reverse order. As we go, we maintain a list of
/// places that will be written to before they are used again. If we find a write to any such place,
/// we can replace it with a nop. Full details and arguments for correctness are in inline comments.
#[instrument(level = "debug", skip(tcx))]
pub fn simple_local_dse<'tcx>(bb: &mut BasicBlockData<'tcx>, tcx: TyCtxt<'tcx>) {
    let mut overwritten: Vec<Place<'tcx>> = Vec::new();
    // We iterate backwards over the statements in the basic block
    for stmt in bb.statements.iter_mut().rev() {
        // For each statement, compute where it reads from and writes to
        let data = compute_statement_data(stmt, tcx);
        // If the statement definitely derefs something, then assume any write that is visible now
        // is not dead
        let Some(data) = data else {
            overwritten.clear();
            continue;
        };
        if let Some(p) = data.stored {
            // If the statement writes somewhere, and we know that a "parent" place is over-written
            // later, the statement can be optimized out. This uses the assumptions that 1) `p` does
            // not include any `Deref`s (this is enforced in `compute_statement_data`) and that 2)
            // this write is the entirety of the statements behavior, ie that knowing that the write
            // is dead lets us remove the statement entirely.

            // It may be possible to make this smarter. For example, if a type with no padding has
            // all of its fields overwritten, then the whole type can be considered overwritten.
            // Leave that for the future though.
            if overwritten
                .iter()
                .copied()
                .any(|dp| dp.local == p.local && p.projection[..].starts_with(&dp.projection[..]))
            {
                debug!("Optimizing {:?}", stmt);
                stmt.make_nop();
                continue;
            }

            // If we get here, this write can't be optimized out. We may now be able to add it to
            // `overwritten`, but to do that, we need to check that the place does not contain any
            // non-constant indexing. The values of such indexes may change, which may make the part
            // of memory that the place points to inconsistent.
            if p.projection.iter().all(|x| !matches!(x, ProjectionElem::Index(_))) {
                overwritten.push(p);
            }
        }

        // We need to kick elements out of `overwritten` if their value was used.
        overwritten.retain(|p1| data.loaded.iter().copied().all(|p2| !p1.may_overlap(p2)))
    }
}

struct StatementData<'tcx> {
    /// The place that this statement writes to, or `None`, if it doesn't write anywhere. If this is
    /// `Some`, it is assumed that the corresponding write represents the
    /// entirety of the statement's behavior.
    stored: Option<Place<'tcx>>,
    /// The places that this statement loads from. If `stored` is `Some(_)`, then it is assumed that
    /// these loads are conditioned on the above store not being optimized out.
    loaded: SmallVec<[Place<'tcx>; 8]>,
}

/// Returns information about how one statement interacts with memory.
///
/// Returning `None` indicates that execution of this statement accesses memory not inside a local.
fn compute_statement_data<'tcx>(
    stmt: &Statement<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> Option<StatementData<'tcx>> {
    let mut loaded = SmallVec::new();

    // Adds a `load` to the `loaded` list. This also adds in any locals that are used as indexes.
    let mut add_load = |p: Place<'tcx>| {
        loaded.push(p);
        for elem in p.projection.iter() {
            if let ProjectionElem::Index(i) = elem {
                loaded.push(Place { local: i, projection: List::empty() });
            }
        }
    };
    // Adds the address of `p` to the loaded list.
    let add_address_of = |p: Place<'tcx>, loaded: &mut SmallVec<_>| {
        // First of all, computing the address unconditionally uses any `Index`s that appear in the
        // place.
        for elem in p.projection.iter() {
            if let ProjectionElem::Index(i) = elem {
                loaded.push(Place { local: i, projection: List::empty() });
            }
        }

        // Now, we additionally use the place that is outside the innermost `Deref`, since that
        // contains the pointer from which we're computing the address.
        if let Some(i) = p.projection.iter().rposition(|x| x == ProjectionElem::Deref) {
            let prefix = &p.projection[0..i];
            loaded.push(Place { local: p.local, projection: tcx.intern_place_elems(prefix) });
        };
    };

    let mut stored = match &stmt.kind {
        StatementKind::FakeRead(x) => {
            add_load(x.1);
            Some(x.1)
        }
        StatementKind::SetDiscriminant { .. } => {
            // There isn't really a clear place associated with a discriminant, so we won't report one
            None
        }
        StatementKind::StorageDead(x) => Some(Place { local: *x, projection: List::empty() }),
        StatementKind::Retag(_, x) => {
            add_load(**x);
            Some(**x)
        }
        StatementKind::AscribeUserType(x, _) => {
            add_load((**x).0);
            Some((**x).0)
        }
        StatementKind::Coverage(_) | StatementKind::StorageLive(_) | StatementKind::Nop => None,
        StatementKind::CopyNonOverlapping(_) => {
            return None;
        }
        StatementKind::Assign(x) => {
            let mut dest = Some(x.0);
            let src = &x.1;
            match src {
                Rvalue::Use(op)
                | Rvalue::Repeat(op, _)
                | Rvalue::Cast(_, op, _)
                | Rvalue::UnaryOp(_, op)
                | Rvalue::ShallowInitBox(op, _) => {
                    op.place().map(&mut add_load);
                }
                Rvalue::Len(p) | Rvalue::Discriminant(p) => {
                    add_load(*p);
                }
                Rvalue::Ref(_, _, p) | Rvalue::AddressOf(_, p) => {
                    add_address_of(*p, &mut loaded);
                }
                Rvalue::BinaryOp(_, x) | Rvalue::CheckedBinaryOp(_, x) => {
                    x.0.place().map(&mut add_load);
                    x.1.place().map(&mut add_load);
                }
                Rvalue::Aggregate(_, v) => {
                    for op in v {
                        op.place().map(&mut add_load);
                    }
                }
                Rvalue::ThreadLocalRef(_) => {
                    // Creating a thread local ref has side-effects, so don't optimize that away
                    dest = None;
                }
                Rvalue::NullaryOp(..) => {}
            };
            dest
        }
    };
    if let Some(p) = stored {
        add_address_of(p, &mut loaded);

        if p.projection.iter().any(|x| x == ProjectionElem::Deref) {
            // We don't reason about memory, so we cannot optimize this statement
            stored = None;
        }
    }

    Some(StatementData { stored, loaded })
}
