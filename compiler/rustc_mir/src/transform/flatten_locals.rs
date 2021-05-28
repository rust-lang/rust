use crate::transform::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use std::collections::hash_map::Entry;

pub struct FlattenLocals;

impl<'tcx> MirPass<'tcx> for FlattenLocals {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 4 {
            return;
        }

        let replacements = compute_flattening(tcx, body);
        let mut all_dead_locals = FxHashSet::default();
        all_dead_locals.extend(replacements.fields.keys().map(|p| p.local));
        if all_dead_locals.is_empty() {
            return;
        }

        ReplacementVisitor { tcx, map: &replacements, all_dead_locals }.visit_body(body);

        let mut replaced_locals: IndexVec<_, _> = IndexVec::new();
        for (k, v) in replacements.fields {
            replaced_locals.ensure_contains_elem(k.local, || Vec::new());
            replaced_locals[k.local].push(v)
        }
        // Sort locals to avoid depending on FxHashMap order.
        for v in replaced_locals.iter_mut() {
            v.sort_unstable()
        }
        for bbdata in body.basic_blocks_mut().iter_mut() {
            bbdata.expand_statements(|stmt| {
                let source_info = stmt.source_info;
                let (live, origin_local) = match &stmt.kind {
                    StatementKind::StorageLive(l) => (true, *l),
                    StatementKind::StorageDead(l) => (false, *l),
                    _ => return None,
                };
                replaced_locals.get(origin_local).map(move |final_locals| {
                    final_locals.iter().map(move |&l| {
                        let kind = if live {
                            StatementKind::StorageLive(l)
                        } else {
                            StatementKind::StorageDead(l)
                        };
                        Statement { source_info, kind }
                    })
                })
            });
        }
    }
}

fn escaping_locals(body: &Body<'_>) -> FxHashSet<Local> {
    let mut set: FxHashSet<_> = (0..body.arg_count + 1).map(Local::new).collect();
    for (local, decl) in body.local_decls().iter_enumerated() {
        if decl.ty.is_union() || decl.ty.is_enum() {
            set.insert(local);
        }
    }
    let mut visitor = EscapeVisitor { set };
    visitor.visit_body(body);
    return visitor.set;

    struct EscapeVisitor {
        set: FxHashSet<Local>,
    }

    impl Visitor<'_> for EscapeVisitor {
        fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
            self.set.insert(*local);
        }

        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
            // Mirror the implementation in PreFlattenVisitor.
            if let &[PlaceElem::Field(..), ..] = &place.projection[..] {
                return;
            }
            self.super_place(place, context, location);
        }

        fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
            if let Rvalue::AddressOf(.., place) | Rvalue::Ref(.., place) = rvalue {
                if !place.is_indirect() {
                    // Raw pointers may be used to access anything inside the enclosing place.
                    self.set.insert(place.local);
                    return;
                }
            }
            self.super_rvalue(rvalue, location)
        }

        fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
            if let StatementKind::StorageLive(..) | StatementKind::StorageDead(..) = statement.kind
            {
                // Storage statements are expanded in run_pass.
                return;
            }
            self.super_statement(statement, location)
        }

        fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
            if let TerminatorKind::Drop { place, .. }
            | TerminatorKind::DropAndReplace { place, .. } = terminator.kind
            {
                if !place.is_indirect() {
                    // Raw pointers may be used to access anything inside the enclosing place.
                    self.set.insert(place.local);
                    return;
                }
            }
            self.super_terminator(terminator, location);
        }
    }
}

#[derive(Default, Debug)]
struct ReplacementMap<'tcx> {
    fields: FxHashMap<PlaceRef<'tcx>, Local>,
}

fn compute_flattening<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> ReplacementMap<'tcx> {
    let escaping = escaping_locals(&*body);
    let (basic_blocks, local_decls, var_debug_info) =
        body.basic_blocks_local_decls_mut_and_var_debug_info();
    let mut visitor =
        PreFlattenVisitor { tcx, escaping, local_decls: local_decls, map: Default::default() };
    for (block, bbdata) in basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(block, bbdata);
    }
    for var_debug_info in &*var_debug_info {
        visitor.visit_var_debug_info(var_debug_info);
    }
    return visitor.map;

    struct PreFlattenVisitor<'tcx, 'll> {
        tcx: TyCtxt<'tcx>,
        local_decls: &'ll mut LocalDecls<'tcx>,
        escaping: FxHashSet<Local>,
        map: ReplacementMap<'tcx>,
    }

    impl<'tcx, 'll> PreFlattenVisitor<'tcx, 'll> {
        fn create_place(&mut self, place: PlaceRef<'tcx>) {
            if self.escaping.contains(&place.local) {
                return;
            }

            match self.map.fields.entry(place) {
                Entry::Occupied(_) => {}
                Entry::Vacant(v) => {
                    let ty = place.ty(&*self.local_decls, self.tcx).ty;
                    let local = self.local_decls.push(LocalDecl {
                        ty,
                        user_ty: None,
                        ..self.local_decls[place.local].clone()
                    });
                    v.insert(local);
                }
            }
        }
    }

    impl<'tcx, 'll> Visitor<'tcx> for PreFlattenVisitor<'tcx, 'll> {
        fn visit_place(&mut self, place: &Place<'tcx>, _: PlaceContext, _: Location) {
            if let &[PlaceElem::Field(..), ..] = &place.projection[..] {
                let pr = PlaceRef { local: place.local, projection: &place.projection[..1] };
                self.create_place(pr)
            }
        }
    }
}

struct ReplacementVisitor<'tcx, 'll> {
    tcx: TyCtxt<'tcx>,
    map: &'ll ReplacementMap<'tcx>,
    all_dead_locals: FxHashSet<Local>,
}

impl<'tcx, 'll> MutVisitor<'tcx> for ReplacementVisitor<'tcx, 'll> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let StatementKind::StorageLive(..) | StatementKind::StorageDead(..) = statement.kind {
            // Storage statements are expanded in run_pass.
            return;
        }
        self.super_statement(statement, location)
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let &[PlaceElem::Field(..), ref rest @ ..] = &place.projection[..] {
            let pr = PlaceRef { local: place.local, projection: &place.projection[..1] };
            if let Some(local) = self.map.fields.get(&pr) {
                *place = Place { local: *local, projection: self.tcx.intern_place_elems(&rest) };
                return;
            }
        }
        self.super_place(place, context, location)
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert!(!self.all_dead_locals.contains(local),);
    }
}
