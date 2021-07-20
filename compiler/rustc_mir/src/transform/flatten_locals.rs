use crate::transform::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::vec::Idx;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use std::collections::hash_map::Entry;

pub struct FlattenLocals;

impl<'tcx> MirPass<'tcx> for FlattenLocals {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 4 {
            return;
        }

        let replacements = compute_flattening(tcx, body);
        let mut all_dead_locals = FxHashSet::default();
        all_dead_locals.extend(replacements.discr.keys().copied());
        all_dead_locals.extend(replacements.fields.keys().map(|p| p.local));
        if all_dead_locals.is_empty() {
            return;
        }

        let mut visitor = ReplacementVisitor { tcx, map: &replacements, all_dead_locals };
        visitor.visit_body(body);
        super::simplify::simplify_locals(body, tcx);
    }
}

fn escaping_locals(body: &Body<'_>) -> FxHashSet<Local> {
    let mut set: FxHashSet<_> = (0..body.arg_count + 1).map(Local::new).collect();
    for (local, decl) in body.local_decls().iter_enumerated() {
        if decl.ty.is_union() || decl.ty.is_enum() {
            set.insert(local);
        }
    }
    let mut visitor = EscapeVisitor { escaping: false, set };
    visitor.visit_body(body);
    return visitor.set;

    struct EscapeVisitor {
        escaping: bool,
        set: FxHashSet<Local>,
    }

    impl Visitor<'_> for EscapeVisitor {
        fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
            if self.escaping {
                self.set.insert(*local);
            }
        }

        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
            if place.projection.is_empty()
                && !matches!(
                    context,
                    PlaceContext::NonUse(NonUseContext::StorageLive | NonUseContext::StorageDead)
                )
            {
                self.set.insert(place.local);
            } else {
                self.super_place(place, context, location);
            }
        }

        fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
            if let Rvalue::Discriminant(place) = rvalue {
                if place.projection.is_empty() {
                    // Getting discriminant on an enum is ok.
                    return;
                }
            }
            if let Rvalue::AddressOf(..) | Rvalue::Ref(..) = rvalue {
                // Raw pointers may be used to access anything inside the enclosing place.
                self.escaping = true;
                self.super_rvalue(rvalue, location);
                self.escaping = false;
            } else {
                self.super_rvalue(rvalue, location)
            }
        }

        fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
            if let StatementKind::SetDiscriminant { place, .. } = &statement.kind {
                if place.projection.is_empty() {
                    // Setting discriminant on an enum is ok.
                    return;
                }
            }
            self.super_statement(statement, location)
        }

        fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
            if let TerminatorKind::Drop { .. } | TerminatorKind::DropAndReplace { .. } =
                terminator.kind
            {
                // Raw pointers may be used to access anything inside the enclosing place.
                self.escaping = true;
                self.super_terminator(terminator, location);
                self.escaping = false;
            } else {
                self.super_terminator(terminator, location);
            }
        }
    }
}

#[derive(Default, Debug)]
struct ReplacementMap<'tcx> {
    discr: FxHashMap<Local, (Local, Ty<'tcx>)>,
    fields: FxHashMap<PlaceRef<'tcx>, Local>,
}

fn compute_flattening<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> ReplacementMap<'tcx> {
    let escaping = escaping_locals(&*body);
    let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
    let mut visitor =
        PreFlattenVisitor { tcx, escaping, local_decls: local_decls, map: Default::default() };
    for (block, bbdata) in basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(block, bbdata);
    }
    return visitor.map;

    struct PreFlattenVisitor<'tcx, 'll> {
        tcx: TyCtxt<'tcx>,
        local_decls: &'ll mut LocalDecls<'tcx>,
        escaping: FxHashSet<Local>,
        map: ReplacementMap<'tcx>,
    }

    impl<'tcx, 'll> PreFlattenVisitor<'tcx, 'll> {
        fn create_discriminant(&mut self, local: Local) -> bool {
            if self.escaping.contains(&local) {
                return false;
            }

            if let Entry::Vacant(v) = self.map.discr.entry(local) {
                let ty = self.local_decls[local].ty.discriminant_ty(self.tcx);
                let local = self.local_decls.push(LocalDecl {
                    ty,
                    user_ty: None,
                    ..self.local_decls[local].clone()
                });
                v.insert((local, ty));
            }

            true
        }

        fn create_place(&mut self, place: PlaceRef<'tcx>) {
            if self.escaping.contains(&place.local) {
                return;
            }

            match self.map.fields.entry(place.clone()) {
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
        fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
            if let StatementKind::SetDiscriminant { place, .. } = &statement.kind {
                if place.projection.is_empty() {
                    if self.create_discriminant(place.local) {
                        return;
                    }
                }
            }
            self.super_statement(statement, location)
        }

        fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
            if let Rvalue::Discriminant(place) = rvalue {
                if let Some(local) = place.as_local() {
                    if self.create_discriminant(local) {
                        return;
                    }
                }
            }
            self.super_rvalue(rvalue, location)
        }

        fn visit_place(&mut self, place: &Place<'tcx>, _: PlaceContext, _: Location) {
            let nproj = match &place.projection[..] {
                &[PlaceElem::Field(..), ..] => 1,
                &[PlaceElem::Downcast(..), PlaceElem::Field(..), ..] => 2,
                _ => return,
            };
            let pr = PlaceRef { local: place.local, projection: &place.projection[..nproj] };
            self.create_place(pr)
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
        match &statement.kind {
            StatementKind::SetDiscriminant { place, variant_index }
                if place.projection.is_empty() =>
            {
                if let Some((local, discr_ty)) = self.map.discr.get(&place.local) {
                    statement.kind = StatementKind::Assign(box (
                        Place::from(*local),
                        Rvalue::Use(Operand::const_from_scalar(
                            self.tcx,
                            discr_ty,
                            Scalar::from_u64(variant_index.as_u32().into()),
                            statement.source_info.span,
                        )),
                    ));

                    return;
                }
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local)
                if self.all_dead_locals.contains(local) =>
            {
                statement.make_nop();
                return;
            }
            _ => {}
        }
        self.super_statement(statement, location)
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if let Rvalue::Discriminant(place) = rvalue {
            if let Some(local) = place.as_local() {
                if let Some((local, _)) = self.map.discr.get(&local) {
                    *rvalue = Rvalue::Use(Operand::Copy(Place::from(*local)));
                }
            }
        }
        self.super_rvalue(rvalue, location)
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let &[PlaceElem::Field(..), ref rest @ ..] = &place.projection[..] {
            let pr = PlaceRef { local: place.local, projection: &place.projection[..1] };
            if let Some(local) = self.map.fields.get(&pr) {
                *place = Place { local: *local, projection: self.tcx.intern_place_elems(&rest) };
                return;
            }
        } else if let &[PlaceElem::Downcast(..), PlaceElem::Field(..), ref rest @ ..] =
            &place.projection[..]
        {
            let pr = PlaceRef { local: place.local, projection: &place.projection[..2] };
            if let Some(local) = self.map.fields.get(&pr) {
                *place = Place { local: *local, projection: self.tcx.intern_place_elems(&rest) };
                return;
            }
        }
        self.super_place(place, context, location)
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert!(!self.all_dead_locals.contains(local));
    }
}
