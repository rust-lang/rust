use crate::MirPass;
use rustc_data_structures::fx::{FxIndexMap, IndexEntry};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct ScalarReplacementOfAggregates;

impl<'tcx> MirPass<'tcx> for ScalarReplacementOfAggregates {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 4
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let escaping = escaping_locals(&*body);
        debug!(?escaping);
        let replacements = compute_flattening(tcx, body, escaping);
        debug!(?replacements);
        replace_flattened_locals(tcx, body, replacements);
    }
}

/// Identify all locals that are not eligible for SROA.
///
/// There are 3 cases:
/// - the aggegated local is used or passed to other code (function parameters and arguments);
/// - the locals is a union or an enum;
/// - the local's address is taken, and thus the relative addresses of the fields are observable to
///   client code.
fn escaping_locals(body: &Body<'_>) -> BitSet<Local> {
    let mut set = BitSet::new_empty(body.local_decls.len());
    set.insert_range(RETURN_PLACE..=Local::from_usize(body.arg_count));
    for (local, decl) in body.local_decls().iter_enumerated() {
        if decl.ty.is_union() || decl.ty.is_enum() {
            set.insert(local);
        }
    }
    let mut visitor = EscapeVisitor { set };
    visitor.visit_body(body);
    return visitor.set;

    struct EscapeVisitor {
        set: BitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for EscapeVisitor {
        fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
            self.set.insert(local);
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
            if let StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Deinit(..) = statement.kind
            {
                // Storage statements are expanded in run_pass.
                return;
            }
            self.super_statement(statement, location)
        }

        fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
            // Drop implicitly calls `drop_in_place`, which takes a `&mut`.
            // This implies that `Drop` implicitly takes the address of the place.
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

        // We ignore anything that happens in debuginfo, since we expand it using
        // `VarDebugInfoContents::Composite`.
        fn visit_var_debug_info(&mut self, _: &VarDebugInfo<'tcx>) {}
    }
}

#[derive(Default, Debug)]
struct ReplacementMap<'tcx> {
    fields: FxIndexMap<PlaceRef<'tcx>, Local>,
}

/// Compute the replacement of flattened places into locals.
///
/// For each eligible place, we assign a new local to each accessed field.
/// The replacement will be done later in `ReplacementVisitor`.
fn compute_flattening<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    escaping: BitSet<Local>,
) -> ReplacementMap<'tcx> {
    let mut visitor = PreFlattenVisitor {
        tcx,
        escaping,
        local_decls: &mut body.local_decls,
        map: Default::default(),
    };
    for (block, bbdata) in body.basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(block, bbdata);
    }
    return visitor.map;

    struct PreFlattenVisitor<'tcx, 'll> {
        tcx: TyCtxt<'tcx>,
        local_decls: &'ll mut LocalDecls<'tcx>,
        escaping: BitSet<Local>,
        map: ReplacementMap<'tcx>,
    }

    impl<'tcx, 'll> PreFlattenVisitor<'tcx, 'll> {
        fn create_place(&mut self, place: PlaceRef<'tcx>) {
            if self.escaping.contains(place.local) {
                return;
            }

            match self.map.fields.entry(place) {
                IndexEntry::Occupied(_) => {}
                IndexEntry::Vacant(v) => {
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

/// Perform the replacement computed by `compute_flattening`.
fn replace_flattened_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    replacements: ReplacementMap<'tcx>,
) {
    let mut all_dead_locals = BitSet::new_empty(body.local_decls.len());
    for p in replacements.fields.keys() {
        all_dead_locals.insert(p.local);
    }
    debug!(?all_dead_locals);
    if all_dead_locals.is_empty() {
        return;
    }

    let mut fragments = IndexVec::new();
    for (k, v) in &replacements.fields {
        fragments.ensure_contains_elem(k.local, || Vec::new());
        fragments[k.local].push((&k.projection[..], *v));
    }
    debug!(?fragments);

    let mut visitor = ReplacementVisitor {
        tcx,
        local_decls: &body.local_decls,
        replacements,
        all_dead_locals,
        fragments,
    };
    for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        visitor.visit_basic_block_data(bb, data);
    }
    for scope in &mut body.source_scopes {
        visitor.visit_source_scope_data(scope);
    }
    for (index, annotation) in body.user_type_annotations.iter_enumerated_mut() {
        visitor.visit_user_type_annotation(index, annotation);
    }
    for var_debug_info in &mut body.var_debug_info {
        visitor.visit_var_debug_info(var_debug_info);
    }
}

struct ReplacementVisitor<'tcx, 'll> {
    tcx: TyCtxt<'tcx>,
    /// This is only used to compute the type for `VarDebugInfoContents::Composite`.
    local_decls: &'ll LocalDecls<'tcx>,
    /// Work to do.
    replacements: ReplacementMap<'tcx>,
    /// This is used to check that we are not leaving references to replaced locals behind.
    all_dead_locals: BitSet<Local>,
    /// Pre-computed list of all "new" locals for each "old" local.  This is used to expand storage
    /// and deinit statement and debuginfo.
    fragments: IndexVec<Local, Vec<(&'tcx [PlaceElem<'tcx>], Local)>>,
}

impl<'tcx, 'll> ReplacementVisitor<'tcx, 'll> {
    fn gather_debug_info_fragments(
        &self,
        place: PlaceRef<'tcx>,
    ) -> Vec<VarDebugInfoFragment<'tcx>> {
        let mut fragments = Vec::new();
        let parts = &self.fragments[place.local];
        for (proj, replacement_local) in parts {
            if proj.starts_with(place.projection) {
                fragments.push(VarDebugInfoFragment {
                    projection: proj[place.projection.len()..].to_vec(),
                    contents: Place::from(*replacement_local),
                });
            }
        }
        fragments
    }

    fn replace_place(&self, place: PlaceRef<'tcx>) -> Option<Place<'tcx>> {
        if let &[PlaceElem::Field(..), ref rest @ ..] = place.projection {
            let pr = PlaceRef { local: place.local, projection: &place.projection[..1] };
            let local = self.replacements.fields.get(&pr)?;
            Some(Place { local: *local, projection: self.tcx.intern_place_elems(&rest) })
        } else {
            None
        }
    }
}

impl<'tcx, 'll> MutVisitor<'tcx> for ReplacementVisitor<'tcx, 'll> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let StatementKind::StorageLive(..)
        | StatementKind::StorageDead(..)
        | StatementKind::Deinit(..) = statement.kind
        {
            // Storage statements are expanded in run_pass.
            return;
        }
        self.super_statement(statement, location)
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some(repl) = self.replace_place(place.as_ref()) {
            *place = repl
        } else {
            self.super_place(place, context, location)
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        match &mut var_debug_info.value {
            VarDebugInfoContents::Place(ref mut place) => {
                if let Some(repl) = self.replace_place(place.as_ref()) {
                    *place = repl;
                } else if self.all_dead_locals.contains(place.local) {
                    let ty = place.ty(self.local_decls, self.tcx).ty;
                    let fragments = self.gather_debug_info_fragments(place.as_ref());
                    var_debug_info.value = VarDebugInfoContents::Composite { ty, fragments };
                }
            }
            VarDebugInfoContents::Composite { ty: _, ref mut fragments } => {
                let mut new_fragments = Vec::new();
                fragments
                    .drain_filter(|fragment| {
                        if let Some(repl) = self.replace_place(fragment.contents.as_ref()) {
                            fragment.contents = repl;
                            true
                        } else if self.all_dead_locals.contains(fragment.contents.local) {
                            let frg = self.gather_debug_info_fragments(fragment.contents.as_ref());
                            new_fragments.extend(frg.into_iter().map(|mut f| {
                                f.projection.splice(0..0, fragment.projection.iter().copied());
                                f
                            }));
                            false
                        } else {
                            true
                        }
                    })
                    .for_each(drop);
                fragments.extend(new_fragments);
            }
            VarDebugInfoContents::Const(_) => {}
        }
    }

    fn visit_basic_block_data(&mut self, bb: BasicBlock, bbdata: &mut BasicBlockData<'tcx>) {
        self.super_basic_block_data(bb, bbdata);

        #[derive(Debug)]
        enum Stmt {
            StorageLive,
            StorageDead,
            Deinit,
        }

        bbdata.expand_statements(|stmt| {
            let source_info = stmt.source_info;
            let (stmt, origin_local) = match &stmt.kind {
                StatementKind::StorageLive(l) => (Stmt::StorageLive, *l),
                StatementKind::StorageDead(l) => (Stmt::StorageDead, *l),
                StatementKind::Deinit(p) if let Some(l) = p.as_local() => (Stmt::Deinit, l),
                _ => return None,
            };
            if !self.all_dead_locals.contains(origin_local) {
                return None;
            }
            let final_locals = self.fragments.get(origin_local)?;
            Some(final_locals.iter().map(move |&(_, l)| {
                let kind = match stmt {
                    Stmt::StorageLive => StatementKind::StorageLive(l),
                    Stmt::StorageDead => StatementKind::StorageDead(l),
                    Stmt::Deinit => StatementKind::Deinit(Box::new(l.into())),
                };
                Statement { source_info, kind }
            }))
        });
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert!(!self.all_dead_locals.contains(*local));
    }
}
