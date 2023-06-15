use crate::MirPass;
use rustc_index::bit_set::{BitSet, GrowableBitSet};
use rustc_index::IndexVec;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::value_analysis::{excluded_locals, iter_fields};
use rustc_target::abi::{FieldIdx, ReprFlags, FIRST_VARIANT};

pub struct ScalarReplacementOfAggregates;

impl<'tcx> MirPass<'tcx> for ScalarReplacementOfAggregates {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[instrument(level = "debug", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());

        // Avoid query cycles (generators require optimized MIR for layout).
        if tcx.type_of(body.source.def_id()).subst_identity().is_generator() {
            return;
        }

        let mut excluded = excluded_locals(body);
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        loop {
            debug!(?excluded);
            let escaping = escaping_locals(tcx, param_env, &excluded, body);
            debug!(?escaping);
            let replacements = compute_flattening(tcx, param_env, body, escaping);
            debug!(?replacements);
            let all_dead_locals = replace_flattened_locals(tcx, body, replacements);
            if !all_dead_locals.is_empty() {
                excluded.union(&all_dead_locals);
                excluded = {
                    let mut growable = GrowableBitSet::from(excluded);
                    growable.ensure(body.local_decls.len());
                    growable.into()
                };
            } else {
                break;
            }
        }
    }
}

/// Identify all locals that are not eligible for SROA.
///
/// There are 3 cases:
/// - the aggregated local is used or passed to other code (function parameters and arguments);
/// - the locals is a union or an enum;
/// - the local's address is taken, and thus the relative addresses of the fields are observable to
///   client code.
fn escaping_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    excluded: &BitSet<Local>,
    body: &Body<'tcx>,
) -> BitSet<Local> {
    let is_excluded_ty = |ty: Ty<'tcx>| {
        if ty.is_union() || ty.is_enum() {
            return true;
        }
        if let ty::Adt(def, _substs) = ty.kind() {
            if def.repr().flags.contains(ReprFlags::IS_SIMD) {
                // Exclude #[repr(simd)] types so that they are not de-optimized into an array
                return true;
            }
            // We already excluded unions and enums, so this ADT must have one variant
            let variant = def.variant(FIRST_VARIANT);
            if variant.fields.len() > 1 {
                // If this has more than one field, it cannot be a wrapper that only provides a
                // niche, so we do not want to automatically exclude it.
                return false;
            }
            let Ok(layout) = tcx.layout_of(param_env.and(ty)) else {
                // We can't get the layout
                return true;
            };
            if layout.layout.largest_niche().is_some() {
                // This type has a niche
                return true;
            }
        }
        // Default for non-ADTs
        false
    };

    let mut set = BitSet::new_empty(body.local_decls.len());
    set.insert_range(RETURN_PLACE..=Local::from_usize(body.arg_count));
    for (local, decl) in body.local_decls().iter_enumerated() {
        if excluded.contains(local) || is_excluded_ty(decl.ty) {
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

        fn visit_assign(
            &mut self,
            lvalue: &Place<'tcx>,
            rvalue: &Rvalue<'tcx>,
            location: Location,
        ) {
            if lvalue.as_local().is_some() {
                match rvalue {
                    // Aggregate assignments are expanded in run_pass.
                    Rvalue::Aggregate(..) | Rvalue::Use(..) => {
                        self.visit_rvalue(rvalue, location);
                        return;
                    }
                    _ => {}
                }
            }
            self.super_assign(lvalue, rvalue, location)
        }

        fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
            match statement.kind {
                // Storage statements are expanded in run_pass.
                StatementKind::StorageLive(..)
                | StatementKind::StorageDead(..)
                | StatementKind::Deinit(..) => return,
                _ => self.super_statement(statement, location),
            }
        }

        // We ignore anything that happens in debuginfo, since we expand it using
        // `VarDebugInfoContents::Composite`.
        fn visit_var_debug_info(&mut self, _: &VarDebugInfo<'tcx>) {}
    }
}

#[derive(Default, Debug)]
struct ReplacementMap<'tcx> {
    /// Pre-computed list of all "new" locals for each "old" local. This is used to expand storage
    /// and deinit statement and debuginfo.
    fragments: IndexVec<Local, Option<IndexVec<FieldIdx, Option<(Ty<'tcx>, Local)>>>>,
}

impl<'tcx> ReplacementMap<'tcx> {
    fn replace_place(&self, tcx: TyCtxt<'tcx>, place: PlaceRef<'tcx>) -> Option<Place<'tcx>> {
        let &[PlaceElem::Field(f, _), ref rest @ ..] = place.projection else { return None; };
        let fields = self.fragments[place.local].as_ref()?;
        let (_, new_local) = fields[f]?;
        Some(Place { local: new_local, projection: tcx.mk_place_elems(&rest) })
    }

    fn place_fragments(
        &self,
        place: Place<'tcx>,
    ) -> Option<impl Iterator<Item = (FieldIdx, Ty<'tcx>, Local)> + '_> {
        let local = place.as_local()?;
        let fields = self.fragments[local].as_ref()?;
        Some(fields.iter_enumerated().filter_map(|(field, &opt_ty_local)| {
            let (ty, local) = opt_ty_local?;
            Some((field, ty, local))
        }))
    }
}

/// Compute the replacement of flattened places into locals.
///
/// For each eligible place, we assign a new local to each accessed field.
/// The replacement will be done later in `ReplacementVisitor`.
fn compute_flattening<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body: &mut Body<'tcx>,
    escaping: BitSet<Local>,
) -> ReplacementMap<'tcx> {
    let mut fragments = IndexVec::from_elem(None, &body.local_decls);

    for local in body.local_decls.indices() {
        if escaping.contains(local) {
            continue;
        }
        let decl = body.local_decls[local].clone();
        let ty = decl.ty;
        iter_fields(ty, tcx, param_env, |variant, field, field_ty| {
            if variant.is_some() {
                // Downcasts are currently not supported.
                return;
            };
            let new_local =
                body.local_decls.push(LocalDecl { ty: field_ty, user_ty: None, ..decl.clone() });
            fragments.get_or_insert_with(local, IndexVec::new).insert(field, (field_ty, new_local));
        });
    }
    ReplacementMap { fragments }
}

/// Perform the replacement computed by `compute_flattening`.
fn replace_flattened_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    replacements: ReplacementMap<'tcx>,
) -> BitSet<Local> {
    let mut all_dead_locals = BitSet::new_empty(replacements.fragments.len());
    for (local, replacements) in replacements.fragments.iter_enumerated() {
        if replacements.is_some() {
            all_dead_locals.insert(local);
        }
    }
    debug!(?all_dead_locals);
    if all_dead_locals.is_empty() {
        return all_dead_locals;
    }

    let mut visitor = ReplacementVisitor {
        tcx,
        local_decls: &body.local_decls,
        replacements: &replacements,
        all_dead_locals,
        patch: MirPatch::new(body),
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
    let ReplacementVisitor { patch, all_dead_locals, .. } = visitor;
    patch.apply(body);
    all_dead_locals
}

struct ReplacementVisitor<'tcx, 'll> {
    tcx: TyCtxt<'tcx>,
    /// This is only used to compute the type for `VarDebugInfoContents::Composite`.
    local_decls: &'ll LocalDecls<'tcx>,
    /// Work to do.
    replacements: &'ll ReplacementMap<'tcx>,
    /// This is used to check that we are not leaving references to replaced locals behind.
    all_dead_locals: BitSet<Local>,
    patch: MirPatch<'tcx>,
}

impl<'tcx> ReplacementVisitor<'tcx, '_> {
    fn gather_debug_info_fragments(&self, local: Local) -> Option<Vec<VarDebugInfoFragment<'tcx>>> {
        let mut fragments = Vec::new();
        let parts = self.replacements.place_fragments(local.into())?;
        for (field, ty, replacement_local) in parts {
            fragments.push(VarDebugInfoFragment {
                projection: vec![PlaceElem::Field(field, ty)],
                contents: Place::from(replacement_local),
            });
        }
        Some(fragments)
    }
}

impl<'tcx, 'll> MutVisitor<'tcx> for ReplacementVisitor<'tcx, 'll> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some(repl) = self.replacements.replace_place(self.tcx, place.as_ref()) {
            *place = repl
        } else {
            self.super_place(place, context, location)
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match statement.kind {
            // Duplicate storage and deinit statements, as they pretty much apply to all fields.
            StatementKind::StorageLive(l) => {
                if let Some(final_locals) = self.replacements.place_fragments(l.into()) {
                    for (_, _, fl) in final_locals {
                        self.patch.add_statement(location, StatementKind::StorageLive(fl));
                    }
                    statement.make_nop();
                }
                return;
            }
            StatementKind::StorageDead(l) => {
                if let Some(final_locals) = self.replacements.place_fragments(l.into()) {
                    for (_, _, fl) in final_locals {
                        self.patch.add_statement(location, StatementKind::StorageDead(fl));
                    }
                    statement.make_nop();
                }
                return;
            }
            StatementKind::Deinit(box place) => {
                if let Some(final_locals) = self.replacements.place_fragments(place) {
                    for (_, _, fl) in final_locals {
                        self.patch
                            .add_statement(location, StatementKind::Deinit(Box::new(fl.into())));
                    }
                    statement.make_nop();
                    return;
                }
            }

            // We have `a = Struct { 0: x, 1: y, .. }`.
            // We replace it by
            // ```
            // a_0 = x
            // a_1 = y
            // ...
            // ```
            StatementKind::Assign(box (place, Rvalue::Aggregate(_, ref mut operands))) => {
                if let Some(local) = place.as_local()
                    && let Some(final_locals) = &self.replacements.fragments[local]
                {
                    // This is ok as we delete the statement later.
                    let operands = std::mem::take(operands);
                    for (&opt_ty_local, mut operand) in final_locals.iter().zip(operands) {
                        if let Some((_, new_local)) = opt_ty_local {
                            // Replace mentions of SROA'd locals that appear in the operand.
                            self.visit_operand(&mut operand, location);

                            let rvalue = Rvalue::Use(operand);
                            self.patch.add_statement(
                                location,
                                StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                            );
                        }
                    }
                    statement.make_nop();
                    return;
                }
            }

            // We have `a = some constant`
            // We add the projections.
            // ```
            // a_0 = a.0
            // a_1 = a.1
            // ...
            // ```
            // ConstProp will pick up the pieces and replace them by actual constants.
            StatementKind::Assign(box (place, Rvalue::Use(Operand::Constant(_)))) => {
                if let Some(final_locals) = self.replacements.place_fragments(place) {
                    // Put the deaggregated statements *after* the original one.
                    let location = location.successor_within_block();
                    for (field, ty, new_local) in final_locals {
                        let rplace = self.tcx.mk_place_field(place, field, ty);
                        let rvalue = Rvalue::Use(Operand::Move(rplace));
                        self.patch.add_statement(
                            location,
                            StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                        );
                    }
                    // We still need `place.local` to exist, so don't make it nop.
                    return;
                }
            }

            // We have `a = move? place`
            // We replace it by
            // ```
            // a_0 = move? place.0
            // a_1 = move? place.1
            // ...
            // ```
            StatementKind::Assign(box (lhs, Rvalue::Use(ref op))) => {
                let (rplace, copy) = match *op {
                    Operand::Copy(rplace) => (rplace, true),
                    Operand::Move(rplace) => (rplace, false),
                    Operand::Constant(_) => bug!(),
                };
                if let Some(final_locals) = self.replacements.place_fragments(lhs) {
                    for (field, ty, new_local) in final_locals {
                        let rplace = self.tcx.mk_place_field(rplace, field, ty);
                        debug!(?rplace);
                        let rplace = self
                            .replacements
                            .replace_place(self.tcx, rplace.as_ref())
                            .unwrap_or(rplace);
                        debug!(?rplace);
                        let rvalue = if copy {
                            Rvalue::Use(Operand::Copy(rplace))
                        } else {
                            Rvalue::Use(Operand::Move(rplace))
                        };
                        self.patch.add_statement(
                            location,
                            StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                        );
                    }
                    statement.make_nop();
                    return;
                }
            }

            _ => {}
        }
        self.super_statement(statement, location)
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        match &mut var_debug_info.value {
            VarDebugInfoContents::Place(ref mut place) => {
                if let Some(repl) = self.replacements.replace_place(self.tcx, place.as_ref()) {
                    *place = repl;
                } else if let Some(local) = place.as_local()
                    && let Some(fragments) = self.gather_debug_info_fragments(local)
                {
                    let ty = place.ty(self.local_decls, self.tcx).ty;
                    var_debug_info.value = VarDebugInfoContents::Composite { ty, fragments };
                }
            }
            VarDebugInfoContents::Composite { ty: _, ref mut fragments } => {
                let mut new_fragments = Vec::new();
                debug!(?fragments);
                fragments.retain_mut(|fragment| {
                    if let Some(repl) =
                            self.replacements.replace_place(self.tcx, fragment.contents.as_ref())
                        {
                            fragment.contents = repl;
                            true
                        } else if let Some(local) = fragment.contents.as_local()
                            && let Some(frg) = self.gather_debug_info_fragments(local)
                        {
                            new_fragments.extend(frg.into_iter().map(|mut f| {
                                f.projection.splice(0..0, fragment.projection.iter().copied());
                                f
                            }));
                            false
                        } else {
                            true
                        }
                });
                debug!(?fragments);
                debug!(?new_fragments);
                fragments.extend(new_fragments);
            }
            VarDebugInfoContents::Const(_) => {}
        }
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert!(!self.all_dead_locals.contains(*local));
    }
}
