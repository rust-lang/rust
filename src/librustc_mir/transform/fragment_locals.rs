use crate::transform::{MirPass, MirSource};
use rustc::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc::mir::*;
use rustc::ty::layout::VariantIdx;
use rustc::ty::util::IntTypeExt;
use rustc::ty::{self, Ty, TyCtxt};
use rustc_index::vec::IndexVec;
use rustc_span::Span;
use std::collections::BTreeMap;
use std::iter::Step;
use std::ops::Range;

pub struct FragmentLocals;

impl MirPass<'tcx> for FragmentLocals {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        let mut collector = FragmentTreeCollector {
            tcx,
            locals: body.local_decls.iter().map(|decl| FragmentTree::new(decl.ty)).collect(),
        };

        // Can't fragment return and arguments.
        collector.locals[RETURN_PLACE].make_opaque();
        for arg in body.args_iter() {
            collector.locals[arg].make_opaque();
        }

        collector.visit_body(read_only!(body));

        // Enforce current limitations of `VarDebugInfo` wrt fragmentation.
        for var_debug_info in &body.var_debug_info {
            match &var_debug_info.contents {
                VarDebugInfoContents::Compact(place) => {
                    if let Some(node) = collector.place_node(place) {
                        node.ensure_debug_info_compatible_descendents();
                    }
                }
                VarDebugInfoContents::Composite { ty: _, fragments } => {
                    for fragment in fragments {
                        if let Some(node) = collector.place_node(&fragment.contents) {
                            node.ensure_debug_info_compatible_descendents();
                        }
                    }
                }
            }
        }

        let replacements = collector
            .locals
            .iter_enumerated_mut()
            .map(|(local, root)| {
                // Don't rename locals that are entirely opaque.
                match root.kind {
                    FragmentTreeNodeKind::OpaqueLeaf { .. } => local..local.add_one(),
                    FragmentTreeNodeKind::Nested(_) => {
                        let source_info = body.local_decls[local].source_info;
                        let first = body.local_decls.next_index();
                        root.assign_locals(&mut body.local_decls, source_info);
                        first..body.local_decls.next_index()
                    }
                }
            })
            .collect::<IndexVec<Local, Range<Local>>>();

        // Expand `Storage{Live,Dead}` statements to refer to the replacement locals.
        for bb in body.basic_blocks_mut() {
            bb.expand_statements(|stmt| {
                let (local, is_live) = match stmt.kind {
                    StatementKind::StorageLive(local) => (local, true),
                    StatementKind::StorageDead(local) => (local, false),
                    _ => return None,
                };
                let range = replacements[local].clone();
                // FIXME(eddyb) `Range<Local>` should itself be iterable.
                let range = (range.start.as_u32()..range.end.as_u32()).map(Local::from_u32);
                let source_info = stmt.source_info;
                Some(range.map(move |new_local| Statement {
                    source_info,
                    kind: if is_live {
                        StatementKind::StorageLive(new_local)
                    } else {
                        StatementKind::StorageDead(new_local)
                    },
                }))
            });
        }
        drop(replacements);

        // Lastly, replace all the opaque nodes with their new locals.
        let mut replacer = FragmentTreeReplacer { tcx, span: body.span, locals: collector.locals };
        replacer.visit_body(body);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Fragment {
    Discriminant,
    Field(Option<VariantIdx>, Field),
}

struct FragmentTree<'tcx> {
    ty: Ty<'tcx>,
    kind: FragmentTreeNodeKind<'tcx>,
}

// FIXME(eddyb) find a shorter name for this
enum FragmentTreeNodeKind<'tcx> {
    /// This node needs to remain compact, e.g. due to accesses / borrows.
    OpaqueLeaf { replacement_local: Option<Local> },

    /// This node can be fragmented into separate locals for its fields.
    Nested(BTreeMap<Fragment, FragmentTree<'tcx>>),
}

impl FragmentTree<'tcx> {
    fn new(ty: Ty<'tcx>) -> Self {
        let mut node = FragmentTree { ty, kind: FragmentTreeNodeKind::Nested(BTreeMap::new()) };

        if let ty::Adt(adt_def, _) = ty.kind {
            // Unions have (observably) overlapping members, so don't fragment them.
            if adt_def.is_union() {
                node.make_opaque();
            }
        }

        node
    }

    fn fragment(&mut self, fragment: Fragment, ty: Ty<'tcx>) -> Option<&mut Self> {
        match self.kind {
            FragmentTreeNodeKind::Nested(ref mut fragments) => {
                Some(fragments.entry(fragment).or_insert_with(|| FragmentTree::new(ty)))
            }
            FragmentTreeNodeKind::OpaqueLeaf { .. } => None,
        }
    }

    fn discriminant(&mut self, tcx: TyCtxt<'tcx>) -> Option<&mut Self> {
        match self.ty.kind {
            ty::Adt(adt_def, _) if adt_def.is_enum() => {
                let discr_ty = adt_def.repr.discr_type().to_ty(tcx);
                self.fragment(Fragment::Discriminant, discr_ty)
            }
            _ => None,
        }
    }

    fn make_opaque(&mut self) {
        if let FragmentTreeNodeKind::Nested(_) = self.kind {
            self.kind = FragmentTreeNodeKind::OpaqueLeaf { replacement_local: None };
        }
    }

    /// Make any descendent node which has discriminant/variant fragments opaque,
    /// as `enum`s (and similarly, generators) are not compatible with variable
    /// debuginfo currently (also see comments in `VarDebugInfoFragment`).
    fn ensure_debug_info_compatible_descendents(&mut self) {
        if let FragmentTreeNodeKind::Nested(ref mut fragments) = self.kind {
            let enum_like = fragments.keys().any(|f| match f {
                Fragment::Discriminant => true,
                Fragment::Field(variant_index, _) => variant_index.is_some(),
            });
            if enum_like {
                self.make_opaque();
            } else {
                for fragment in fragments.values_mut() {
                    fragment.ensure_debug_info_compatible_descendents();
                }
            }
        }
    }

    fn project(
        mut self: &'a mut Self,
        mut proj_elems: &'tcx [PlaceElem<'tcx>],
    ) -> (&'a mut Self, &'tcx [PlaceElem<'tcx>]) {
        let mut variant_index = None;
        while let [elem, rest @ ..] = proj_elems {
            if let FragmentTreeNodeKind::OpaqueLeaf { .. } = self.kind {
                break;
            }

            match *elem {
                ProjectionElem::Field(f, ty) => {
                    let field = Fragment::Field(variant_index, f);
                    // FIXME(eddyb) use `self.fragment(field)` post-Polonius(?).
                    match self.kind {
                        FragmentTreeNodeKind::Nested(ref mut fragments) => {
                            self = fragments.entry(field).or_insert_with(|| FragmentTree::new(ty));
                        }
                        FragmentTreeNodeKind::OpaqueLeaf { .. } => unreachable!(),
                    }
                }

                ProjectionElem::Downcast(..) => {}

                // FIXME(eddyb) support indexing by constants.
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } |
                // Can't support without alias analysis.
                ProjectionElem::Index(_) | ProjectionElem::Deref => {
                    // If we can't project, we must be opaque.
                    self.make_opaque();
                    break;
                }
            }

            proj_elems = rest;
            variant_index = match *elem {
                ProjectionElem::Downcast(_, v) => Some(v),
                _ => None,
            };
        }

        (self, proj_elems)
    }

    fn assign_locals(
        &mut self,
        local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
        source_info: SourceInfo,
    ) {
        match self.kind {
            FragmentTreeNodeKind::OpaqueLeaf { ref mut replacement_local } => {
                let mut decl = LocalDecl::new_internal(self.ty, source_info.span);
                decl.source_info = source_info;
                *replacement_local = Some(local_decls.push(decl));
            }
            FragmentTreeNodeKind::Nested(ref mut fragments) => {
                for fragment in fragments.values_mut() {
                    fragment.assign_locals(local_decls, source_info);
                }
            }
        }
    }

    /// Push debuginfo for all leaves into `fragments`, pointing to
    /// their respective `replacement_local`s (set by `assign_locals`).
    fn gather_debug_info_fragments(
        &self,
        dbg_fragment_projection: &mut Vec<ProjectionKind>,
        dbg_fragments: &mut Vec<VarDebugInfoFragment<'tcx>>,
    ) {
        match self.kind {
            FragmentTreeNodeKind::OpaqueLeaf { replacement_local } => {
                dbg_fragments.push(VarDebugInfoFragment {
                    projection: dbg_fragment_projection.clone(),
                    contents: Place::from(replacement_local.expect("missing replacement")),
                })
            }
            FragmentTreeNodeKind::Nested(ref fragments) => {
                for (&f, fragment) in fragments {
                    match f {
                        Fragment::Discriminant => unreachable!(),
                        Fragment::Field(variant_index, field) => {
                            assert_eq!(variant_index, None);

                            dbg_fragment_projection.push(ProjectionElem::Field(field, ()));
                        }
                    }
                    fragment.gather_debug_info_fragments(dbg_fragment_projection, dbg_fragments);
                    dbg_fragment_projection.pop();
                }
            }
        }
    }
}

struct FragmentTreeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    locals: IndexVec<Local, FragmentTree<'tcx>>,
}

impl FragmentTreeCollector<'tcx> {
    fn place_node(&'a mut self, place: &Place<'tcx>) -> Option<&'a mut FragmentTree<'tcx>> {
        let (node, proj_elems) = self.locals[place.local].project(place.projection);
        if proj_elems.is_empty() { Some(node) } else { None }
    }
}

impl Visitor<'tcx> for FragmentTreeCollector<'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
        // Borrows of fields might be used to access the entire local,
        // by unsafe code, so it's better for the time being to remain
        // conservative, until such uses have been definitely deemed UB.
        if context.is_borrow() {
            self.locals[place.local].make_opaque();
        }

        if let Some(node) = self.place_node(place) {
            if context.is_use() {
                node.make_opaque();
            }
        }
    }

    // Special-case `(Set)Discriminant(place)` to only mark `Fragment::Discriminant` as opaque.
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx;

        if let Rvalue::Discriminant(ref place) = *rvalue {
            if let Some(node) = self.place_node(place) {
                if let Some(discr) = node.discriminant(tcx) {
                    discr.make_opaque();
                }
            }
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        let tcx = self.tcx;

        if let StatementKind::SetDiscriminant { ref place, .. } = statement.kind {
            if let Some(node) = self.place_node(place) {
                if let Some(discr) = node.discriminant(tcx) {
                    discr.make_opaque();
                }
            }
        } else {
            self.super_statement(statement, location);
        }
    }
}

struct FragmentTreeReplacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    span: Span,
    locals: IndexVec<Local, FragmentTree<'tcx>>,
}

impl FragmentTreeReplacer<'tcx> {
    fn replace(
        &mut self,
        place: &Place<'tcx>,
    ) -> Option<Result<Place<'tcx>, &mut FragmentTree<'tcx>>> {
        let base_node = &mut self.locals[place.local];

        // Avoid identity replacements, which would re-intern projections.
        if let FragmentTreeNodeKind::OpaqueLeaf { replacement_local: None } = base_node.kind {
            return None;
        }

        let (node, proj_elems) = base_node.project(place.projection);

        Some(match node.kind {
            FragmentTreeNodeKind::OpaqueLeaf { replacement_local } => Ok(Place {
                local: replacement_local.expect("missing replacement"),
                projection: self.tcx.intern_place_elems(proj_elems),
            }),

            // HACK(eddyb) this only exists to support `(Set)Discriminant` below.
            FragmentTreeNodeKind::Nested(_) => {
                assert_eq!(proj_elems, &[]);

                Err(node)
            }
        })
    }
}

impl MutVisitor<'tcx> for FragmentTreeReplacer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext, _: Location) {
        if let Some(place_replacement) = self.replace(place) {
            match place_replacement {
                Ok(place_replacement) => *place = place_replacement,
                // HACK(eddyb) this only exists to support `(Set)Discriminant` below.
                Err(_) => unreachable!(),
            }
        }
    }

    // Break up `VarDebugInfo` into fragments where necessary.
    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        match &mut var_debug_info.contents {
            VarDebugInfoContents::Compact(place) => {
                if let Some(place_replacement) = self.replace(place) {
                    match place_replacement {
                        Ok(place_replacement) => *place = place_replacement,
                        Err(node) => {
                            let mut fragments = vec![];
                            node.gather_debug_info_fragments(&mut vec![], &mut fragments);
                            var_debug_info.contents =
                                VarDebugInfoContents::Composite { ty: node.ty, fragments };
                        }
                    }
                }
            }
            VarDebugInfoContents::Composite { ty: _, fragments } => {
                for fragment in fragments {
                    if let Some(place_replacement) = self.replace(&fragment.contents) {
                        match place_replacement {
                            Ok(place_replacement) => fragment.contents = place_replacement,
                            // FIXME(eddyb) implement!!
                            Err(_) => span_bug!(
                                var_debug_info.source_info.span,
                                "FIXME: implement fragmentation for {:?}",
                                var_debug_info,
                            ),
                        }
                    }
                }
            }
        }
    }

    // Special-case `(Set)Discriminant(place)` to use `discr_local` for `place`.
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx;
        let span = self.span;

        if let Rvalue::Discriminant(ref mut place) = rvalue {
            if let Some(place_replacement) = self.replace(place) {
                match place_replacement {
                    Ok(place_replacement) => *place = place_replacement,
                    Err(node) => {
                        let discr = if let Some(discr) = node.discriminant(tcx) {
                            let discr = match discr.kind {
                                FragmentTreeNodeKind::OpaqueLeaf { replacement_local } => {
                                    replacement_local
                                }
                                FragmentTreeNodeKind::Nested(_) => unreachable!(),
                            };
                            Operand::Copy(Place::from(discr.expect("missing discriminant")))
                        } else {
                            // Non-enums don't have discriminants other than `0u8`.
                            let discr_value = ty::Const::from_bits(
                                tcx,
                                0,
                                ty::ParamEnv::empty().and(tcx.types.u8),
                            );
                            Operand::Constant(box Constant {
                                span,
                                user_ty: None,
                                literal: discr_value,
                            })
                        };
                        *rvalue = Rvalue::Use(discr);
                    }
                }
            }
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        self.span = statement.source_info.span;

        let tcx = self.tcx;
        let span = self.span;

        if let StatementKind::SetDiscriminant { ref mut place, variant_index } = statement.kind {
            if let Some(place_replacement) = self.replace(place) {
                match place_replacement {
                    Ok(place_replacement) => **place = place_replacement,
                    Err(node) => {
                        if let Some(discr) = node.discriminant(tcx) {
                            let discr_ty = discr.ty;
                            let discr_local = match discr.kind {
                                FragmentTreeNodeKind::OpaqueLeaf { replacement_local } => {
                                    replacement_local
                                }
                                FragmentTreeNodeKind::Nested(_) => unreachable!(),
                            };
                            let discr_place =
                                Place::from(discr_local.expect("missing discriminant"));
                            let discr_value = ty::Const::from_bits(
                                tcx,
                                node.ty.discriminant_for_variant(tcx, variant_index).unwrap().val,
                                ty::ParamEnv::empty().and(discr_ty),
                            );
                            let discr_rvalue = Rvalue::Use(Operand::Constant(box Constant {
                                span,
                                user_ty: None,
                                literal: discr_value,
                            }));
                            statement.kind = StatementKind::Assign(box (discr_place, discr_rvalue));
                        } else {
                            // Non-enums don't have discriminants to set.
                            statement.kind = StatementKind::Nop;
                        }
                    }
                }
            }
        } else {
            self.super_statement(statement, location);
        }
    }
}
