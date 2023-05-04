use crate::infer::region_constraints::MemberConstraint;
use rustc_middle::ty;
use ty::subst::{GenericArg, GenericArgKind};
use ty::{Const, OutlivesPredicate, Placeholder, Region, Ty, TyCtxt};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::sync::Lrc;

pub type Outlives<'tcx> = OutlivesPredicate<GenericArg<'tcx>, Region<'tcx>>;

/// Walks over constraints and fetches variables present in the constraint, in order
/// Also has the ability to re-write the variables there
pub struct ConstraintWalker<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    fetch_var: &'a mut dyn FnMut(usize) -> usize,
    pub vars: Vec<usize>,
}
impl<'tcx, 'a> ConstraintWalker<'tcx, 'a> {
    pub fn new(tcx: TyCtxt<'tcx>, fetch_var: &'a mut dyn FnMut(usize) -> usize) -> Self {
        Self { tcx, fetch_var: fetch_var, vars: Vec::new() }
    }
    pub fn reset(&mut self) {
        self.vars.clear();
    }
    pub fn add_var(&mut self, var: usize) -> usize {
        self.vars.push(var);
        return (self.fetch_var)(var);
    }

    // The walk functions recursively walk over an Enum, extracting the variables present inside (in order)
    // Then, a substitution for each variable is computed using the fetch_var closure stored as a property
    // If we don't want to re-write variables, we can just use |x| x as the fetch_var closure

    pub fn walk_outlives(&mut self, input: &Outlives<'tcx>) -> Outlives<'tcx> {
        ty::OutlivesPredicate(
            self.walk_generic_arg(input.0.clone()),
            self.walk_region(input.1.clone()),
        )
    }
    pub fn walk_members(&mut self, input: &MemberConstraint<'tcx>) -> MemberConstraint<'tcx> {
        MemberConstraint {
            key: ty::OpaqueTypeKey {
                def_id: input.key.def_id,
                substs: self.walk_substs_ref(input.key.substs),
            },
            definition_span: input.definition_span,
            hidden_ty: self.walk_ty(input.hidden_ty),
            member_region: self.walk_region(input.member_region),
            choice_regions: Lrc::new(
                input.choice_regions.iter().map(|x| self.walk_region(*x)).collect::<Vec<_>>(),
            ),
        }
    }
    fn walk_generic_arg(&mut self, input: GenericArg<'tcx>) -> GenericArg<'tcx> {
        match input.unpack() {
            GenericArgKind::Lifetime(region) => GenericArg::<'tcx>::from(self.walk_region(region)),
            GenericArgKind::Const(const_var) => {
                GenericArg::<'tcx>::from(self.walk_const(const_var))
            }
            _ => input,
        }
    }
    fn walk_region(&mut self, input: Region<'tcx>) -> Region<'tcx> {
        let rewritten = match input.kind() {
            // FIXME: Are these all of the variants that can have variables?
            ty::ReLateBound(db_indx, bound_region) => {
                ty::ReLateBound(db_indx, self.walk_bound_region(bound_region))
            }
            ty::ReVar(region_id) => {
                ty::ReVar(ty::RegionVid::from_usize(self.add_var(region_id.index())))
            }
            ty::RePlaceholder(region) => ty::RePlaceholder(Placeholder {
                universe: region.universe,
                bound: self.walk_bound_region(region.bound),
            }),
            _ => return input,
        };
        self.tcx.mk_region_from_kind(rewritten)
    }
    fn walk_ty(&mut self, input: Ty<'tcx>) -> Ty<'tcx> {
        let rewritten = match input.kind() {
            // FIXME: Quite a few are missing
            ty::Adt(adt_def, substs) => ty::Adt(*adt_def, self.walk_substs_ref(substs)),
            ty::Array(elem_ty, count) => ty::Array(self.walk_ty(*elem_ty), self.walk_const(*count)),
            ty::Slice(elem_ty) => ty::Slice(self.walk_ty(*elem_ty)),
            ty::RawPtr(ptr) => {
                ty::RawPtr(ty::TypeAndMut { ty: self.walk_ty(ptr.ty), mutbl: ptr.mutbl })
            }
            ty::Ref(ref_region, ref_ty, ref_mutbl) => {
                ty::Ref(self.walk_region(*ref_region), self.walk_ty(*ref_ty), *ref_mutbl)
            }
            ty::FnDef(def_id, substs) => ty::FnDef(*def_id, self.walk_substs_ref(substs)),
            ty::Tuple(elems) => ty::Tuple(
                self.tcx
                    .mk_type_list(&elems.into_iter().map(|x| self.walk_ty(x)).collect::<Vec<_>>()),
            ),
            ty::Bound(indx, bound_ty) => ty::Bound(*indx, self.walk_bound_ty(*bound_ty)),
            ty::Placeholder(placeholder) => ty::Placeholder(Placeholder {
                universe: placeholder.universe,
                bound: self.walk_bound_ty(placeholder.bound),
            }),
            _ => return input,
        };
        self.tcx.mk_ty_from_kind(rewritten)
    }
    fn walk_const(&mut self, input: Const<'tcx>) -> Const<'tcx> {
        let rewritten_ty = self.walk_ty(input.ty());
        match input.kind() {
            ty::ConstKind::Param(param) => self.tcx.mk_const(
                ty::ParamConst {
                    index: self.add_var(param.index as usize) as u32,
                    name: param.name,
                },
                rewritten_ty,
            ),
            _ => input,
        }
    }
    fn walk_bound_region(&mut self, input: ty::BoundRegion) -> ty::BoundRegion {
        ty::BoundRegion {
            var: ty::BoundVar::from_usize(self.add_var(input.var.index())),
            kind: input.kind,
        }
    }
    fn walk_bound_ty(&mut self, input: ty::BoundTy) -> ty::BoundTy {
        ty::BoundTy {
            var: ty::BoundVar::from_usize(self.add_var(input.var.index())),
            kind: input.kind,
        }
    }
    fn walk_substs_ref(
        &mut self,
        input: &'tcx ty::List<GenericArg<'tcx>>,
    ) -> &'tcx ty::List<GenericArg<'tcx>> {
        self.tcx.mk_substs(&input.into_iter().map(|x| self.walk_generic_arg(x)).collect::<Vec<_>>())
    }
}
