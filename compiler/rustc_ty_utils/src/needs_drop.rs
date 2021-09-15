//! Check whether a type has (potentially) non-trivial drop glue.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::util::{needs_drop_components, AlwaysRequiresDrop};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::Limit;
use rustc_span::{sym, DUMMY_SP};

type NeedsDropResult<T> = Result<T, AlwaysRequiresDrop>;

fn needs_drop_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    let adt_components =
        move |adt_def: &ty::AdtDef| tcx.adt_drop_tys(adt_def.did).map(|tys| tys.iter());

    // If we don't know a type doesn't need drop, for example if it's a type
    // parameter without a `Copy` bound, then we conservatively return that it
    // needs drop.
    let res =
        NeedsDropTypes::new(tcx, query.param_env, query.value, adt_components).next().is_some();

    debug!("needs_drop_raw({:?}) = {:?}", query, res);
    res
}

fn has_significant_drop_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> bool {
    let significant_drop_fields =
        move |adt_def: &ty::AdtDef| tcx.adt_significant_drop_tys(adt_def.did).map(|tys| tys.iter());
    let res = NeedsDropTypes::new(tcx, query.param_env, query.value, significant_drop_fields)
        .next()
        .is_some();
    debug!("has_significant_drop_raw({:?}) = {:?}", query, res);
    res
}

struct NeedsDropTypes<'tcx, F> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    query_ty: Ty<'tcx>,
    seen_tys: FxHashSet<Ty<'tcx>>,
    /// A stack of types left to process, and the recursion depth when we
    /// pushed that type. Each round, we pop something from the stack and check
    /// if it needs drop. If the result depends on whether some other types
    /// need drop we push them onto the stack.
    unchecked_tys: Vec<(Ty<'tcx>, usize)>,
    recursion_limit: Limit,
    adt_components: F,
}

impl<'tcx, F> NeedsDropTypes<'tcx, F> {
    fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        adt_components: F,
    ) -> Self {
        let mut seen_tys = FxHashSet::default();
        seen_tys.insert(ty);
        Self {
            tcx,
            param_env,
            seen_tys,
            query_ty: ty,
            unchecked_tys: vec![(ty, 0)],
            recursion_limit: tcx.recursion_limit(),
            adt_components,
        }
    }
}

impl<'tcx, F, I> Iterator for NeedsDropTypes<'tcx, F>
where
    F: Fn(&ty::AdtDef) -> NeedsDropResult<I>,
    I: Iterator<Item = Ty<'tcx>>,
{
    type Item = NeedsDropResult<Ty<'tcx>>;

    fn next(&mut self) -> Option<NeedsDropResult<Ty<'tcx>>> {
        let tcx = self.tcx;

        while let Some((ty, level)) = self.unchecked_tys.pop() {
            if !self.recursion_limit.value_within_limit(level) {
                // Not having a `Span` isn't great. But there's hopefully some other
                // recursion limit error as well.
                tcx.sess.span_err(
                    DUMMY_SP,
                    &format!("overflow while checking whether `{}` requires drop", self.query_ty),
                );
                return Some(Err(AlwaysRequiresDrop));
            }

            let components = match needs_drop_components(ty, &tcx.data_layout) {
                Err(e) => return Some(Err(e)),
                Ok(components) => components,
            };
            debug!("needs_drop_components({:?}) = {:?}", ty, components);

            let queue_type = move |this: &mut Self, component: Ty<'tcx>| {
                if this.seen_tys.insert(component) {
                    this.unchecked_tys.push((component, level + 1));
                }
            };

            for component in components {
                match *component.kind() {
                    _ if component.is_copy_modulo_regions(tcx.at(DUMMY_SP), self.param_env) => (),

                    ty::Closure(_, substs) => {
                        queue_type(self, substs.as_closure().tupled_upvars_ty());
                    }

                    ty::Generator(def_id, substs, _) => {
                        let substs = substs.as_generator();
                        queue_type(self, substs.tupled_upvars_ty());

                        let witness = substs.witness();
                        let interior_tys = match witness.kind() {
                            &ty::GeneratorWitness(tys) => tcx.erase_late_bound_regions(tys),
                            _ => {
                                tcx.sess.delay_span_bug(
                                    tcx.hir().span_if_local(def_id).unwrap_or(DUMMY_SP),
                                    &format!("unexpected generator witness type {:?}", witness),
                                );
                                return Some(Err(AlwaysRequiresDrop));
                            }
                        };

                        for interior_ty in interior_tys {
                            queue_type(self, interior_ty);
                        }
                    }

                    // Check for a `Drop` impl and whether this is a union or
                    // `ManuallyDrop`. If it's a struct or enum without a `Drop`
                    // impl then check whether the field types need `Drop`.
                    ty::Adt(adt_def, substs) => {
                        let tys = match (self.adt_components)(adt_def) {
                            Err(e) => return Some(Err(e)),
                            Ok(tys) => tys,
                        };
                        for required_ty in tys {
                            let subst_ty = tcx.normalize_erasing_regions(
                                self.param_env,
                                required_ty.subst(tcx, substs),
                            );
                            queue_type(self, subst_ty);
                        }
                    }
                    ty::Array(..) | ty::Opaque(..) | ty::Projection(..) | ty::Param(_) => {
                        if ty == component {
                            // Return the type to the caller: they may be able
                            // to normalize further than we can.
                            return Some(Ok(component));
                        } else {
                            // Store the type for later. We can't return here
                            // because we would then lose any other components
                            // of the type.
                            queue_type(self, component);
                        }
                    }
                    _ => return Some(Err(AlwaysRequiresDrop)),
                }
            }
        }

        None
    }
}

// This is a helper function for `adt_drop_tys` and `adt_significant_drop_tys`.
// Depending on the implentation of `adt_has_dtor`, it is used to check if the
// ADT has a destructor or if the ADT only has a significant destructor. For
// understanding significant destructor look at `adt_significant_drop_tys`.
fn adt_drop_tys_helper(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    adt_has_dtor: impl Fn(&ty::AdtDef) -> bool,
) -> Result<&ty::List<Ty<'_>>, AlwaysRequiresDrop> {
    let adt_components = move |adt_def: &ty::AdtDef| {
        if adt_def.is_manually_drop() {
            debug!("adt_drop_tys: `{:?}` is manually drop", adt_def);
            return Ok(Vec::new().into_iter());
        } else if adt_has_dtor(adt_def) {
            debug!("adt_drop_tys: `{:?}` implements `Drop`", adt_def);
            return Err(AlwaysRequiresDrop);
        } else if adt_def.is_union() {
            debug!("adt_drop_tys: `{:?}` is a union", adt_def);
            return Ok(Vec::new().into_iter());
        }
        Ok(adt_def.all_fields().map(|field| tcx.type_of(field.did)).collect::<Vec<_>>().into_iter())
    };

    let adt_ty = tcx.type_of(def_id);
    let param_env = tcx.param_env(def_id);
    let res: Result<Vec<_>, _> =
        NeedsDropTypes::new(tcx, param_env, adt_ty, adt_components).collect();

    debug!("adt_drop_tys(`{}`) = `{:?}`", tcx.def_path_str(def_id), res);
    res.map(|components| tcx.intern_type_list(&components))
}

fn adt_drop_tys(tcx: TyCtxt<'_>, def_id: DefId) -> Result<&ty::List<Ty<'_>>, AlwaysRequiresDrop> {
    let adt_has_dtor = |adt_def: &ty::AdtDef| adt_def.destructor(tcx).is_some();
    adt_drop_tys_helper(tcx, def_id, adt_has_dtor)
}

fn adt_significant_drop_tys(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Result<&ty::List<Ty<'_>>, AlwaysRequiresDrop> {
    let adt_has_dtor = |adt_def: &ty::AdtDef| {
        adt_def
            .destructor(tcx)
            .map(|dtor| !tcx.has_attr(dtor.did, sym::rustc_insignificant_dtor))
            .unwrap_or(false)
    };
    adt_drop_tys_helper(tcx, def_id, adt_has_dtor)
}

pub(crate) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        needs_drop_raw,
        has_significant_drop_raw,
        adt_drop_tys,
        adt_significant_drop_tys,
        ..*providers
    };
}
