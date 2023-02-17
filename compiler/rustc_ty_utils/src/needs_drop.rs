//! Check whether a type has (potentially) non-trivial drop glue.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::util::{needs_drop_components, AlwaysRequiresDrop};
use rustc_middle::ty::{self, EarlyBinder, Ty, TyCtxt};
use rustc_session::Limit;
use rustc_span::{sym, DUMMY_SP};

use crate::errors::NeedsDropOverflow;

type NeedsDropResult<T> = Result<T, AlwaysRequiresDrop>;

fn needs_drop_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    // If we don't know a type doesn't need drop, for example if it's a type
    // parameter without a `Copy` bound, then we conservatively return that it
    // needs drop.
    let adt_has_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.destructor(tcx).map(|_| DtorType::Significant);
    let res =
        drop_tys_helper(tcx, query.value, query.param_env, adt_has_dtor, false).next().is_some();

    debug!("needs_drop_raw({:?}) = {:?}", query, res);
    res
}

fn has_significant_drop_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> bool {
    let res = drop_tys_helper(
        tcx,
        query.value,
        query.param_env,
        adt_consider_insignificant_dtor(tcx),
        true,
    )
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
    F: Fn(ty::AdtDef<'tcx>, SubstsRef<'tcx>) -> NeedsDropResult<I>,
    I: Iterator<Item = Ty<'tcx>>,
{
    type Item = NeedsDropResult<Ty<'tcx>>;

    fn next(&mut self) -> Option<NeedsDropResult<Ty<'tcx>>> {
        let tcx = self.tcx;

        while let Some((ty, level)) = self.unchecked_tys.pop() {
            if !self.recursion_limit.value_within_limit(level) {
                // Not having a `Span` isn't great. But there's hopefully some other
                // recursion limit error as well.
                tcx.sess.emit_err(NeedsDropOverflow { query_ty: self.query_ty });
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
                    // The information required to determine whether a generator has drop is
                    // computed on MIR, while this very method is used to build MIR.
                    // To avoid cycles, we consider that generators always require drop.
                    ty::Generator(..) if tcx.sess.opts.unstable_opts.drop_tracking_mir => {
                        return Some(Err(AlwaysRequiresDrop));
                    }

                    _ if component.is_copy_modulo_regions(tcx, self.param_env) => (),

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
                        let tys = match (self.adt_components)(adt_def, substs) {
                            Err(e) => return Some(Err(e)),
                            Ok(tys) => tys,
                        };
                        for required_ty in tys {
                            let required = tcx
                                .try_normalize_erasing_regions(self.param_env, required_ty)
                                .unwrap_or(required_ty);

                            queue_type(self, required);
                        }
                    }
                    ty::Array(..) | ty::Alias(..) | ty::Param(_) => {
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

enum DtorType {
    /// Type has a `Drop` but it is considered insignificant.
    /// Check the query `adt_significant_drop_tys` for understanding
    /// "significant" / "insignificant".
    Insignificant,

    /// Type has a `Drop` implantation.
    Significant,
}

// This is a helper function for `adt_drop_tys` and `adt_significant_drop_tys`.
// Depending on the implantation of `adt_has_dtor`, it is used to check if the
// ADT has a destructor or if the ADT only has a significant destructor. For
// understanding significant destructor look at `adt_significant_drop_tys`.
fn drop_tys_helper<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    param_env: rustc_middle::ty::ParamEnv<'tcx>,
    adt_has_dtor: impl Fn(ty::AdtDef<'tcx>) -> Option<DtorType>,
    only_significant: bool,
) -> impl Iterator<Item = NeedsDropResult<Ty<'tcx>>> {
    fn with_query_cache<'tcx>(
        tcx: TyCtxt<'tcx>,
        iter: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> NeedsDropResult<Vec<Ty<'tcx>>> {
        iter.into_iter().try_fold(Vec::new(), |mut vec, subty| {
            match subty.kind() {
                ty::Adt(adt_id, subst) => {
                    for subty in tcx.adt_drop_tys(adt_id.did())? {
                        vec.push(EarlyBinder(subty).subst(tcx, subst));
                    }
                }
                _ => vec.push(subty),
            };
            Ok(vec)
        })
    }

    let adt_components = move |adt_def: ty::AdtDef<'tcx>, substs: SubstsRef<'tcx>| {
        if adt_def.is_manually_drop() {
            debug!("drop_tys_helper: `{:?}` is manually drop", adt_def);
            Ok(Vec::new())
        } else if let Some(dtor_info) = adt_has_dtor(adt_def) {
            match dtor_info {
                DtorType::Significant => {
                    debug!("drop_tys_helper: `{:?}` implements `Drop`", adt_def);
                    Err(AlwaysRequiresDrop)
                }
                DtorType::Insignificant => {
                    debug!("drop_tys_helper: `{:?}` drop is insignificant", adt_def);

                    // Since the destructor is insignificant, we just want to make sure all of
                    // the passed in type parameters are also insignificant.
                    // Eg: Vec<T> dtor is insignificant when T=i32 but significant when T=Mutex.
                    Ok(substs.types().collect())
                }
            }
        } else if adt_def.is_union() {
            debug!("drop_tys_helper: `{:?}` is a union", adt_def);
            Ok(Vec::new())
        } else {
            let field_tys = adt_def.all_fields().map(|field| {
                let r = tcx.type_of(field.did).subst(tcx, substs);
                debug!("drop_tys_helper: Subst into {:?} with {:?} gettng {:?}", field, substs, r);
                r
            });
            if only_significant {
                // We can't recurse through the query system here because we might induce a cycle
                Ok(field_tys.collect())
            } else {
                // We can use the query system if we consider all drops significant. In that case,
                // ADTs are `needs_drop` exactly if they `impl Drop` or if any of their "transitive"
                // fields do. There can be no cycles here, because ADTs cannot contain themselves as
                // fields.
                with_query_cache(tcx, field_tys)
            }
        }
        .map(|v| v.into_iter())
    };

    NeedsDropTypes::new(tcx, param_env, ty, adt_components)
}

fn adt_consider_insignificant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> impl Fn(ty::AdtDef<'tcx>) -> Option<DtorType> + 'tcx {
    move |adt_def: ty::AdtDef<'tcx>| {
        let is_marked_insig = tcx.has_attr(adt_def.did(), sym::rustc_insignificant_dtor);
        if is_marked_insig {
            // In some cases like `std::collections::HashMap` where the struct is a wrapper around
            // a type that is a Drop type, and the wrapped type (eg: `hashbrown::HashMap`) lies
            // outside stdlib, we might choose to still annotate the wrapper (std HashMap) with
            // `rustc_insignificant_dtor`, even if the type itself doesn't have a `Drop` impl.
            Some(DtorType::Insignificant)
        } else if adt_def.destructor(tcx).is_some() {
            // There is a Drop impl and the type isn't marked insignificant, therefore Drop must be
            // significant.
            Some(DtorType::Significant)
        } else {
            // No destructor found nor the type is annotated with `rustc_insignificant_dtor`, we
            // treat this as the simple case of Drop impl for type.
            None
        }
    }
}

fn adt_drop_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Result<&ty::List<Ty<'tcx>>, AlwaysRequiresDrop> {
    // This is for the "adt_drop_tys" query, that considers all `Drop` impls, therefore all dtors are
    // significant.
    let adt_has_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.destructor(tcx).map(|_| DtorType::Significant);
    // `tcx.type_of(def_id)` identical to `tcx.make_adt(def, identity_substs)`
    drop_tys_helper(
        tcx,
        tcx.type_of(def_id).subst_identity(),
        tcx.param_env(def_id),
        adt_has_dtor,
        false,
    )
    .collect::<Result<Vec<_>, _>>()
    .map(|components| tcx.intern_type_list(&components))
}
// If `def_id` refers to a generic ADT, the queries above and below act as if they had been handed
// a `tcx.make_ty(def, identity_substs)` and as such it is legal to substitute the generic parameters
// of the ADT into the outputted `ty`s.
fn adt_significant_drop_tys(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Result<&ty::List<Ty<'_>>, AlwaysRequiresDrop> {
    drop_tys_helper(
        tcx,
        tcx.type_of(def_id).subst_identity(), // identical to `tcx.make_adt(def, identity_substs)`
        tcx.param_env(def_id),
        adt_consider_insignificant_dtor(tcx),
        true,
    )
    .collect::<Result<Vec<_>, _>>()
    .map(|components| tcx.intern_type_list(&components))
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
