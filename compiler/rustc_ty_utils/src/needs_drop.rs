//! Check whether a type has (potentially) non-trivial drop glue.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_hir::limit::Limit;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::util::{AlwaysRequiresDrop, needs_drop_components};
use rustc_middle::ty::{self, EarlyBinder, GenericArgsRef, Ty, TyCtxt};
use rustc_span::sym;
use tracing::{debug, instrument};

use crate::errors::NeedsDropOverflow;

type NeedsDropResult<T> = Result<T, AlwaysRequiresDrop>;

fn needs_drop_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> bool {
    // If we don't know a type doesn't need drop, for example if it's a type
    // parameter without a `Copy` bound, then we conservatively return that it
    // needs drop.
    let adt_has_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.destructor(tcx).map(|_| DtorType::Significant);
    let res = drop_tys_helper(tcx, query.value, query.typing_env, adt_has_dtor, false, false)
        .filter(filter_array_elements(tcx, query.typing_env))
        .next()
        .is_some();

    debug!("needs_drop_raw({:?}) = {:?}", query, res);
    res
}

fn needs_async_drop_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> bool {
    // If we don't know a type doesn't need async drop, for example if it's a
    // type parameter without a `Copy` bound, then we conservatively return that
    // it needs async drop.
    let adt_has_async_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.async_destructor(tcx).map(|_| DtorType::Significant);
    let res = drop_tys_helper(tcx, query.value, query.typing_env, adt_has_async_dtor, false, false)
        .filter(filter_array_elements_async(tcx, query.typing_env))
        .next()
        .is_some();

    debug!("needs_async_drop_raw({:?}) = {:?}", query, res);
    res
}

/// HACK: in order to not mistakenly assume that `[PhantomData<T>; N]` requires drop glue
/// we check the element type for drop glue. The correct fix would be looking at the
/// entirety of the code around `needs_drop_components` and this file and come up with
/// logic that is easier to follow while not repeating any checks that may thus diverge.
fn filter_array_elements<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
) -> impl Fn(&Result<Ty<'tcx>, AlwaysRequiresDrop>) -> bool {
    move |ty| match ty {
        Ok(ty) => match *ty.kind() {
            ty::Array(elem, _) => tcx.needs_drop_raw(typing_env.as_query_input(elem)),
            _ => true,
        },
        Err(AlwaysRequiresDrop) => true,
    }
}
fn filter_array_elements_async<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
) -> impl Fn(&Result<Ty<'tcx>, AlwaysRequiresDrop>) -> bool {
    move |ty| match ty {
        Ok(ty) => match *ty.kind() {
            ty::Array(elem, _) => tcx.needs_async_drop_raw(typing_env.as_query_input(elem)),
            _ => true,
        },
        Err(AlwaysRequiresDrop) => true,
    }
}

fn has_significant_drop_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> bool {
    let res = drop_tys_helper(
        tcx,
        query.value,
        query.typing_env,
        adt_consider_insignificant_dtor(tcx),
        true,
        false,
    )
    .filter(filter_array_elements(tcx, query.typing_env))
    .next()
    .is_some();
    debug!("has_significant_drop_raw({:?}) = {:?}", query, res);
    res
}

struct NeedsDropTypes<'tcx, F> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    query_ty: Ty<'tcx>,
    seen_tys: FxHashSet<Ty<'tcx>>,
    /// A stack of types left to process, and the recursion depth when we
    /// pushed that type. Each round, we pop something from the stack and check
    /// if it needs drop. If the result depends on whether some other types
    /// need drop we push them onto the stack.
    unchecked_tys: Vec<(Ty<'tcx>, usize)>,
    recursion_limit: Limit,
    adt_components: F,
    /// Set this to true if an exhaustive list of types involved in
    /// drop obligation is requested.
    // FIXME: Calling this bool `exhaustive` is confusing and possibly a footgun,
    // since it does two things: It makes the iterator yield *all* of the types
    // that need drop, and it also affects the computation of the drop components
    // on `Coroutine`s. The latter is somewhat confusing, and probably should be
    // a function of `typing_env`. See the HACK comment below for why this is
    // necessary. If this isn't possible, then we probably should turn this into
    // a `NeedsDropMode` so that we can have a variant like `CollectAllSignificantDrops`,
    // which will more accurately indicate that we want *all* of the *significant*
    // drops, which are the two important behavioral changes toggled by this bool.
    exhaustive: bool,
}

impl<'tcx, F> NeedsDropTypes<'tcx, F> {
    fn new(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
        exhaustive: bool,
        adt_components: F,
    ) -> Self {
        let mut seen_tys = FxHashSet::default();
        seen_tys.insert(ty);
        Self {
            tcx,
            typing_env,
            seen_tys,
            query_ty: ty,
            unchecked_tys: vec![(ty, 0)],
            recursion_limit: tcx.recursion_limit(),
            adt_components,
            exhaustive,
        }
    }

    /// Called when `ty` is found to always require drop.
    /// If the exhaustive flag is true, then `Ok(ty)` is returned like any other type.
    /// Otherwise, `Err(AlwaysRequireDrop)` is returned, which will cause iteration to abort.
    fn always_drop_component(&self, ty: Ty<'tcx>) -> NeedsDropResult<Ty<'tcx>> {
        if self.exhaustive { Ok(ty) } else { Err(AlwaysRequiresDrop) }
    }
}

impl<'tcx, F, I> Iterator for NeedsDropTypes<'tcx, F>
where
    F: Fn(ty::AdtDef<'tcx>, GenericArgsRef<'tcx>) -> NeedsDropResult<I>,
    I: Iterator<Item = Ty<'tcx>>,
{
    type Item = NeedsDropResult<Ty<'tcx>>;

    #[instrument(level = "debug", skip(self), ret)]
    fn next(&mut self) -> Option<NeedsDropResult<Ty<'tcx>>> {
        let tcx = self.tcx;

        while let Some((ty, level)) = self.unchecked_tys.pop() {
            debug!(?ty, "needs_drop_components: inspect");
            if !self.recursion_limit.value_within_limit(level) {
                // Not having a `Span` isn't great. But there's hopefully some other
                // recursion limit error as well.
                debug!("needs_drop_components: recursion limit exceeded");
                tcx.dcx().emit_err(NeedsDropOverflow { query_ty: self.query_ty });
                return Some(self.always_drop_component(ty));
            }

            let components = match needs_drop_components(tcx, ty) {
                Err(AlwaysRequiresDrop) => return Some(self.always_drop_component(ty)),
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
                    // The information required to determine whether a coroutine has drop is
                    // computed on MIR, while this very method is used to build MIR.
                    // To avoid cycles, we consider that coroutines always require drop.
                    //
                    // HACK: Because we erase regions contained in the coroutine witness, we
                    // have to conservatively assume that every region captured by the
                    // coroutine has to be live when dropped. This results in a lot of
                    // undesirable borrowck errors. During borrowck, we call `needs_drop`
                    // for the coroutine witness and check whether any of the contained types
                    // need to be dropped, and only require the captured types to be live
                    // if they do.
                    ty::Coroutine(def_id, args) => {
                        // FIXME: See FIXME on `exhaustive` field above.
                        if self.exhaustive {
                            for upvar in args.as_coroutine().upvar_tys() {
                                queue_type(self, upvar);
                            }
                            queue_type(self, args.as_coroutine().resume_ty());
                            if let Some(witness) = tcx.mir_coroutine_witnesses(def_id) {
                                for field_ty in &witness.field_tys {
                                    queue_type(
                                        self,
                                        EarlyBinder::bind(field_ty.ty).instantiate(tcx, args),
                                    );
                                }
                            }
                        } else {
                            return Some(self.always_drop_component(ty));
                        }
                    }
                    ty::CoroutineWitness(..) => {
                        unreachable!("witness should be handled in parent");
                    }

                    ty::UnsafeBinder(bound_ty) => {
                        let ty = self.tcx.instantiate_bound_regions_with_erased(bound_ty.into());
                        queue_type(self, ty);
                    }

                    _ if tcx.type_is_copy_modulo_regions(self.typing_env, component) => {}

                    ty::Closure(_, args) => {
                        for upvar in args.as_closure().upvar_tys() {
                            queue_type(self, upvar);
                        }
                    }

                    ty::CoroutineClosure(_, args) => {
                        for upvar in args.as_coroutine_closure().upvar_tys() {
                            queue_type(self, upvar);
                        }
                    }

                    // Check for a `Drop` impl and whether this is a union or
                    // `ManuallyDrop`. If it's a struct or enum without a `Drop`
                    // impl then check whether the field types need `Drop`.
                    ty::Adt(adt_def, args) => {
                        let tys = match (self.adt_components)(adt_def, args) {
                            Err(AlwaysRequiresDrop) => {
                                return Some(self.always_drop_component(ty));
                            }
                            Ok(tys) => tys,
                        };
                        for required_ty in tys {
                            let required = tcx
                                .try_normalize_erasing_regions(self.typing_env, required_ty)
                                .unwrap_or(required_ty);

                            queue_type(self, required);
                        }
                    }
                    ty::Alias(..) | ty::Array(..) | ty::Placeholder(_) | ty::Param(_) => {
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

                    ty::Foreign(_) | ty::Dynamic(..) => {
                        debug!("needs_drop_components: foreign or dynamic");
                        return Some(self.always_drop_component(ty));
                    }

                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
                    | ty::Str
                    | ty::Slice(_)
                    | ty::Ref(..)
                    | ty::RawPtr(..)
                    | ty::FnDef(..)
                    | ty::Pat(..)
                    | ty::FnPtr(..)
                    | ty::Tuple(_)
                    | ty::Bound(..)
                    | ty::Never
                    | ty::Infer(_)
                    | ty::Error(_) => {
                        bug!("unexpected type returned by `needs_drop_components`: {component}")
                    }
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
    typing_env: ty::TypingEnv<'tcx>,
    adt_has_dtor: impl Fn(ty::AdtDef<'tcx>) -> Option<DtorType>,
    only_significant: bool,
    exhaustive: bool,
) -> impl Iterator<Item = NeedsDropResult<Ty<'tcx>>> {
    fn with_query_cache<'tcx>(
        tcx: TyCtxt<'tcx>,
        iter: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> NeedsDropResult<Vec<Ty<'tcx>>> {
        iter.into_iter().try_fold(Vec::new(), |mut vec, subty| {
            match subty.kind() {
                ty::Adt(adt_id, args) => {
                    for subty in tcx.adt_drop_tys(adt_id.did())? {
                        vec.push(EarlyBinder::bind(subty).instantiate(tcx, args));
                    }
                }
                _ => vec.push(subty),
            };
            Ok(vec)
        })
    }

    let adt_components = move |adt_def: ty::AdtDef<'tcx>, args: GenericArgsRef<'tcx>| {
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
                    Ok(args.types().collect())
                }
            }
        } else if adt_def.is_union() {
            debug!("drop_tys_helper: `{:?}` is a union", adt_def);
            Ok(Vec::new())
        } else {
            let field_tys = adt_def.all_fields().map(|field| {
                let r = tcx.type_of(field.did).instantiate(tcx, args);
                debug!(
                    "drop_tys_helper: Instantiate into {:?} with {:?} getting {:?}",
                    field, args, r
                );
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

    NeedsDropTypes::new(tcx, typing_env, ty, exhaustive, adt_components)
}

fn adt_consider_insignificant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> impl Fn(ty::AdtDef<'tcx>) -> Option<DtorType> {
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
) -> Result<&'tcx ty::List<Ty<'tcx>>, AlwaysRequiresDrop> {
    // This is for the "adt_drop_tys" query, that considers all `Drop` impls, therefore all dtors are
    // significant.
    let adt_has_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.destructor(tcx).map(|_| DtorType::Significant);
    // `tcx.type_of(def_id)` identical to `tcx.make_adt(def, identity_args)`
    drop_tys_helper(
        tcx,
        tcx.type_of(def_id).instantiate_identity(),
        ty::TypingEnv::non_body_analysis(tcx, def_id),
        adt_has_dtor,
        false,
        false,
    )
    .collect::<Result<Vec<_>, _>>()
    .map(|components| tcx.mk_type_list(&components))
}

fn adt_async_drop_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Result<&'tcx ty::List<Ty<'tcx>>, AlwaysRequiresDrop> {
    // This is for the "adt_async_drop_tys" query, that considers all `AsyncDrop` impls.
    let adt_has_dtor =
        |adt_def: ty::AdtDef<'tcx>| adt_def.async_destructor(tcx).map(|_| DtorType::Significant);
    // `tcx.type_of(def_id)` identical to `tcx.make_adt(def, identity_args)`
    drop_tys_helper(
        tcx,
        tcx.type_of(def_id).instantiate_identity(),
        ty::TypingEnv::non_body_analysis(tcx, def_id),
        adt_has_dtor,
        false,
        false,
    )
    .collect::<Result<Vec<_>, _>>()
    .map(|components| tcx.mk_type_list(&components))
}

// If `def_id` refers to a generic ADT, the queries above and below act as if they had been handed
// a `tcx.make_ty(def, identity_args)` and as such it is legal to instantiate the generic parameters
// of the ADT into the outputted `ty`s.
fn adt_significant_drop_tys(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Result<&ty::List<Ty<'_>>, AlwaysRequiresDrop> {
    drop_tys_helper(
        tcx,
        tcx.type_of(def_id).instantiate_identity(), // identical to `tcx.make_adt(def, identity_args)`
        ty::TypingEnv::non_body_analysis(tcx, def_id),
        adt_consider_insignificant_dtor(tcx),
        true,
        false,
    )
    .collect::<Result<Vec<_>, _>>()
    .map(|components| tcx.mk_type_list(&components))
}

#[instrument(level = "debug", skip(tcx), ret)]
fn list_significant_drop_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> &'tcx ty::List<Ty<'tcx>> {
    tcx.mk_type_list(
        &drop_tys_helper(
            tcx,
            key.value,
            key.typing_env,
            adt_consider_insignificant_dtor(tcx),
            true,
            true,
        )
        .filter_map(|res| res.ok())
        .collect::<Vec<_>>(),
    )
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        needs_drop_raw,
        needs_async_drop_raw,
        has_significant_drop_raw,
        adt_drop_tys,
        adt_async_drop_tys,
        adt_significant_drop_tys,
        list_significant_drop_tys,
        ..*providers
    };
}
