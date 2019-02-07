use crate::hir::def_id::DefId;
use crate::hir;
use crate::hir::Node;
use crate::infer::{self, InferCtxt, InferOk, TypeVariableOrigin};
use crate::infer::outlives::free_region_map::FreeRegionRelations;
use rustc_data_structures::fx::FxHashMap;
use syntax::ast;
use crate::traits::{self, PredicateObligation};
use crate::ty::{self, Ty, TyCtxt, GenericParamDefKind};
use crate::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder};
use crate::ty::outlives::Component;
use crate::ty::subst::{Kind, InternalSubsts, SubstsRef, UnpackedKind};
use crate::util::nodemap::DefIdMap;

pub type OpaqueTypeMap<'tcx> = DefIdMap<OpaqueTypeDecl<'tcx>>;

/// Information about the opaque, abstract types whose values we
/// are inferring in this function (these are the `impl Trait` that
/// appear in the return type).
#[derive(Copy, Clone, Debug)]
pub struct OpaqueTypeDecl<'tcx> {
    /// The substitutions that we apply to the abstract that this
    /// `impl Trait` desugars to. e.g., if:
    ///
    ///     fn foo<'a, 'b, T>() -> impl Trait<'a>
    ///
    /// winds up desugared to:
    ///
    ///     abstract type Foo<'x, X>: Trait<'x>
    ///     fn foo<'a, 'b, T>() -> Foo<'a, T>
    ///
    /// then `substs` would be `['a, T]`.
    pub substs: SubstsRef<'tcx>,

    /// The type variable that represents the value of the abstract type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    ///
    ///     Foo<'a, T> = ?C
    ///
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub concrete_ty: Ty<'tcx>,

    /// Returns `true` if the `impl Trait` bounds include region bounds.
    /// For example, this would be true for:
    ///
    ///     fn foo<'a, 'b, 'c>() -> impl Trait<'c> + 'a + 'b
    ///
    /// but false for:
    ///
    ///     fn foo<'c>() -> impl Trait<'c>
    ///
    /// unless `Trait` was declared like:
    ///
    ///     trait Trait<'c>: 'c
    ///
    /// in which case it would be true.
    ///
    /// This is used during regionck to decide whether we need to
    /// impose any additional constraints to ensure that region
    /// variables in `concrete_ty` wind up being constrained to
    /// something from `substs` (or, at minimum, things that outlive
    /// the fn body). (Ultimately, writeback is responsible for this
    /// check.)
    pub has_required_region_bounds: bool,
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    /// Replaces all opaque types in `value` with fresh inference variables
    /// and creates appropriate obligations. For example, given the input:
    ///
    ///     impl Iterator<Item = impl Debug>
    ///
    /// this method would create two type variables, `?0` and `?1`. It would
    /// return the type `?0` but also the obligations:
    ///
    ///     ?0: Iterator<Item = ?1>
    ///     ?1: Debug
    ///
    /// Moreover, it returns a `OpaqueTypeMap` that would map `?0` to
    /// info about the `impl Iterator<..>` type and `?1` to info about
    /// the `impl Debug` type.
    ///
    /// # Parameters
    ///
    /// - `parent_def_id` -- the `DefId` of the function in which the opaque type
    ///   is defined
    /// - `body_id` -- the body-id with which the resulting obligations should
    ///   be associated
    /// - `param_env` -- the in-scope parameter environment to be used for
    ///   obligations
    /// - `value` -- the value within which we are instantiating opaque types
    pub fn instantiate_opaque_types<T: TypeFoldable<'tcx>>(
        &self,
        parent_def_id: DefId,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: &T,
    ) -> InferOk<'tcx, (T, OpaqueTypeMap<'tcx>)> {
        debug!("instantiate_opaque_types(value={:?}, parent_def_id={:?}, body_id={:?}, \
                param_env={:?})",
               value, parent_def_id, body_id, param_env,
        );
        let mut instantiator = Instantiator {
            infcx: self,
            parent_def_id,
            body_id,
            param_env,
            opaque_types: Default::default(),
            obligations: vec![],
        };
        let value = instantiator.instantiate_opaque_types_in_map(value);
        InferOk {
            value: (value, instantiator.opaque_types),
            obligations: instantiator.obligations,
        }
    }

    /// Given the map `opaque_types` containing the existential `impl
    /// Trait` types whose underlying, hidden types are being
    /// inferred, this method adds constraints to the regions
    /// appearing in those underlying hidden types to ensure that they
    /// at least do not refer to random scopes within the current
    /// function. These constraints are not (quite) sufficient to
    /// guarantee that the regions are actually legal values; that
    /// final condition is imposed after region inference is done.
    ///
    /// # The Problem
    ///
    /// Let's work through an example to explain how it works. Assume
    /// the current function is as follows:
    ///
    /// ```text
    /// fn foo<'a, 'b>(..) -> (impl Bar<'a>, impl Bar<'b>)
    /// ```
    ///
    /// Here, we have two `impl Trait` types whose values are being
    /// inferred (the `impl Bar<'a>` and the `impl
    /// Bar<'b>`). Conceptually, this is sugar for a setup where we
    /// define underlying abstract types (`Foo1`, `Foo2`) and then, in
    /// the return type of `foo`, we *reference* those definitions:
    ///
    /// ```text
    /// abstract type Foo1<'x>: Bar<'x>;
    /// abstract type Foo2<'x>: Bar<'x>;
    /// fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    ///                    //  ^^^^ ^^
    ///                    //  |    |
    ///                    //  |    substs
    ///                    //  def_id
    /// ```
    ///
    /// As indicating in the comments above, each of those references
    /// is (in the compiler) basically a substitution (`substs`)
    /// applied to the type of a suitable `def_id` (which identifies
    /// `Foo1` or `Foo2`).
    ///
    /// Now, at this point in compilation, what we have done is to
    /// replace each of the references (`Foo1<'a>`, `Foo2<'b>`) with
    /// fresh inference variables C1 and C2. We wish to use the values
    /// of these variables to infer the underlying types of `Foo1` and
    /// `Foo2`. That is, this gives rise to higher-order (pattern) unification
    /// constraints like:
    ///
    /// ```text
    /// for<'a> (Foo1<'a> = C1)
    /// for<'b> (Foo1<'b> = C2)
    /// ```
    ///
    /// For these equation to be satisfiable, the types `C1` and `C2`
    /// can only refer to a limited set of regions. For example, `C1`
    /// can only refer to `'static` and `'a`, and `C2` can only refer
    /// to `'static` and `'b`. The job of this function is to impose that
    /// constraint.
    ///
    /// Up to this point, C1 and C2 are basically just random type
    /// inference variables, and hence they may contain arbitrary
    /// regions. In fact, it is fairly likely that they do! Consider
    /// this possible definition of `foo`:
    ///
    /// ```text
    /// fn foo<'a, 'b>(x: &'a i32, y: &'b i32) -> (impl Bar<'a>, impl Bar<'b>) {
    ///         (&*x, &*y)
    ///     }
    /// ```
    ///
    /// Here, the values for the concrete types of the two impl
    /// traits will include inference variables:
    ///
    /// ```text
    /// &'0 i32
    /// &'1 i32
    /// ```
    ///
    /// Ordinarily, the subtyping rules would ensure that these are
    /// sufficiently large. But since `impl Bar<'a>` isn't a specific
    /// type per se, we don't get such constraints by default. This
    /// is where this function comes into play. It adds extra
    /// constraints to ensure that all the regions which appear in the
    /// inferred type are regions that could validly appear.
    ///
    /// This is actually a bit of a tricky constraint in general. We
    /// want to say that each variable (e.g., `'0`) can only take on
    /// values that were supplied as arguments to the abstract type
    /// (e.g., `'a` for `Foo1<'a>`) or `'static`, which is always in
    /// scope. We don't have a constraint quite of this kind in the current
    /// region checker.
    ///
    /// # The Solution
    ///
    /// We make use of the constraint that we *do* have in the `<=`
    /// relation. To do that, we find the "minimum" of all the
    /// arguments that appear in the substs: that is, some region
    /// which is less than all the others. In the case of `Foo1<'a>`,
    /// that would be `'a` (it's the only choice, after all). Then we
    /// apply that as a least bound to the variables (e.g., `'a <=
    /// '0`).
    ///
    /// In some cases, there is no minimum. Consider this example:
    ///
    /// ```text
    /// fn baz<'a, 'b>() -> impl Trait<'a, 'b> { ... }
    /// ```
    ///
    /// Here we would report an error, because `'a` and `'b` have no
    /// relation to one another.
    ///
    /// # The `free_region_relations` parameter
    ///
    /// The `free_region_relations` argument is used to find the
    /// "minimum" of the regions supplied to a given abstract type.
    /// It must be a relation that can answer whether `'a <= 'b`,
    /// where `'a` and `'b` are regions that appear in the "substs"
    /// for the abstract type references (the `<'a>` in `Foo1<'a>`).
    ///
    /// Note that we do not impose the constraints based on the
    /// generic regions from the `Foo1` definition (e.g., `'x`). This
    /// is because the constraints we are imposing here is basically
    /// the concern of the one generating the constraining type C1,
    /// which is the current function. It also means that we can
    /// take "implied bounds" into account in some cases:
    ///
    /// ```text
    /// trait SomeTrait<'a, 'b> { }
    /// fn foo<'a, 'b>(_: &'a &'b u32) -> impl SomeTrait<'a, 'b> { .. }
    /// ```
    ///
    /// Here, the fact that `'b: 'a` is known only because of the
    /// implied bounds from the `&'a &'b u32` parameter, and is not
    /// "inherent" to the abstract type definition.
    ///
    /// # Parameters
    ///
    /// - `opaque_types` -- the map produced by `instantiate_opaque_types`
    /// - `free_region_relations` -- something that can be used to relate
    ///   the free regions (`'a`) that appear in the impl trait.
    pub fn constrain_opaque_types<FRR: FreeRegionRelations<'tcx>>(
        &self,
        opaque_types: &OpaqueTypeMap<'tcx>,
        free_region_relations: &FRR,
    ) {
        debug!("constrain_opaque_types()");

        for (&def_id, opaque_defn) in opaque_types {
            self.constrain_opaque_type(def_id, opaque_defn, free_region_relations);
        }
    }

    pub fn constrain_opaque_type<FRR: FreeRegionRelations<'tcx>>(
        &self,
        def_id: DefId,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
        free_region_relations: &FRR,
    ) {
        debug!("constrain_opaque_type()");
        debug!("constrain_opaque_type: def_id={:?}", def_id);
        debug!("constrain_opaque_type: opaque_defn={:#?}", opaque_defn);

        let concrete_ty = self.resolve_type_vars_if_possible(&opaque_defn.concrete_ty);

        debug!("constrain_opaque_type: concrete_ty={:?}", concrete_ty);

        let abstract_type_generics = self.tcx.generics_of(def_id);

        let span = self.tcx.def_span(def_id);

        // If there are required region bounds, we can just skip
        // ahead.  There will already be a registered region
        // obligation related `concrete_ty` to those regions.
        if opaque_defn.has_required_region_bounds {
            return;
        }

        // There were no `required_region_bounds`,
        // so we have to search for a `least_region`.
        // Go through all the regions used as arguments to the
        // abstract type. These are the parameters to the abstract
        // type; so in our example above, `substs` would contain
        // `['a]` for the first impl trait and `'b` for the
        // second.
        let mut least_region = None;
        for param in &abstract_type_generics.params {
            match param.kind {
                GenericParamDefKind::Lifetime => {}
                _ => continue
            }
            // Get the value supplied for this region from the substs.
            let subst_arg = opaque_defn.substs.region_at(param.index as usize);

            // Compute the least upper bound of it with the other regions.
            debug!("constrain_opaque_types: least_region={:?}", least_region);
            debug!("constrain_opaque_types: subst_arg={:?}", subst_arg);
            match least_region {
                None => least_region = Some(subst_arg),
                Some(lr) => {
                    if free_region_relations.sub_free_regions(lr, subst_arg) {
                        // keep the current least region
                    } else if free_region_relations.sub_free_regions(subst_arg, lr) {
                        // switch to `subst_arg`
                        least_region = Some(subst_arg);
                    } else {
                        // There are two regions (`lr` and
                        // `subst_arg`) which are not relatable. We can't
                        // find a best choice.
                        self.tcx
                            .sess
                            .struct_span_err(span, "ambiguous lifetime bound in `impl Trait`")
                            .span_label(
                                span,
                                format!("neither `{}` nor `{}` outlives the other", lr, subst_arg),
                            )
                            .emit();

                        least_region = Some(self.tcx.mk_region(ty::ReEmpty));
                        break;
                    }
                }
            }
        }

        let least_region = least_region.unwrap_or(self.tcx.types.re_static);
        debug!("constrain_opaque_types: least_region={:?}", least_region);

        // Require that the type `concrete_ty` outlives
        // `least_region`, modulo any type parameters that appear
        // in the type, which we ignore. This is because impl
        // trait values are assumed to capture all the in-scope
        // type parameters. This little loop here just invokes
        // `outlives` repeatedly, draining all the nested
        // obligations that result.
        let mut types = vec![concrete_ty];
        let bound_region = |r| self.sub_regions(infer::CallReturn(span), least_region, r);
        while let Some(ty) = types.pop() {
            let mut components = smallvec![];
            self.tcx.push_outlives_components(ty, &mut components);
            while let Some(component) = components.pop() {
                match component {
                    Component::Region(r) => {
                        bound_region(r);
                    }

                    Component::Param(_) => {
                        // ignore type parameters like `T`, they are captured
                        // implicitly by the `impl Trait`
                    }

                    Component::UnresolvedInferenceVariable(_) => {
                        // we should get an error that more type
                        // annotations are needed in this case
                        self.tcx
                            .sess
                            .delay_span_bug(span, "unresolved inf var in opaque");
                    }

                    Component::Projection(ty::ProjectionTy {
                        substs,
                        item_def_id: _,
                    }) => {
                        for r in substs.regions() {
                            bound_region(r);
                        }
                        types.extend(substs.types());
                    }

                    Component::EscapingProjection(more_components) => {
                        components.extend(more_components);
                    }
                }
            }
        }
    }

    /// Given the fully resolved, instantiated type for an opaque
    /// type, i.e., the value of an inference variable like C1 or C2
    /// (*), computes the "definition type" for an abstract type
    /// definition -- that is, the inferred value of `Foo1<'x>` or
    /// `Foo2<'x>` that we would conceptually use in its definition:
    ///
    ///     abstract type Foo1<'x>: Bar<'x> = AAA; <-- this type AAA
    ///     abstract type Foo2<'x>: Bar<'x> = BBB; <-- or this type BBB
    ///     fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    ///
    /// Note that these values are defined in terms of a distinct set of
    /// generic parameters (`'x` instead of `'a`) from C1 or C2. The main
    /// purpose of this function is to do that translation.
    ///
    /// (*) C1 and C2 were introduced in the comments on
    /// `constrain_opaque_types`. Read that comment for more context.
    ///
    /// # Parameters
    ///
    /// - `def_id`, the `impl Trait` type
    /// - `opaque_defn`, the opaque definition created in `instantiate_opaque_types`
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    pub fn infer_opaque_definition_from_instantiation(
        &self,
        def_id: DefId,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
        instantiated_ty: Ty<'gcx>,
    ) -> Ty<'gcx> {
        debug!(
            "infer_opaque_definition_from_instantiation(def_id={:?}, instantiated_ty={:?})",
            def_id, instantiated_ty
        );

        let gcx = self.tcx.global_tcx();

        // Use substs to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_substs = InternalSubsts::identity_for_item(gcx, def_id);
        let map: FxHashMap<Kind<'tcx>, Kind<'gcx>> = opaque_defn
            .substs
            .iter()
            .enumerate()
            .map(|(index, subst)| (*subst, id_substs[index]))
            .collect();

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        let Ok(definition_ty) =
            instantiated_ty.fold_with(&mut ReverseMapper::new(
                self.tcx,
                self.is_tainted_by_errors(),
                def_id,
                map,
                instantiated_ty,
            ));
        debug!(
            "infer_opaque_definition_from_instantiation: definition_ty={:?}",
            definition_ty
        );

        // We can unwrap here because our reverse mapper always
        // produces things with 'gcx lifetime, though the type folder
        // obscures that.
        let definition_ty = gcx.lift(&definition_ty).unwrap();

        definition_ty
    }
}

struct ReverseMapper<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,

    /// If errors have already been reported in this fn, we suppress
    /// our own errors because they are sometimes derivative.
    tainted_by_errors: bool,

    opaque_type_def_id: DefId,
    map: FxHashMap<Kind<'tcx>, Kind<'gcx>>,
    map_missing_regions_to_empty: bool,

    /// initially `Some`, set to `None` once error has been reported
    hidden_ty: Option<Ty<'tcx>>,
}

impl<'cx, 'gcx, 'tcx> ReverseMapper<'cx, 'gcx, 'tcx> {
    fn new(
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        tainted_by_errors: bool,
        opaque_type_def_id: DefId,
        map: FxHashMap<Kind<'tcx>, Kind<'gcx>>,
        hidden_ty: Ty<'tcx>,
    ) -> Self {
        Self {
            tcx,
            tainted_by_errors,
            opaque_type_def_id,
            map,
            map_missing_regions_to_empty: false,
            hidden_ty: Some(hidden_ty),
        }
    }

    fn fold_kind_mapping_missing_regions_to_empty(&mut self, kind: Kind<'tcx>) -> Kind<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        self.map_missing_regions_to_empty = true;
        let Ok(kind) = kind.fold_with(self);
        self.map_missing_regions_to_empty = false;
        kind
    }

    fn fold_kind_normally(&mut self, kind: Kind<'tcx>) -> Kind<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        kind.fold_with(self).unwrap_or_else(|e: !| e)
    }
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for ReverseMapper<'cx, 'gcx, 'tcx> {
    type Error = !;

    fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        match r {
            // ignore bound regions that appear in the type (e.g., this
            // would ignore `'r` in a type like `for<'r> fn(&'r u32)`.
            ty::ReLateBound(..) |

            // ignore `'static`, as that can appear anywhere
            ty::ReStatic |

            // ignore `ReScope`, as that can appear anywhere
            // See `src/test/run-pass/issue-49556.rs` for example.
            ty::ReScope(..) => return Ok(r),

            _ => { }
        }

        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(UnpackedKind::Lifetime(r1)) => Ok(r1),
            Some(u) => panic!("region mapped to unexpected kind: {:?}", u),
            None => {
                if !self.map_missing_regions_to_empty && !self.tainted_by_errors {
                    if let Some(hidden_ty) = self.hidden_ty.take() {
                        let span = self.tcx.def_span(self.opaque_type_def_id);
                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0700,
                            "hidden type for `impl Trait` captures lifetime that \
                             does not appear in bounds",
                        );

                        // Assuming regionck succeeded, then we must
                        // be capturing *some* region from the fn
                        // header, and hence it must be free, so it's
                        // ok to invoke this fn (which doesn't accept
                        // all regions, and would ICE if an
                        // inappropriate region is given). We check
                        // `is_tainted_by_errors` by errors above, so
                        // we don't get in here unless regionck
                        // succeeded. (Note also that if regionck
                        // failed, then the regions we are attempting
                        // to map here may well be giving errors
                        // *because* the constraints were not
                        // satisfiable.)
                        self.tcx.note_and_explain_free_region(
                            &mut err,
                            &format!("hidden type `{}` captures ", hidden_ty),
                            r,
                            ""
                        );

                        err.emit();
                    }
                }
                Ok(self.tcx.types.re_empty)
            },
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        match ty.sty {
            ty::Closure(def_id, substs) => {
                // I am a horrible monster and I pray for death. When
                // we encounter a closure here, it is always a closure
                // from within the function that we are currently
                // type-checking -- one that is now being encapsulated
                // in an existential abstract type. Ideally, we would
                // go through the types/lifetimes that it references
                // and treat them just like we would any other type,
                // which means we would error out if we find any
                // reference to a type/region that is not in the
                // "reverse map".
                //
                // **However,** in the case of closures, there is a
                // somewhat subtle (read: hacky) consideration. The
                // problem is that our closure types currently include
                // all the lifetime parameters declared on the
                // enclosing function, even if they are unused by the
                // closure itself. We can't readily filter them out,
                // so here we replace those values with `'empty`. This
                // can't really make a difference to the rest of the
                // compiler; those regions are ignored for the
                // outlives relation, and hence don't affect trait
                // selection or auto traits, and they are erased
                // during codegen.

                let generics = self.tcx.generics_of(def_id);
                let substs = self.tcx.mk_substs(substs.substs.iter().enumerate().map(
                    |(index, &kind)| {
                        if index < generics.parent_count {
                            // Accommodate missing regions in the parent kinds...
                            self.fold_kind_mapping_missing_regions_to_empty(kind)
                        } else {
                            // ...but not elsewhere.
                            self.fold_kind_normally(kind)
                        }
                    },
                ));

                Ok(self.tcx.mk_closure(def_id, ty::ClosureSubsts { substs }))
            }

            _ => ty.super_fold_with(self),
        }
    }
}

struct Instantiator<'a, 'gcx: 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    parent_def_id: DefId,
    body_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,
    opaque_types: OpaqueTypeMap<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Instantiator<'a, 'gcx, 'tcx> {
    fn instantiate_opaque_types_in_map<T: TypeFoldable<'tcx>>(&mut self, value: &T) -> T {
        debug!("instantiate_opaque_types_in_map(value={:?})", value);
        let tcx = self.infcx.tcx;
        value.fold_with(&mut BottomUpFolder {
            tcx,
            reg_op: |reg| reg,
            fldop: |ty| {
                if let ty::Opaque(def_id, substs) = ty.sty {
                    // Check that this is `impl Trait` type is
                    // declared by `parent_def_id` -- i.e., one whose
                    // value we are inferring.  At present, this is
                    // always true during the first phase of
                    // type-check, but not always true later on during
                    // NLL. Once we support named abstract types more fully,
                    // this same scenario will be able to arise during all phases.
                    //
                    // Here is an example using `abstract type` that indicates
                    // the distinction we are checking for:
                    //
                    // ```rust
                    // mod a {
                    //   pub abstract type Foo: Iterator;
                    //   pub fn make_foo() -> Foo { .. }
                    // }
                    //
                    // mod b {
                    //   fn foo() -> a::Foo { a::make_foo() }
                    // }
                    // ```
                    //
                    // Here, the return type of `foo` references a
                    // `Opaque` indeed, but not one whose value is
                    // presently being inferred. You can get into a
                    // similar situation with closure return types
                    // today:
                    //
                    // ```rust
                    // fn foo() -> impl Iterator { .. }
                    // fn bar() {
                    //     let x = || foo(); // returns the Opaque assoc with `foo`
                    // }
                    // ```
                    if let Some(opaque_node_id) = tcx.hir().as_local_node_id(def_id) {
                        let parent_def_id = self.parent_def_id;
                        let def_scope_default = || {
                            let opaque_parent_node_id = tcx.hir().get_parent(opaque_node_id);
                            parent_def_id == tcx.hir().local_def_id(opaque_parent_node_id)
                        };
                        let in_definition_scope = match tcx.hir().find(opaque_node_id) {
                            Some(Node::Item(item)) => match item.node {
                                // impl trait
                                hir::ItemKind::Existential(hir::ExistTy {
                                    impl_trait_fn: Some(parent),
                                    ..
                                }) => parent == self.parent_def_id,
                                // named existential types
                                hir::ItemKind::Existential(hir::ExistTy {
                                    impl_trait_fn: None,
                                    ..
                                }) => may_define_existential_type(
                                    tcx,
                                    self.parent_def_id,
                                    opaque_node_id,
                                ),
                                _ => def_scope_default(),
                            },
                            Some(Node::ImplItem(item)) => match item.node {
                                hir::ImplItemKind::Existential(_) => may_define_existential_type(
                                    tcx,
                                    self.parent_def_id,
                                    opaque_node_id,
                                ),
                                _ => def_scope_default(),
                            },
                            _ => bug!(
                                "expected (impl) item, found {}",
                                tcx.hir().node_to_string(opaque_node_id),
                            ),
                        };
                        if in_definition_scope {
                            return self.fold_opaque_ty(ty, def_id, substs);
                        }

                        debug!(
                            "instantiate_opaque_types_in_map: \
                             encountered opaque outside its definition scope \
                             def_id={:?}",
                            def_id,
                        );
                    }
                }

                ty
            },
        }).unwrap_or_else(|e: !| e)
    }

    fn fold_opaque_ty(
        &mut self,
        ty: Ty<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Ty<'tcx> {
        let infcx = self.infcx;
        let tcx = infcx.tcx;

        debug!(
            "instantiate_opaque_types: Opaque(def_id={:?}, substs={:?})",
            def_id, substs
        );

        // Use the same type variable if the exact same Opaque appears more
        // than once in the return type (e.g., if it's passed to a type alias).
        if let Some(opaque_defn) = self.opaque_types.get(&def_id) {
            return opaque_defn.concrete_ty;
        }
        let span = tcx.def_span(def_id);
        let ty_var = infcx.next_ty_var(TypeVariableOrigin::TypeInference(span));

        let predicates_of = tcx.predicates_of(def_id);
        debug!(
            "instantiate_opaque_types: predicates: {:#?}",
            predicates_of,
        );
        let bounds = predicates_of.instantiate(tcx, substs);
        debug!("instantiate_opaque_types: bounds={:?}", bounds);

        let required_region_bounds = tcx.required_region_bounds(ty, bounds.predicates.clone());
        debug!(
            "instantiate_opaque_types: required_region_bounds={:?}",
            required_region_bounds
        );

        // make sure that we are in fact defining the *entire* type
        // e.g., `existential type Foo<T: Bound>: Bar;` needs to be
        // defined by a function like `fn foo<T: Bound>() -> Foo<T>`.
        debug!(
            "instantiate_opaque_types: param_env: {:#?}",
            self.param_env,
        );
        debug!(
            "instantiate_opaque_types: generics: {:#?}",
            tcx.generics_of(def_id),
        );

        self.opaque_types.insert(
            def_id,
            OpaqueTypeDecl {
                substs,
                concrete_ty: ty_var,
                has_required_region_bounds: !required_region_bounds.is_empty(),
            },
        );
        debug!("instantiate_opaque_types: ty_var={:?}", ty_var);

        self.obligations.reserve(bounds.predicates.len());
        for predicate in bounds.predicates {
            // Change the predicate to refer to the type variable,
            // which will be the concrete type instead of the opaque type.
            // This also instantiates nested instances of `impl Trait`.
            let predicate = self.instantiate_opaque_types_in_map(&predicate);

            let cause = traits::ObligationCause::new(span, self.body_id, traits::SizedReturnType);

            // Require that the predicate holds for the concrete type.
            debug!("instantiate_opaque_types: predicate={:?}", predicate);
            self.obligations
                .push(traits::Obligation::new(cause, self.param_env, predicate));
        }

        ty_var
    }
}

/// Returns `true` if `opaque_node_id` is a sibling or a child of a sibling of `def_id`.
///
/// ```rust
/// pub mod foo {
///     pub mod bar {
///         pub existential type Baz;
///
///         fn f1() -> Baz { .. }
///     }
///
///     fn f2() -> bar::Baz { .. }
/// }
/// ```
///
/// Here, `def_id` is the `DefId` of the existential type `Baz` and `opaque_node_id` is the
/// `NodeId` of the reference to `Baz` (i.e., the return type of both `f1` and `f2`).
/// We return `true` if the reference is within the same module as the existential type
/// (i.e., `true` for `f1`, `false` for `f2`).
pub fn may_define_existential_type(
    tcx: TyCtxt<'_, '_, '_>,
    def_id: DefId,
    opaque_node_id: ast::NodeId,
) -> bool {
    let mut node_id = tcx
        .hir()
        .as_local_node_id(def_id)
        .unwrap();
    // named existential types can be defined by any siblings or
    // children of siblings
    let mod_id = tcx.hir().get_parent(opaque_node_id);
    // so we walk up the node tree until we hit the root or the parent
    // of the opaque type
    while node_id != mod_id && node_id != ast::CRATE_NODE_ID {
        node_id = tcx.hir().get_parent(node_id);
    }
    // syntactically we are allowed to define the concrete type
    node_id == mod_id
}
