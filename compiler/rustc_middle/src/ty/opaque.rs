use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorGuaranteed;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;

use crate::error::ConstNotUsedTraitAlias;
use crate::error::OpaqueHiddenTypeMismatch;
use crate::ty::fold::{TypeFolder, TypeSuperFoldable};
use crate::ty::subst::{GenericArg, GenericArgKind};
use crate::ty::{self, InternalSubsts, SubstsRef, Ty, TyCtxt, TypeFoldable, TypeMismatchReason};

#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable, HashStable, TyEncodable, TyDecodable)]
pub struct OpaqueHiddenType<'tcx> {
    /// The span of this particular definition of the opaque type. So
    /// for example:
    ///
    /// ```ignore (incomplete snippet)
    /// type Foo = impl Baz;
    /// fn bar() -> Foo {
    /// //          ^^^ This is the span we are looking for!
    /// }
    /// ```
    ///
    /// In cases where the fn returns `(impl Trait, impl Trait)` or
    /// other such combinations, the result is currently
    /// over-approximated, but better than nothing.
    pub span: Span,

    /// The type variable that represents the value of the opaque type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    /// ```ignore (pseudo-rust)
    /// Foo<'a, T> = ?C
    /// ```
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub ty: Ty<'tcx>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct OpaqueTypeKey<'tcx> {
    pub def_id: LocalDefId,
    pub substs: SubstsRef<'tcx>,
}

/// Converts generic params of a TypeFoldable from one
/// item's generics to another. Usually from a function's generics
/// list to the opaque type's own generics.
struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,
    map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
    /// see call sites to fold_kind_no_missing_regions_error
    /// for an explanation of this field.
    do_not_error: bool,

    /// We do not want to emit any errors in typeck because
    /// the spans in typeck are subpar at the moment.
    /// Borrowck will do the same work again (this time with
    /// lifetime information) and thus report better errors.
    ignore_errors: bool,

    /// Span of function being checked.
    span: Span,
}

impl<'tcx> OpaqueHiddenType<'tcx> {
    pub fn report_mismatch(&self, other: &Self, tcx: TyCtxt<'tcx>) -> ErrorGuaranteed {
        // Found different concrete types for the opaque type.
        let sub_diag = if self.span == other.span {
            TypeMismatchReason::ConflictType { span: self.span }
        } else {
            TypeMismatchReason::PreviousUse { span: self.span }
        };
        tcx.sess.emit_err(OpaqueHiddenTypeMismatch {
            self_ty: self.ty,
            other_ty: other.ty,
            other_span: other.span,
            sub: sub_diag,
        })
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn remap_generic_params_to_declaration_params(
        self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        tcx: TyCtxt<'tcx>,
        // typeck errors have subpar spans for opaque types, so delay error reporting until borrowck.
        ignore_errors: bool,
    ) -> Self {
        let OpaqueTypeKey { def_id, substs } = opaque_type_key;

        // Use substs to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_substs = InternalSubsts::identity_for_item(tcx, def_id);
        debug!(?id_substs);

        // This zip may have several times the same lifetime in `substs` paired with a different
        // lifetime from `id_substs`. Simply `collect`ing the iterator is the correct behaviour:
        // it will pick the last one, which is the one we introduced in the impl-trait desugaring.
        let map = substs.iter().zip(id_substs).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        self.fold_with(&mut ReverseMapper::new(tcx, map, self.span, ignore_errors))
    }
}

impl<'tcx> ReverseMapper<'tcx> {
    pub(super) fn new(
        tcx: TyCtxt<'tcx>,
        map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
        span: Span,
        ignore_errors: bool,
    ) -> Self {
        Self { tcx, map, do_not_error: false, ignore_errors, span }
    }

    fn fold_kind_no_missing_regions_error(&mut self, kind: GenericArg<'tcx>) -> GenericArg<'tcx> {
        assert!(!self.do_not_error);
        self.do_not_error = true;
        let kind = kind.fold_with(self);
        self.do_not_error = false;
        kind
    }

    fn fold_kind_normally(&mut self, kind: GenericArg<'tcx>) -> GenericArg<'tcx> {
        assert!(!self.do_not_error);
        kind.fold_with(self)
    }

    fn fold_closure_substs(
        &mut self,
        def_id: DefId,
        substs: ty::SubstsRef<'tcx>,
    ) -> ty::SubstsRef<'tcx> {
        // I am a horrible monster and I pray for death. When
        // we encounter a closure here, it is always a closure
        // from within the function that we are currently
        // type-checking -- one that is now being encapsulated
        // in an opaque type. Ideally, we would
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
        self.tcx.mk_substs_from_iter(substs.iter().enumerate().map(|(index, kind)| {
            if index < generics.parent_count {
                // Accommodate missing regions in the parent kinds...
                self.fold_kind_no_missing_regions_error(kind)
            } else {
                // ...but not elsewhere.
                self.fold_kind_normally(kind)
            }
        }))
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReverseMapper<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            // Ignore bound regions and `'static` regions that appear in the
            // type, we only need to remap regions that reference lifetimes
            // from the function declaration.
            // This would ignore `'r` in a type like `for<'r> fn(&'r u32)`.
            ty::ReLateBound(..) | ty::ReStatic => return r,

            // If regions have been erased (by writeback), don't try to unerase
            // them.
            ty::ReErased => return r,

            ty::ReError(_) => return r,

            // The regions that we expect from borrow checking.
            ty::ReEarlyBound(_) | ty::ReFree(_) => {}

            ty::RePlaceholder(_) | ty::ReVar(_) => {
                // All of the regions in the type should either have been
                // erased by writeback, or mapped back to named regions by
                // borrow checking.
                bug!("unexpected region kind in opaque type: {:?}", r);
            }
        }

        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(GenericArgKind::Lifetime(r1)) => r1,
            Some(u) => panic!("region mapped to unexpected kind: {:?}", u),
            None if self.do_not_error => self.tcx.lifetimes.re_static,
            None => {
                let e = self
                    .tcx
                    .sess
                    .struct_span_err(self.span, "non-defining opaque type use in defining scope")
                    .span_label(
                        self.span,
                        format!(
                            "lifetime `{}` is part of concrete type but not used in \
                             parameter list of the `impl Trait` type alias",
                            r
                        ),
                    )
                    .emit();

                self.interner().mk_re_error(e)
            }
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Closure(def_id, substs) => {
                let substs = self.fold_closure_substs(def_id, substs);
                self.tcx.mk_closure(def_id, substs)
            }

            ty::Generator(def_id, substs, movability) => {
                let substs = self.fold_closure_substs(def_id, substs);
                self.tcx.mk_generator(def_id, substs, movability)
            }

            ty::GeneratorWitnessMIR(def_id, substs) => {
                let substs = self.fold_closure_substs(def_id, substs);
                self.tcx.mk_generator_witness_mir(def_id, substs)
            }

            ty::Param(param) => {
                // Look it up in the substitution list.
                match self.map.get(&ty.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list; replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Type(t1)) => t1,
                    Some(u) => panic!("type mapped to unexpected kind: {:?}", u),
                    None => {
                        debug!(?param, ?self.map);
                        if !self.ignore_errors {
                            self.tcx
                                .sess
                                .struct_span_err(
                                    self.span,
                                    format!(
                                        "type parameter `{}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias",
                                        ty
                                    ),
                                )
                                .emit();
                        }

                        self.interner().ty_error_misc()
                    }
                }
            }

            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        trace!("checking const {:?}", ct);
        // Find a const parameter
        match ct.kind() {
            ty::ConstKind::Param(..) => {
                // Look it up in the substitution list.
                match self.map.get(&ct.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list, replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Const(c1)) => c1,
                    Some(u) => panic!("const mapped to unexpected kind: {:?}", u),
                    None => {
                        if !self.ignore_errors {
                            self.tcx.sess.emit_err(ConstNotUsedTraitAlias {
                                ct: ct.to_string(),
                                span: self.span,
                            });
                        }

                        self.interner().const_error(ct.ty())
                    }
                }
            }

            _ => ct,
        }
    }
}
