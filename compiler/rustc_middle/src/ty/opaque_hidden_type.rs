use rustc_errors::ErrorGuaranteed;
use rustc_span::Span;

use crate::error::OpaqueHiddenTypeMismatch;
use crate::ty::{
    opaque_types, InternalSubsts, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable, TypeMismatchReason,
};

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
        self.fold_with(&mut opaque_types::ReverseMapper::new(tcx, map, self.span, ignore_errors))
    }
}
