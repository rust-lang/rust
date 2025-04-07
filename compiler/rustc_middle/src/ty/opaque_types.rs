use rustc_data_structures::fx::FxHashMap;
use rustc_span::Span;
use rustc_span::def_id::DefId;
use tracing::{debug, instrument, trace};

use crate::error::ConstNotUsedTraitAlias;
use crate::ty::{
    self, GenericArg, GenericArgKind, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
};

pub type OpaqueTypeKey<'tcx> = rustc_type_ir::OpaqueTypeKey<TyCtxt<'tcx>>;

/// Converts generic params of a TypeFoldable from one
/// item's generics to another. Usually from a function's generics
/// list to the opaque type's own generics.
pub(super) struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,
    map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
    /// see call sites to fold_kind_no_missing_regions_error
    /// for an explanation of this field.
    do_not_error: bool,

    /// Span of function being checked.
    span: Span,
}

impl<'tcx> ReverseMapper<'tcx> {
    pub(super) fn new(
        tcx: TyCtxt<'tcx>,
        map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
        span: Span,
    ) -> Self {
        Self { tcx, map, do_not_error: false, span }
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

    fn fold_closure_args(
        &mut self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> ty::GenericArgsRef<'tcx> {
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
        self.tcx.mk_args_from_iter(args.iter().enumerate().map(|(index, kind)| {
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
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            // Ignore bound regions and `'static` regions that appear in the
            // type, we only need to remap regions that reference lifetimes
            // from the function declaration.
            //
            // E.g. We ignore `'r` in a type like `for<'r> fn(&'r u32)`.
            ty::ReBound(..) | ty::ReStatic => return r,

            // If regions have been erased (by writeback), don't try to unerase
            // them.
            ty::ReErased => return r,

            ty::ReError(_) => return r,

            // The regions that we expect from borrow checking.
            ty::ReEarlyParam(_) | ty::ReLateParam(_) => {}

            ty::RePlaceholder(_) | ty::ReVar(_) => {
                // All of the regions in the type should either have been
                // erased by writeback, or mapped back to named regions by
                // borrow checking.
                bug!("unexpected region kind in opaque type: {:?}", r);
            }
        }

        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(GenericArgKind::Lifetime(r1)) => r1,
            Some(u) => panic!("region mapped to unexpected kind: {u:?}"),
            None if self.do_not_error => self.tcx.lifetimes.re_static,
            None => {
                let e = self
                    .tcx
                    .dcx()
                    .struct_span_err(self.span, "non-defining opaque type use in defining scope")
                    .with_span_label(
                        self.span,
                        format!(
                            "lifetime `{r}` is part of concrete type but not used in \
                             parameter list of the `impl Trait` type alias"
                        ),
                    )
                    .emit();

                ty::Region::new_error(self.cx(), e)
            }
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Closure(def_id, args) => {
                let args = self.fold_closure_args(def_id, args);
                Ty::new_closure(self.tcx, def_id, args)
            }

            ty::Coroutine(def_id, args) => {
                let args = self.fold_closure_args(def_id, args);
                Ty::new_coroutine(self.tcx, def_id, args)
            }

            ty::CoroutineWitness(def_id, args) => {
                let args = self.fold_closure_args(def_id, args);
                Ty::new_coroutine_witness(self.tcx, def_id, args)
            }

            ty::Param(param) => {
                // Look it up in the generic parameters list.
                match self.map.get(&ty.into()).map(|k| k.unpack()) {
                    // Found it in the generic parameters list; replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Type(t1)) => t1,
                    Some(u) => panic!("type mapped to unexpected kind: {u:?}"),
                    None => {
                        debug!(?param, ?self.map);
                        let guar = self
                            .tcx
                            .dcx()
                            .struct_span_err(
                                self.span,
                                format!(
                                    "type parameter `{ty}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias"
                                ),
                            )
                            .emit();
                        Ty::new_error(self.tcx, guar)
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
                // Look it up in the generic parameters list.
                match self.map.get(&ct.into()).map(|k| k.unpack()) {
                    // Found it in the generic parameters list, replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Const(c1)) => c1,
                    Some(u) => panic!("const mapped to unexpected kind: {u:?}"),
                    None => {
                        let guar = self
                            .tcx
                            .dcx()
                            .create_err(ConstNotUsedTraitAlias {
                                ct: ct.to_string(),
                                span: self.span,
                            })
                            .emit();
                        ty::Const::new_error(self.tcx, guar)
                    }
                }
            }

            _ => ct,
        }
    }
}
