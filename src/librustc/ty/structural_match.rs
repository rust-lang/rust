use crate::hir;
use rustc_data_structures::fx::{FxHashSet};

use syntax::symbol::{sym};

use crate::ty::{self, AdtDef, Ty, TyCtxt};
use crate::ty::fold::{TypeFoldable, TypeVisitor};

pub enum NonStructuralMatchTy<'tcx> {
    Adt(&'tcx AdtDef),
    Param,
}

/// This method traverses the structure of `ty`, trying to find an
/// instance of an ADT (i.e. struct or enum) that was declared without
/// the `#[structural_match]` attribute, or a generic type parameter
/// (which cannot be determined to be `structural_match`).
///
/// The "structure of a type" includes all components that would be
/// considered when doing a pattern match on a constant of that
/// type.
///
///  * This means this method descends into fields of structs/enums,
///    and also descends into the inner type `T` of `&T` and `&mut T`
///
///  * The traversal doesn't dereference unsafe pointers (`*const T`,
///    `*mut T`), and it does not visit the type arguments of an
///    instantiated generic like `PhantomData<T>`.
///
/// The reason we do this search is Rust currently require all ADTs
/// reachable from a constant's type to be annotated with
/// `#[structural_match]`, an attribute which essentially says that
/// the implementation of `PartialEq::eq` behaves *equivalently* to a
/// comparison against the unfolded structure.
///
/// For more background on why Rust has this requirement, and issues
/// that arose when the requirement was not enforced completely, see
/// Rust RFC 1445, rust-lang/rust#61188, and rust-lang/rust#62307.
pub fn search_for_structural_match_violation<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<NonStructuralMatchTy<'tcx>> {
    let mut search = Search { tcx, found: None, seen: FxHashSet::default() };
    ty.visit_with(&mut search);
    return search.found;

    struct Search<'tcx> {
        tcx: TyCtxt<'tcx>,

        // Records the first ADT or type parameter we find without `#[structural_match`.
        found: Option<NonStructuralMatchTy<'tcx>>,

        // Tracks ADTs previously encountered during search, so that
        // we will not recurse on them again.
        seen: FxHashSet<hir::def_id::DefId>,
    }

    impl<'tcx> TypeVisitor<'tcx> for Search<'tcx> {
        fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
            debug!("Search visiting ty: {:?}", ty);

            let (adt_def, substs) = match ty.kind {
                ty::Adt(adt_def, substs) => (adt_def, substs),
                ty::Param(_) => {
                    self.found = Some(NonStructuralMatchTy::Param);
                    return true; // Stop visiting.
                }
                ty::RawPtr(..) => {
                    // `#[structural_match]` ignores substructure of
                    // `*const _`/`*mut _`, so skip super_visit_with
                    //
                    // (But still tell caller to continue search.)
                    return false;
                }
                ty::FnDef(..) | ty::FnPtr(..) => {
                    // types of formals and return in `fn(_) -> _` are also irrelevant
                    //
                    // (But still tell caller to continue search.)
                    return false;
                }
                ty::Array(_, n) if n.try_eval_usize(self.tcx, ty::ParamEnv::reveal_all()) == Some(0)
                => {
                    // rust-lang/rust#62336: ignore type of contents
                    // for empty array.
                    return false;
                }
                _ => {
                    ty.super_visit_with(self);
                    return false;
                }
            };

            if !self.tcx.has_attr(adt_def.did, sym::structural_match) {
                self.found = Some(NonStructuralMatchTy::Adt(&adt_def));
                debug!("Search found adt_def: {:?}", adt_def);
                return true; // Stop visiting.
            }

            if !self.seen.insert(adt_def.did) {
                debug!("Search already seen adt_def: {:?}", adt_def);
                // let caller continue its search
                return false;
            }

            // `#[structural_match]` does not care about the
            // instantiation of the generics in an ADT (it
            // instead looks directly at its fields outside
            // this match), so we skip super_visit_with.
            //
            // (Must not recur on substs for `PhantomData<T>` cf
            // rust-lang/rust#55028 and rust-lang/rust#55837; but also
            // want to skip substs when only uses of generic are
            // behind unsafe pointers `*const T`/`*mut T`.)

            // even though we skip super_visit_with, we must recur on
            // fields of ADT.
            let tcx = self.tcx;
            for field_ty in adt_def.all_fields().map(|field| field.ty(tcx, substs)) {
                if field_ty.visit_with(self) {
                    // found an ADT without `#[structural_match]`; halt visiting!
                    assert!(self.found.is_some());
                    return true;
                }
            }

            // Even though we do not want to recur on substs, we do
            // want our caller to continue its own search.
            false
        }
    }
}
