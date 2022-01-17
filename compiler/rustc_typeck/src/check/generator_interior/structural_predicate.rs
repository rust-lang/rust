use super::FnCtxt;

use rustc_data_structures::stable_set::FxHashSet;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

pub(super) struct StructuralPredicateElaborator<'a, 'tcx> {
    stack: Vec<Ty<'tcx>>,
    seen: FxHashSet<Ty<'tcx>>,
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
}

impl<'a, 'tcx> StructuralPredicateElaborator<'a, 'tcx> {
    pub fn new(fcx: &'a FnCtxt<'a, 'tcx>, tys: impl Iterator<Item = Ty<'tcx>>, span: Span) -> Self {
        let mut seen = FxHashSet::default();
        let stack =
            tys.map(|ty| fcx.resolve_vars_if_possible(ty)).filter(|ty| seen.insert(*ty)).collect();
        StructuralPredicateElaborator { seen, stack, fcx, span }
    }

    /// For default impls, we need to break apart a type into its
    /// "constituent types" -- meaning, the types that it contains.
    ///
    /// Here are some (simple) examples:
    ///
    /// ```
    /// (i32, u32) -> [i32, u32]
    /// Foo where struct Foo { x: i32, y: u32 } -> [i32, u32]
    /// Bar<i32> where struct Bar<T> { x: T, y: u32 } -> [i32, u32]
    /// Zed<i32> where enum Zed { A(T), B(u32) } -> [i32, u32]
    /// ```
    fn constituent_types_for_auto_trait(&self, t: Ty<'tcx>) -> Vec<Ty<'tcx>> {
        match *t.kind() {
            ty::Projection(..) => {
                bug!("this type should be handled separately: {:?}", t)
            }

            ty::Placeholder(..) | ty::Bound(..) => {
                bug!("do not know how to handle this type: {:?}", t)
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("this type shouldn't show up after normalization: {:?}", t)
            }

            // These types have no constituents
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Str
            | ty::Error(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Never
            | ty::Char
            | ty::Param(_) => vec![],

            // We're treating these types as opaque
            ty::Dynamic(..) | ty::GeneratorWitness(..) | ty::Opaque(..) | ty::Foreign(..) => {
                vec![]
            }

            ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
                vec![element_ty]
            }

            ty::Array(element_ty, _) | ty::Slice(element_ty) => vec![element_ty],

            ty::Tuple(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                tys.iter().map(|k| k.expect_ty()).collect()
            }

            ty::Closure(_, ref substs) => {
                let ty = self.fcx.resolve_vars_if_possible(substs.as_closure().tupled_upvars_ty());
                vec![ty]
            }

            ty::Generator(_, ref substs, _) => {
                let generator = substs.as_generator();
                let ty = self.fcx.resolve_vars_if_possible(generator.tupled_upvars_ty());
                vec![ty]
            }

            // For `PhantomData<T>`, we pass `T`.
            ty::Adt(def, substs) if def.is_phantom_data() => substs.types().collect(),

            ty::Adt(def, substs) => def.all_fields().map(|f| f.ty(self.fcx.tcx, substs)).collect(),
        }
    }
}

impl<'tcx> Iterator for StructuralPredicateElaborator<'_, 'tcx> {
    type Item = ty::GeneratorPredicate<'tcx>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ty) = self.stack.pop() {
            if let ty::Projection(projection_ty) = *ty.kind() {
                let mut normalized_ty = self.fcx.normalize_associated_types_in(self.span, ty);
                if normalized_ty.is_ty_var() {
                    self.fcx.select_obligations_where_possible(false, |_| {});
                    normalized_ty = self.fcx.resolve_vars_if_possible(normalized_ty);
                }
                if !normalized_ty.is_ty_var() && normalized_ty != ty {
                    if self.seen.insert(normalized_ty) {
                        self.stack.push(normalized_ty);
                    }
                    return Some(ty::GeneratorPredicate::Projection(ty::ProjectionPredicate {
                        projection_ty,
                        term: ty::Term::Ty(normalized_ty),
                    }));
                }
            } else {
                let structural: Vec<_> = self
                    .constituent_types_for_auto_trait(ty)
                    .into_iter()
                    .map(|ty| self.fcx.resolve_vars_if_possible(ty))
                    .filter(|ty| self.seen.insert(ty))
                    .collect();
                self.stack.extend(structural);
            }
        }

        None
    }
}
