use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::undo_log::NoUndo;
use rustc_data_structures::unify as ut;
use rustc_middle::ty;

use crate::infer::InferCtxt;

#[derive(Debug, Copy, Clone, PartialEq)]
struct SubId(u32);
impl ut::UnifyKey for SubId {
    type Value = ();
    #[inline]
    fn index(&self) -> u32 {
        self.0
    }
    #[inline]
    fn from_index(i: u32) -> SubId {
        SubId(i)
    }
    fn tag() -> &'static str {
        "SubId"
    }
}

/// When reporting ambiguity errors, we sometimes want to
/// treat all inference vars which are subtypes of each
/// others as if they are equal. For this case we compute
/// the transitive closure of our subtype obligations here.
///
/// E.g. when encountering ambiguity errors, we want to suggest
/// specifying some method argument or to add a type annotation
/// to a local variable. Because subtyping cannot change the
/// shape of a type, it's fine if the cause of the ambiguity error
/// is only related to the suggested variable via subtyping.
///
/// Even for something like `let x = returns_arg(); x.method();` the
/// type of `x` is only a supertype of the argument of `returns_arg`. We
/// still want to suggest specifying the type of the argument.
#[derive(Default)]
pub struct SubRelations {
    map: FxHashMap<ty::TyVid, SubId>,
    table: ut::UnificationTableStorage<SubId>,
}

impl SubRelations {
    fn get_id<'tcx>(&mut self, infcx: &InferCtxt<'tcx>, vid: ty::TyVid) -> SubId {
        let root_vid = infcx.root_var(vid);
        *self.map.entry(root_vid).or_insert_with(|| self.table.with_log(&mut NoUndo).new_key(()))
    }

    pub fn add_constraints<'tcx>(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obls: impl IntoIterator<Item = ty::Predicate<'tcx>>,
    ) {
        for p in obls {
            let (a, b) = match p.kind().skip_binder() {
                ty::PredicateKind::Subtype(ty::SubtypePredicate { a_is_expected: _, a, b }) => {
                    (a, b)
                }
                ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => (a, b),
                _ => continue,
            };

            match (a.kind(), b.kind()) {
                (&ty::Infer(ty::TyVar(a_vid)), &ty::Infer(ty::TyVar(b_vid))) => {
                    let a = self.get_id(infcx, a_vid);
                    let b = self.get_id(infcx, b_vid);
                    self.table.with_log(&mut NoUndo).unify_var_var(a, b).unwrap();
                }
                _ => continue,
            }
        }
    }

    pub fn unified<'tcx>(&mut self, infcx: &InferCtxt<'tcx>, a: ty::TyVid, b: ty::TyVid) -> bool {
        let a = self.get_id(infcx, a);
        let b = self.get_id(infcx, b);
        self.table.with_log(&mut NoUndo).unioned(a, b)
    }
}
