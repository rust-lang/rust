use crate::ty::{
    Const, ProjectionPredicate, RegionOutlivesPredicate, TraitPredicate, Ty, TypeOutlivesPredicate,
};

/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum Clause<'tcx> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(TraitPredicate<'tcx>),

    /// `where 'a: 'b`
    RegionOutlives(RegionOutlivesPredicate<'tcx>),

    /// `where T: 'a`
    TypeOutlives(TypeOutlivesPredicate<'tcx>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(ProjectionPredicate<'tcx>),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(Const<'tcx>, Ty<'tcx>),
}
