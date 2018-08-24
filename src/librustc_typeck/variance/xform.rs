use rustc::ty;

pub fn glb(v1: ty::Variance, v2: ty::Variance) -> ty::Variance {
    // Greatest lower bound of the variance lattice as
    // defined in The Paper:
    //
    //       *
    //    -     +
    //       o
    match (v1, v2) {
        (ty::Invariant, _) |
        (_, ty::Invariant) => ty::Invariant,

        (ty::Covariant, ty::Contravariant) => ty::Invariant,
        (ty::Contravariant, ty::Covariant) => ty::Invariant,

        (ty::Covariant, ty::Covariant) => ty::Covariant,

        (ty::Contravariant, ty::Contravariant) => ty::Contravariant,

        (x, ty::Bivariant) |
        (ty::Bivariant, x) => x,
    }
}
