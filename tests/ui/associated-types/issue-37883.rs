//@ check-pass

use std::ops::Mul;

fn main() {}

trait Ring {}
trait Real: Ring {}

trait Module: Sized + Mul<<Self as Module>::Ring, Output = Self> {
    type Ring: Ring;
}

trait EuclideanSpace {
    type Coordinates: Module<Ring = Self::Real>;
    type Real: Real;
}

trait Translation<E: EuclideanSpace> {
    fn to_vector(&self) -> E::Coordinates;

    fn powf(&self, n: <E::Coordinates as Module>::Ring) -> E::Coordinates {
        self.to_vector() * n
    }
}
