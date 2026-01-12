// Projection predicate's WFedness requires that the rhs term satisfy all item bounds defined on the
// associated type.
// Generic types have those bounds implied.

trait Required {}

trait AssocHasBound {
    type Assoc: Required;
}

trait Trait<T> {
    type Assoc1: AssocHasBound<Assoc = i32>;
    //~^ ERROR: the trait bound `i32: Required` is not satisfied [E0277]
    type Assoc2: AssocHasBound<Assoc = T>;
    type Assoc3: AssocHasBound<Assoc = Self::DummyAssoc>;
    type DummyAssoc;
}

fn some_func<T1, T2, U>()
where
    T1: AssocHasBound<Assoc = i32>,
    //~^ ERROR: type annotations needed [E0284]
    T1: AssocHasBound<Assoc = U>,
{}

fn opaque_with_concrete_assoc(_: impl AssocHasBound<Assoc = i32>) {}
//~^ ERROR: the trait bound `i32: Required` is not satisfied [E0277]

fn opaque_with_generic_assoc<T>(_: impl AssocHasBound<Assoc = T>) {}

fn main() {}
