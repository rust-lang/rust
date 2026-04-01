//@ compile-flags: -Znext-solver
//@ check-pass

// If we normalize using the impl here the constraints from normalization and
// trait goals can differ. This is especially bad if normalization results
// in stronger constraints.
trait Trait<'a> {
    type Assoc;
}

impl<T> Trait<'static> for T {
    type Assoc = ();
}

// normalizing requires `'a == 'static`, the trait bound does not.
fn foo<'a, T: Trait<'a>>(_: T::Assoc) {}

fn main() {}
