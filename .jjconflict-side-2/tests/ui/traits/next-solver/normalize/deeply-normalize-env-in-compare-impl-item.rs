//@ check-pass
//@ compile-flags: -Znext-solver

// See trait-system-refactor-initiative/issues/166.
// The old solver doesn't check normalization constraints in `compare_impl_item`.
// The new solver performs lazy normalization so those region constraints may get postponed to
// an infcx that considers regions.
trait Trait {
    type Assoc<'a>
    where
        Self: 'a;
}
impl<'b> Trait for &'b u32 {
    type Assoc<'a> = &'a u32
    where
        Self: 'a;
}

trait Bound<T> {}
trait Entailment<T: Trait> {
    fn method()
    where
        Self: for<'a> Bound<<T as Trait>::Assoc<'a>>;
}

impl<'b, T> Entailment<&'b u32> for T {
    // Instantiates trait where-clauses with `&'b u32` and then normalizes
    // `T: for<'a> Bound<<&'b u32 as Trait>::Assoc<'a>>` in a separate infcx
    // without checking region constraints.
    //
    // It normalizes to `T: Bound<&'a u32>`, dropping the `&'b u32: 'a` constraint.
    fn method()
    where
        Self: for<'a> Bound<&'a u32>
    {}
}

fn main() {}
