//@ check-pass
//@ compile-flags: -Znext-solver

// When canonicalizing a response in the trait solver, we bail with overflow
// if there are too many non-region inference variables. Doing so in normalizes-to
// goals ends up hiding inference constraints in cases which we want to support,
// see #131969. To prevent this issue we do not check for too many inference
// variables in normalizes-to goals.
#![recursion_limit = "8"]

trait Bound {}
trait Trait {
    type Assoc;
}


impl<T0, T1, T2, T3, T4, T5, T6, T7> Trait for (T0, T1, T2, T3, T4, T5, T6, T7)
where
    T0: Trait,
    T1: Trait,
    T2: Trait,
    T3: Trait,
    T4: Trait,
    T5: Trait,
    T6: Trait,
    T7: Trait,
    (
        T0::Assoc,
        T1::Assoc,
        T2::Assoc,
        T3::Assoc,
        T4::Assoc,
        T5::Assoc,
        T6::Assoc,
        T7::Assoc,
    ): Clone,
{
    type Assoc = (
        T0::Assoc,
        T1::Assoc,
        T2::Assoc,
        T3::Assoc,
        T4::Assoc,
        T5::Assoc,
        T6::Assoc,
        T7::Assoc,
    );
}

trait Overlap {}
impl<T: Trait<Assoc = ()>> Overlap for T {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Overlap for (T0, T1, T2, T3, T4, T5, T6, T7) {}
fn main() {}
