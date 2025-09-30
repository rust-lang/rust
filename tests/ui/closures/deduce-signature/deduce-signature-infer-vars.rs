//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for an issue found in #146720.

trait Trait {
    type Assoc;
}

struct W<T>(T);
impl<T: Trait> Trait for W<T>
where
    u32: Trait<Assoc = T::Assoc>,
{
    type Assoc = T;
}

impl<T> Trait for (T,) {
    type Assoc = T;
}

impl Trait for u32 {
    type Assoc = u32;
}

fn foo<T: Trait>(_: impl FnOnce(T::Assoc)) {}

fn main() {
    // The closure signature is `fn(<W<?infer> as Trait>::Assoc)`.
    // Normalizing it results in `?t` with `u32: Trait<Assoc = <?t>::Assoc>`.
    // Equating `?t` with the argument pattern constrains it to `(?t,)`, at
    // which point the `u32: Trait<Assoc = <?t>::Assoc>` obligations constrains
    // `(?t,)` to `(u32,)`.
    //
    // This breaks when fudging inference to replace `?t` with an unconstrained
    // infer var.
    foo::<W<_>>(|(field,)| { let _ = field.count_ones(); })
}
