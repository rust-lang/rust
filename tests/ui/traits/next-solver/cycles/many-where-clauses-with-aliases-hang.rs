//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#210.
//
// Trying to prove `T: Trait<...>` ends up trying to apply all the where-clauses,
// most of which require normalizing some `Alias<T, ...>`. This then requires us
// to prove `T: Trait<...>` again.
//
// This results in a lot of solver cycles whose initial result differs from their
// final result. Reevaluating all of them results in exponential blowup and hangs.
//
// With #144991 we now don't reevaluate cycle heads if their provisional value
// didn't actually impact the final result, avoiding these reruns and allowing us
// to compile this in less than a second.

struct A;
struct B;
struct C;

type Alias<T, U> = <T as Trait<U>>::Assoc;
trait Trait<T> {
    type Assoc;
}

fn foo<T>()
where
    T: Trait<A> + Trait<B> + Trait<C>,
    T: Trait<Alias<T, A>>,
    T: Trait<Alias<T, B>>,
    T: Trait<Alias<T, C>>,
    T: Trait<Alias<T, Alias<T, A>>>,
    T: Trait<Alias<T, Alias<T, B>>>,
    T: Trait<Alias<T, Alias<T, C>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, A>>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, B>>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, C>>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, Alias<T, A>>>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, Alias<T, B>>>>>,
    T: Trait<Alias<T, Alias<T, Alias<T, Alias<T, C>>>>>,
{
}
fn main() {}
