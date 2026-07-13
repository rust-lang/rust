//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// The old solver doesn't verify depth when looking up cache.
// To avoid breakage, we evaluate with higher recursion limit in the next solver
// and emit an FCW for this.
// See the `NEXT_TRAIT_SOLVER_OVERFLOW` FCW.

#![recursion_limit = "8"]

// The field order matters 😂
#[allow(dead_code)]
struct Foo<T> {
    t: T,
    opt_t: Option<T>,
}

fn require_sync<T: Sync>() {}

fn main() {
    require_sync::<Foo<Foo<Foo<Foo<Foo<Foo<()>>>>>>>();
    //[next]~^ WARN: overflow evaluating the requirement `Foo<Foo<Foo<Foo<Foo<Foo<()>>>>>>: Sync` [next_trait_solver_overflow]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN: overflow evaluating the requirement `Foo<Foo<Foo<Foo<Foo<Foo<()>>>>>>: Sync` [next_trait_solver_overflow]
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

}
