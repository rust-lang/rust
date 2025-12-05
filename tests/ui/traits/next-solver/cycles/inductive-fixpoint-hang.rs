//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// This currently hangs if we do not erase constraints from
// overflow.
//
// We set the provisional result of `W<?0>` to `?0 := W<_>`.
// The next iteration does not simply result in a `?0 := W<W<_>` constraint as
// one might expect, but instead each time we evaluate the nested `W<T>` goal we
// apply the previously returned constraints: the first fixpoint iteration goes
// as follows: `W<?1>: Trait` constrains `?1` to `W<?2>`, we then evaluate
// `W<W<?2>>: Trait` the next time we try to prove the nested goal. This results
// inn `W<W<W<?3>>>` and so on. This goes on until we reach overflow in
// `try_evaluate_added_goals`.  This means the provisional result after the
// second fixpoint iteration is already `W<W<W<...>>>` with a size proportional
// to the number of steps in `try_evaluate_added_goals`. The size then continues
// to grow. The exponential blowup from having 2 nested goals per impl causes
// the solver to hang without hitting the recursion limit.
trait Trait {}

struct W<T>(*const T);

impl<T> Trait for W<W<T>>
where
    W<T>: Trait,
    W<T>: Trait,
{}

fn impls_trait<T: Trait>() {}

fn main() {
    impls_trait::<W<_>>();
    //~^ ERROR overflow evaluating the requirement
}
