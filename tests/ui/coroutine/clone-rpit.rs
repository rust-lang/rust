//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] check-pass
//@[next] known-bug: trait-system-refactor-initiative#82

#![feature(coroutines, coroutine_trait, coroutine_clone)]

// This stalls the goal `{coroutine} <: impl Clone`, since that has a nested goal
// of `{coroutine}: Clone`. That is only known if we can compute the generator
// witness types, which we don't know until after borrowck. When we later check
// the goal for correctness, we want to be able to bind the `impl Clone` opaque.
pub fn foo<'a, 'b>() -> impl Clone {
    move |_: ()| {
        let () = yield ();
    }
}

fn main() {}
