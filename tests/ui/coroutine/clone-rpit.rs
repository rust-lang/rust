// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// check-pass

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
