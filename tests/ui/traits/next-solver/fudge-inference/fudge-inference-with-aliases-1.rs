//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/252.
// `fn fudge_inference_if_ok` might lose relationships between ty vars so we need to normalize
// them inside the fudge scope.

trait Trait {
    type Assoc;
}
impl<T: Trait> Trait for W<T> {
    type Assoc = T::Assoc;
}
impl Trait for i32 {
    type Assoc = i32;
}

struct W<T>(T);
fn foo<T: Trait>(_: <T as Trait>::Assoc) -> T {
    todo!()
}

fn main() {
    let x: W<_> = foo(1);
    let _: W<i32> = x;
}
