#![feature(generic_arg_infer)]

struct Foo<const N: usize>;

impl Clone for Foo<1> {
    fn clone(&self) -> Self {
        Self
    }
}
impl Copy for Foo<1> {}

fn unify<const N: usize>(_: &[Foo<N>; 2], _: &[String; N]) {}

fn works_if_inference_side_effects() {
    // This will only pass if inference side effects from proving `Foo<?x>: Copy` are
    // able to be relied upon by other repeat expressions.
    let a /* : [Foo<?x>; 2] */ = [Foo::<_>; 2];
    //~^ ERROR: type annotations needed for `[Foo<_>; 2]`
    let b /* : [String; ?x] */ = ["string".to_string(); _];

    unify(&a, &b);
}

fn works_if_fixed_point() {
    // This will only pass if the *second* array repeat expr is checked first
    // allowing `Foo<?x>: Copy` to infer the array length of the first repeat expr.
    let b /* : [String; ?x] */ = ["string".to_string(); _];
    //~^ ERROR: type annotations needed for `[String; _]`
    let a /* : [Foo<?x>; 2] */ = [Foo::<_>; 2];

    unify(&a, &b);
}

fn main() {}
