// Regression test for #83606.

fn foo<const N: usize>(_: impl std::fmt::Display) -> [usize; N] {
    [0; N]
}

fn main() {
    let _ = foo("foo"); //<- Do not suggest `foo::<N>("foo");`!
    //~^ ERROR: type annotations needed for `[usize; _]`
}
