//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for the third variant of trait-system-refactor-initiative#191

trait Ref<'a, F> {
    type Input;
}

impl<'a, F> Ref<'a, F> for u32 {
    type Input = &'a u32;
}

fn needs_super<F: for<'a> Fn(<u32 as Ref<'a, F>>::Input)>(_: F) {}

fn main() {
    needs_super(|_| {});
}
