//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/141708>.

// See description in there; this has to do with fundamental limitations
// to the old trait solver surrounding higher-ranked aliases with infer
// vars. This always worked in the new trait solver, but I added a revision
// just for good measure.

trait Foo<'a> {
    type Assoc;
}

impl Foo<'_> for i32 {
    type Assoc = u32;
}

impl Foo<'_> for u32 {
    type Assoc = u32;
}

fn foo<'b: 'b, T: for<'a> Foo<'a>, F: for<'a> Fn(<T as Foo<'a>>::Assoc)>(_: F) -> (T, F) {
    todo!()
}

fn main() {
    let (x, c): (i32, _) = foo::<'static, _, _>(|_| {});
}
