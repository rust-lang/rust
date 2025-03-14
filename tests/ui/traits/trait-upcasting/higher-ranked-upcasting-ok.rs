//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ build-pass

// Check that we are able to instantiate a binder during trait upcasting,
// and that it doesn't cause any issues with codegen either.

trait Supertrait<'a, 'b> {}
trait Subtrait<'a, 'b>: Supertrait<'a, 'b> {}

impl Supertrait<'_, '_> for () {}
impl Subtrait<'_, '_> for () {}
fn ok(x: &dyn for<'a, 'b> Subtrait<'a, 'b>) -> &dyn for<'a> Supertrait<'a, 'a> {
    x
}

fn main() {
    ok(&());
}
