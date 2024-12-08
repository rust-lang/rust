//@ build-pass
// needs to be build-pass, because it is a regression test for a mir validation failure
// that only happens during codegen.

struct D;

trait Tr {
    type It;
    fn foo(self) -> Option<Self::It>;
}

impl<'a> Tr for &'a D {
    type It = ();
    fn foo(self) -> Option<()> { None }
}

fn run<F>(f: F)
    where for<'a> &'a D: Tr,
          F: Fn(<&D as Tr>::It),
{
    let d = &D;
    while let Some(i) = d.foo() {
        f(i);
    }
}

fn main() {
    run(|_| {});
}
