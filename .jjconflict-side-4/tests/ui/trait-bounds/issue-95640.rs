//@ build-pass
//@ compile-flags:-Zmir-opt-level=3

struct D;

trait Tr {
    type It;
    fn foo(self) -> Option<Self::It>;
}

impl<'a> Tr for &'a D {
    type It = ();
    fn foo(self) -> Option<()> {
        None
    }
}

fn run<F>(f: F)
where
    for<'a> &'a D: Tr,
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
