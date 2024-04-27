//@ check-pass
trait Bar<'a> {
    type Assoc: 'static;
}

impl<'a> Bar<'a> for () {
    type Assoc = ();
}

struct ImplsStatic<CG: Bar<'static>> {
    d: &'static <CG as Bar<'static>>::Assoc,
}

fn caller(b: ImplsStatic<()>)
where
    for<'a> (): Bar<'a>
{
    let _: &<() as Bar<'static>>::Assoc = b.d;
}

fn main() {}
