//@ known-bug: #137813
trait AssocConst {
    const A: u8;
}

impl<T> AssocConst for (T,) {
    const A: u8 = 0;
}

trait Trait {}

impl<U> Trait for () where (U,): AssocConst<A = { 0 }> {}

fn foo()
where
    (): Trait,
{
}
