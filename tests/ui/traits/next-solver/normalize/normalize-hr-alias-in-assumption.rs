//@ compile-flags: -Znext-solver
//@ check-pass

// regression test for https://github.com/rust-lang/rust/issues/163817

trait Ref {
    type Assoc<'z, T>;
}

impl Ref for u8 {
    type Assoc<'a, U> = &'a u8;
}

trait Tr {
    type Out<F>: Iterator<Item = for<'a> fn(<u8 as Ref>::Assoc<'a, F>)>;
}

fn require<T: Tr, F>()
where
    <T as Tr>::Out<F>: Iterator<Item = F>,
{
}

fn outer<T: Tr>() {
    require::<T, _>();
}

fn main() {}
