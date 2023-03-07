// build-pass
// compile-flags: -Zpolymorphize=on -Csymbol-mangling-version=v0

pub(crate) struct Foo<'a, I, E>(I, &'a E);

impl<'a, I, T: 'a, E> Iterator for Foo<'a, I, E>
where
    I: Iterator<Item = &'a (T, E)>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.find(|_| true)
    }
}

fn main() {
    let mut a = Foo([(1u32, 1u16)].iter(), &1u16);
    let mut b = Foo([(1u16, 1u32)].iter(), &1u32);
    let _ = a.next();
    let _ = b.next();
}
