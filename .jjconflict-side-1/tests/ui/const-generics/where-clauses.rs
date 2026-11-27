//@ check-pass
trait Bar<const N: usize> { fn bar() {} }
trait Foo<const N: usize>: Bar<N> {}

fn test<T, const N: usize>() where T: Foo<N> {
    <T as Bar<N>>::bar();
}

struct Faz<const N: usize>;

impl<const N: usize> Faz<N> {
    fn test<T>() where T: Foo<N> {
        <T as Bar<N>>::bar()
    }
}

trait Fiz<const N: usize> {
    fn fiz<T>() where T: Foo<N> {
        <T as Bar<N>>::bar();
    }
}

impl<const N: usize> Bar<N> for u8 {}
impl<const N: usize> Foo<N> for u8 {}
impl<const N: usize> Fiz<N> for u8 {}
fn main() {
    test::<u8, 13>();
    Faz::<3>::test::<u8>();
    <u8 as Fiz<13>>::fiz::<u8>();
}
