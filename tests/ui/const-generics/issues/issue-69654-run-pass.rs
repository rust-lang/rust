//@ run-pass
trait Bar<T> {} //~ WARN trait `Bar` is never used
impl<T> Bar<T> for [u8; 7] {}

struct Foo<const N: usize> {}
impl<const N: usize> Foo<N>
where
    [u8; N]: Bar<[(); N]>,
{
    fn foo() {}
}

fn main() {
    Foo::foo();
}
