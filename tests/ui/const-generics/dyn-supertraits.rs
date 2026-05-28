//@ run-pass

trait Foo<const N: usize> {
    fn myfun(&self) -> usize;
}
trait Bar<const N: usize> : Foo<N> {}
trait Baz: Foo<3> {} //~ WARN trait `Baz` is never used

struct FooType<const N: usize>;
struct BarType<const N: usize>;
struct BazType;

impl<const N: usize> Foo<N> for FooType<N> {
    fn myfun(&self) -> usize { N }
}
impl<const N: usize> Foo<N> for BarType<N> {
    fn myfun(&self) -> usize { N + 1 }
}
impl<const N: usize> Bar<N> for BarType<N> {}
impl Foo<3> for BazType {
    fn myfun(&self) -> usize { 999 }
}
impl Baz for BazType {}

trait Foz {}
trait Boz: Foo<3> + Foz {} //~ WARN trait `Boz` is never used
trait Bok<const N: usize>: Foo<N> + Foz {}

struct FozType; //~ WARN struct `FozType` is never constructed
struct BozType;
struct BokType<const N: usize>;

impl Foz for FozType {}

impl Foz for BozType {}
impl Foo<3> for BozType {
    fn myfun(&self) -> usize { 9999 }
}
impl Boz for BozType {}

impl<const N: usize> Foz for BokType<N> {}
impl<const N: usize> Foo<N> for BokType<N> {
    fn myfun(&self) -> usize { N + 2 }
}
impl<const N: usize> Bok<N> for BokType<N> {}

fn a<const N: usize>(x: &dyn Foo<N>) -> usize { x.myfun() }
fn b(x: &dyn Foo<3>) -> usize { x.myfun() }
fn c<T: Bok<N>, const N: usize>(x: T) -> usize { a::<N>(&x) }
fn d<T: ?Sized + Foo<3>>(x: &T) -> usize { x.myfun() }
fn e(x: &dyn Bar<3>) -> usize { d(x) }

fn main() {
    let foo = FooType::<3> {};
    assert!(a(&foo) == 3);
    assert!(b(&foo) == 3);
    assert!(d(&foo) == 3);

    let bar = BarType::<3> {};
    assert!(a(&bar) == 4);
    assert!(b(&bar) == 4);
    assert!(d(&bar) == 4);
    assert!(e(&bar) == 4);

    let baz = BazType {};
    assert!(a(&baz) == 999);
    assert!(b(&baz) == 999);
    assert!(d(&baz) == 999);

    let boz = BozType {};
    assert!(a(&boz) == 9999);
    assert!(b(&boz) == 9999);
    assert!(d(&boz) == 9999);

    let bok = BokType::<3> {};
    assert!(a(&bok) == 5);
    assert!(b(&bok) == 5);
    assert!(d(&bok) == 5);
    assert!(c(BokType::<3> {}) == 5);
}
