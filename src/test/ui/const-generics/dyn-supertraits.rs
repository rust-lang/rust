// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait Foo<const N: usize> {
    fn myfun(&self) -> usize;
}
trait Bar<const N: usize> : Foo<N> {}
trait Baz: Foo<3> {}

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
trait Boz: Foo<3> + Foz {}
trait Bok<const N: usize>: Foo<N> + Foz {}

struct FozType;
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

fn a<const N: usize>(_: &dyn Foo<N>) {}
fn b(_: &dyn Foo<3>) {}
fn c<T: Bok<N>, const N: usize>(x: T) { a::<N>(&x); }
fn d<T: ?Sized + Foo<3>>(_: &T) {}
fn e(x: &dyn Bar<3>) { d(x); }

fn get_myfun<const N: usize>(x: &dyn Foo<N>) -> usize { x.myfun() }

fn main() {
    let foo = FooType::<3> {};
    a(&foo); b(&foo); d(&foo);
    assert!(get_myfun(&foo) == 3);

    let bar = BarType::<3> {};
    a(&bar); b(&bar); d(&bar); e(&bar);
    assert!(get_myfun(&bar) == 4);

    let baz = BazType {};
    a(&baz); b(&baz); d(&baz);
    assert!(get_myfun(&baz) == 999);

    let boz = BozType {};
    a(&boz); b(&boz); d(&boz);
    assert!(get_myfun(&boz) == 9999);

    let bok = BokType::<3> {};
    a(&bok); b(&bok); d(&bok);
    assert!(get_myfun(&bok) == 5);
    
    c(BokType::<3> {});
}
