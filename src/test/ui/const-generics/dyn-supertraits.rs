// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait Foo<const N: usize> {}
trait Bar<const N: usize> : Foo<N> {}
trait Baz: Foo<3> {}

struct FooType<const N: usize> {}
struct BarType<const N: usize> {}
struct BazType {}

impl<const N: usize> Foo<N> for FooType<N> {}
impl<const N: usize> Foo<N> for BarType<N> {}
impl<const N: usize> Bar<N> for BarType<N> {}
impl Foo<3> for BazType {}
impl Baz for BazType {}

trait Foz {}
trait Boz: Foo<3> + Foz {}
trait Bok<const N: usize>: Foo<N> + Foz {}

struct FozType {}
struct BozType {}
struct BokType<const N: usize> {}

impl Foz for FozType {}

impl Foz for BozType {}
impl Foo<3> for BozType {}
impl Boz for BozType {}

impl<const N: usize> Foz for BokType<N> {}
impl<const N: usize> Foo<N> for BokType<N> {}
impl<const N: usize> Bok<N> for BokType<N> {}

fn a<const N: usize>(x: &dyn Foo<N>) {}
fn b(x: &dyn Foo<3>) {}

fn main() {
    let foo = FooType::<3> {};
    a(&foo); b(&foo);

    let bar = BarType::<3> {};
    a(&bar); b(&bar);

    let baz = BazType {};
    a(&baz); b(&baz);

    let boz = BozType {};
    a(&boz); b(&boz);

    let bok = BokType::<3> {};
    a(&bok); b(&bok);
}
