// build-fail
// compile-flags:-Zpolymorphize=on
#![feature(rustc_attrs)]

// This test checks that `T` is considered used in `foo`, because it is used in a predicate for
// `I`, which is used.

#[rustc_polymorphize_error]
fn bar<I>() {
    //~^ ERROR item has unused generic parameters
}

#[rustc_polymorphize_error]
fn foo<I, T>(_: I)
//~^ ERROR item has unused generic parameters
where
    I: Iterator<Item = T>,
{
    bar::<I>()
}

#[rustc_polymorphize_error]
fn baz<I, T>(_: I)
//~^ ERROR item has unused generic parameters
where
    std::iter::Repeat<I>: Iterator<Item = T>,
{
    bar::<I>()
}

// In addition, check that `I` is considered used in `next::{{closure}}`, because `T` is used and
// `T` is really just `I::Item`. `E` is used due to the fixed-point marking of predicates.

pub(crate) struct Foo<'a, I, E>(I, &'a E);

impl<'a, I, T: 'a, E> Iterator for Foo<'a, I, E>
where
    I: Iterator<Item = &'a (T, E)>,
{
    type Item = T;

    #[rustc_polymorphize_error]
    fn next(&mut self) -> Option<Self::Item> {
        self.find(|_| true)
        //~^ ERROR item has unused generic parameters
    }
}

// Furthermore, check that `B` is considered used because `C` is used, and that `A` is considered
// used because `B` is now used.

trait Baz<Z> {}

impl Baz<u16> for u8 {}
impl Baz<u32> for u16 {}

#[rustc_polymorphize_error]
fn quux<A, B, C: Default>() -> usize
//~^ ERROR item has unused generic parameters
where
    A: Baz<B>,
    B: Baz<C>,
{
    std::mem::size_of::<C>()
}

// Finally, check that `F` is considered used because `G` is used when neither are in the self-ty
// of the predicate.

trait Foobar<F, G> {}

impl Foobar<u32, u32> for () {}

#[rustc_polymorphize_error]
fn foobar<F, G>() -> usize
//~^ ERROR item has unused generic parameters
where
    (): Foobar<F, G>,
{
    std::mem::size_of::<G>()
}

fn main() {
    let x = &[2u32];
    foo(x.iter());
    baz(x.iter());

    let mut a = Foo([(1u32, 1u16)].iter(), &1u16);
    let _ = a.next();

    let _ = quux::<u8, u16, u32>();

    let _ = foobar::<u32, u32>();
}
