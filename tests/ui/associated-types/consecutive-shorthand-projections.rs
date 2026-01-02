//@ check-pass

fn factory0<T: Factory>() {
    let _: T::Output::Category;
}

fn factory1<T: Factory<Output: Product<Category = u16>>>(category: u16) {
    let _: T::Output::Category = category;
}

fn factory2<T: Factory>(_: T::Output::Category) {}

trait Factory {
    type Output: Product;
}

impl Factory for () {
    type Output = u128;
}

trait Product {
    type Category;
}

impl Product for u128 {
    type Category = u16;
}

/////////////////////////

fn chain<C: Chain<Link = C>>(chain: C) {
    let _: C::Link::Link::Link::Link::Link = chain;
}

trait Chain { type Link: Chain; }

impl Chain for () {
    type Link = Self;
}

/////////////////////////

fn scope<'r, T: Main<'static, (i32, U), 1>, U, const Q: usize>() {
    let _: T::Next<'r, (), Q>::Final;
}

trait Main<'a, T, const N: usize> {
    type Next<'b, U, const M: usize>: Aux<'a, 'b, (T, U), N, M>;
}

impl<'a, T, const N: usize> Main<'a, T, N> for () {
    type Next<'_b, _U, const _M: usize> = ();
}

trait Aux<'a, 'b, T, const N: usize, const M: usize> {
    type Final;
}

impl<'a, 'b, T, const N: usize, const M: usize> Aux<'a, 'b, T, N, M> for () {
    type Final = [[(T, &'a (), &'b ()); N]; M];
}

/////////////////////////

fn main() {
    factory0::<()>();
    factory1::<()>(360);
    factory2::<()>(720);
    let _: <() as Factory>::Output::Category;

    chain(());
    let _: <() as Chain>::Link::Link::Link;

    scope::<(), bool, 32>();
}
