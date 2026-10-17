//@ check-pass
//@ compile-flags: -Znext-solver=globally
//
// regression test for #162850
//
// a manual `Send` impl can produce equivalent fixpoint responses with
// old-style region constraints in different insertion orders

#![allow(unused)]

mod vec_impl {
    pub trait Allocator {}

    pub struct Global;
    impl Allocator for Global {}

    struct RawVecInner<A: Allocator = Global>(
        std::ptr::NonNull<u8>,
        usize,
        A,
    );

    struct RawVec<T, A: Allocator = Global>(
        RawVecInner<A>,
        std::marker::PhantomData<T>,
    );

    unsafe impl<A: Allocator + Send> Send for RawVecInner<A> {}

    unsafe impl<T: Send, A: Allocator + Send> Send for RawVec<T, A> {}

    pub struct MyVec<T, A: Allocator = Global>(RawVec<T, A>);

    impl<T> MyVec<T> {
        pub fn new() -> Self {
            panic!()
        }
    }
}

use vec_impl::MyVec;

pub trait IntoParallelIterator {
    type Iter;

    fn into_par_iter(self) -> Self::Iter;
}

impl<T: Send> IntoParallelIterator for MyVec<T> {
    type Iter = T;

    fn into_par_iter(self) -> Self::Iter {
        todo!()
    }
}

struct RuleA<R>(MyVec<CssRule<R>>);
struct RuleB<R>(RuleA<R>);
struct CssRule<R>(RuleB<R>, R);

struct Bundler<T>(T);

trait AtRuleParser<'i> {
    type AtRule;
}

impl<'a, T> Bundler<T>
where
    T: AtRuleParser<'a>,
    T::AtRule: Send,
{
    fn load_file(&self) {
        _ = MyVec::<CssRule<T::AtRule>>::new().into_par_iter();
    }
}

fn main() {}
