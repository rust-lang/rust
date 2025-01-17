//@ run-pass
//@ compile-flags: -Znext-solver
//@ revisions: stock precise

#![feature(const_trait_impl, const_destruct)]
#![feature(never_type)]
#![cfg_attr(precise, feature(const_precise_live_drops))]

use std::marker::Destruct;

struct S<'a>(&'a mut u8);

impl<'a> const Drop for S<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

const fn a<T: ~const Destruct>(_: T) {}
//FIXME ~^ ERROR destructor of

const fn b() -> u8 {
    let mut c = 0;
    let _ = S(&mut c);
    //FIXME ~^ ERROR destructor of
    a(S(&mut c));
    c
}

const C: u8 = b();

macro_rules! implements_const_drop {
    ($($exp:expr),*$(,)?) => {
        $(
            const _: () = a($exp);
        )*
    }
}

#[allow(dead_code)]
mod t {
    pub struct Foo;
    pub enum Bar { A }
    pub fn foo() {}
    pub struct ConstDrop;

    impl const Drop for ConstDrop {
        fn drop(&mut self) {}
    }

    pub struct HasConstDrop(pub ConstDrop);
    pub struct TrivialFields(pub u8, pub i8, pub usize, pub isize);

    #[const_trait]
    pub trait SomeTrait {
        fn foo();
    }
    impl const SomeTrait for () {
        fn foo() {}
    }
    // non-const impl
    impl SomeTrait for i32 {
        fn foo() {}
    }

    pub struct ConstDropWithBound<T: const SomeTrait>(pub core::marker::PhantomData<T>);

    impl<T: const SomeTrait> const Drop for ConstDropWithBound<T> {
        fn drop(&mut self) {
            T::foo();
        }
    }

    pub struct ConstDropWithNonconstBound<T: SomeTrait>(pub core::marker::PhantomData<T>);

    impl<T: SomeTrait> const Drop for ConstDropWithNonconstBound<T> {
        fn drop(&mut self) {
            // Note: we DON'T use the `T: SomeTrait` bound
        }
    }
}

use t::*;

implements_const_drop! {
    1u8,
    2,
    3.0,
    Foo,
    Bar::A,
    foo,
    ConstDrop,
    HasConstDrop(ConstDrop),
    TrivialFields(1, 2, 3, 4),
    &1,
    &1 as *const i32,
    ConstDropWithBound::<()>,
    ConstDropWithNonconstBound::<i32>,
    Result::<i32, !>::Ok(1),
}

fn main() {
    struct HasDropGlue(#[allow(dead_code)] Box<u8>);
    struct HasDropImpl;
    impl Drop for HasDropImpl {
        fn drop(&mut self) {
            println!("not trivial drop");
        }
    }

    // These types should pass because ~const in a non-const context should have no effect.
    a(HasDropGlue(Box::new(0)));
    a(HasDropImpl);

    assert_eq!(C, 2);
}
