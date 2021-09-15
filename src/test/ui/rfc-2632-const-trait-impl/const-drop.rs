// run-pass
// revisions: stock precise
#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]
#![feature(const_mut_refs)]
#![feature(const_panic)]
#![cfg_attr(precise, feature(const_precise_live_drops))]

struct S<'a>(&'a mut u8);

impl<'a> const Drop for S<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

const fn a<T: ~const Drop>(_: T) {}

const fn b() -> u8 {
    let mut c = 0;
    let _ = S(&mut c);
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
}

fn main() {
    struct HasDropGlue(Box<u8>);
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
