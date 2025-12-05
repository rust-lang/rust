//@ build-pass (FIXME(62277): could be check-pass?)

use std::iter::Once;
use std::ops::Range;

pub trait Three { type A; type B; type C; }
pub fn assert_three<T: ?Sized + Three>() {}
pub fn assert_iterator<T: Iterator>() {}
pub fn assert_copy<T: Copy>() {}
pub fn assert_static<T: 'static>() {}
pub fn assert_send<T: Send>() {}
pub fn assert_forall_into<T: for<'a> Into<&'a u8>>() {}

struct A; struct B;
impl<'a> Into<&'a u8> for A { fn into(self) -> &'a u8 { &0 } }
impl Three for B { type A = Range<u8>; type B = Range<u8>; type C = Range<u8>; }

trait Case1<A, B, C, D, E>
where
    A: Iterator<Item: Copy>,
    B: Iterator<Item: 'static>,
    C: Iterator<Item: 'static + Copy + Send>,
    D: Iterator<Item: for<'a> Into<&'a u8>>,
    E: Three<A: Iterator<Item: Copy>, B: Iterator<Item: Copy>, C: Iterator<Item: Copy>>,
    Self: Three<A: 'static, B: Copy, C: Send>,
{
    fn _a() {
        assert_iterator::<A>();
        assert_copy::<A::Item>();
    }
    fn _b() {
        assert_iterator::<B>();
        assert_static::<B::Item>();
    }
    fn _c() {
        assert_iterator::<C>();
        assert_copy::<C::Item>();
        assert_static::<C::Item>();
        assert_send::<C::Item>();
    }
    fn _d() {
        assert_iterator::<D>();
        assert_forall_into::<D::Item>();
    }
    fn _e() {
        assert_three::<E>();
        assert_iterator::<E::A>();
        assert_iterator::<E::B>();
        assert_iterator::<E::C>();
        assert_copy::<<E::A as Iterator>::Item>();
        assert_copy::<<E::B as Iterator>::Item>();
        assert_copy::<<E::C as Iterator>::Item>();
    }
    fn _self() {
        assert_three::<Self>();
        assert_copy::<Self::B>();
        assert_static::<Self::A>();
        assert_send::<Self::C>();
    }
}

struct DataCase1;
impl Three for DataCase1 { type A = u8; type B = u8; type C = u8; }
impl Case1<Range<u8>, Range<u8>, Range<u8>, Once<A>, B> for DataCase1 {}

trait Case2<
    A: Iterator<Item: Copy>,
    B: Iterator<Item: 'static>,
    C: Iterator<Item: 'static + Copy + Send>,
    D: Iterator<Item: for<'a> Into<&'a u8>>,
    E: Three<A: Iterator<Item: Copy>, B: Iterator<Item: Copy>, C: Iterator<Item: Copy>>,
>:
    Three<A: 'static, B: Copy, C: Send>
{
    fn _a() {
        assert_iterator::<A>();
        assert_copy::<A::Item>();
    }
    fn _b() {
        assert_iterator::<B>();
        assert_static::<B::Item>();
    }
    fn _c() {
        assert_iterator::<C>();
        assert_copy::<C::Item>();
        assert_static::<C::Item>();
        assert_send::<C::Item>();
    }
    fn _d() {
        assert_iterator::<D>();
        assert_forall_into::<D::Item>();
    }
    fn _e() {
        assert_three::<E>();
        assert_iterator::<E::A>();
        assert_iterator::<E::B>();
        assert_iterator::<E::C>();
        assert_copy::<<E::A as Iterator>::Item>();
        assert_copy::<<E::B as Iterator>::Item>();
        assert_copy::<<E::C as Iterator>::Item>();
    }
    fn _self() {
        assert_three::<Self>();
        assert_copy::<Self::B>();
        assert_static::<Self::A>();
        assert_send::<Self::C>();
    }
}

struct DataCase2;
impl Three for DataCase2 { type A = u8; type B = u8; type C = u8; }
impl Case2<Range<u8>, Range<u8>, Range<u8>, Once<A>, B> for DataCase2 {}

fn main() {}
