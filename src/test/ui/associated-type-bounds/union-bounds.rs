// run-pass

#![feature(associated_type_bounds)]
#![feature(untagged_unions)]

#![allow(unused_assignments)]

trait Tr1: Copy { type As1: Copy; }
trait Tr2: Copy { type As2: Copy; }
trait Tr3: Copy { type As3: Copy; }
trait Tr4<'a>: Copy { type As4: Copy; }
trait Tr5: Copy { type As5: Copy; }

impl Tr1 for &str { type As1 = bool; }
impl Tr2 for bool { type As2 = u8; }
impl Tr3 for u8 { type As3 = fn() -> u8; }
impl Tr1 for () { type As1 = (usize,); }
impl<'a> Tr4<'a> for (usize,) { type As4 = u8; }
impl Tr5 for bool { type As5 = u16; }

union Un1<T: Tr1<As1: Tr2>> {
    outest: T,
    outer: T::As1,
    inner: <T::As1 as Tr2>::As2,
}

union Un2<T: Tr1<As1: Tr2<As2: Tr3>>> {
    outest: T,
    outer: T::As1,
    inner: <T::As1 as Tr2>::As2,
}

union Un3<T: Tr1<As1: 'static>> {
    outest: T,
    outer: &'static T::As1,
}

union Un4<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l>>> {
    f1: &'x1 <T::As1 as Tr4<'x1>>::As4,
    f2: &'x2 <T::As1 as Tr4<'x2>>::As4,
}

union _Un5<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l, As4: Copy>>> {
    f1: &'x1 <T::As1 as Tr4<'x1>>::As4,
    f2: &'x2 <T::As1 as Tr4<'x2>>::As4,
}

union Un6<T>
where
    T: Tr1<As1: Tr2 + 'static + Tr5>,
{
    f0: T,
    f1: <T::As1 as Tr2>::As2,
    f2: &'static T::As1,
    f3: <T::As1 as Tr5>::As5,
}

union _Un7<'a, 'b, T> // `<T::As1 as Tr2>::As2: 'a` is implied.
where
    T: Tr1<As1: Tr2>,
{
    f0: &'a T,
    f1: &'b <T::As1 as Tr2>::As2,
}

unsafe fn _use_un7<'a, 'b, T>(x: _Un7<'a, 'b, T>)
where
    T: Tr1,
    T::As1: Tr2,
{
    let _: &'a T = &x.f0;
}

#[derive(Copy, Clone)]
union UnSelf<T> where Self: Tr1<As1: Tr2>, T: Copy {
    f0: T,
    f1: <Self as Tr1>::As1,
    f2: <<Self as Tr1>::As1 as Tr2>::As2,
}

impl Tr1 for UnSelf<&'static str> { type As1 = bool; }

fn main() {
    let mut un1 = Un1 { outest: "foo" };
    un1 = Un1 { outer: true };
    assert_eq!(unsafe { un1.outer }, true);
    un1 = Un1 { inner: 42u8 };
    assert_eq!(unsafe { un1.inner }, 42u8);

    let mut un2 = Un2 { outest: "bar" };
    assert_eq!(unsafe { un2.outest }, "bar");
    un2 = Un2 { outer: true };
    assert_eq!(unsafe { un2.outer }, true);
    un2 = Un2 { inner: 42u8 };
    assert_eq!(unsafe { un2.inner }, 42u8);

    let mut un3 = Un3 { outest: "baz" };
    assert_eq!(unsafe { un3.outest }, "baz");
    un3 = Un3 { outer: &true };
    assert_eq!(unsafe { *un3.outer }, true);

    let f1 = (1,);
    let f2 = (2,);
    let mut un4 = Un4::<()> { f1: &f1.0 };
    assert_eq!(1, unsafe { *un4.f1 });
    un4 = Un4 { f2: &f2.0 };
    assert_eq!(2, unsafe { *un4.f2 });

    let mut un6 = Un6 { f0: "bar" };
    assert_eq!(unsafe { un6.f0 }, "bar");
    un6 = Un6 { f1: 24u8 };
    assert_eq!(unsafe { un6.f1 }, 24u8);
    un6 = Un6 { f2: &true };
    assert_eq!(unsafe { un6.f2 }, &true);
    un6 = Un6 { f3: 12u16 };
    assert_eq!(unsafe { un6.f3 }, 12u16);

    let mut unself = UnSelf::<_> { f0: "selfish" };
    assert_eq!(unsafe { unself.f0 }, "selfish");
    unself = UnSelf { f1: true };
    assert_eq!(unsafe { unself.f1 }, true);
    unself = UnSelf { f2: 24u8 };
    assert_eq!(unsafe { unself.f2 }, 24u8);
}
