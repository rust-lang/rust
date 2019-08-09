// run-pass

#![feature(associated_type_bounds)]

trait Tr1 { type As1; }
trait Tr2 { type As2; }
trait Tr3 {}
trait Tr4<'a> { type As4; }
trait Tr5 { type As5; }

impl Tr1 for &str { type As1 = bool; }
impl Tr2 for bool { type As2 = u8; }
impl Tr3 for u8 {}
impl Tr1 for () { type As1 = (usize,); }
impl<'a> Tr4<'a> for (usize,) { type As4 = u8; }
impl Tr5 for bool { type As5 = u16; }

struct St1<T: Tr1<As1: Tr2>> {
    outest: T,
    outer: T::As1,
    inner: <T::As1 as Tr2>::As2,
}

fn unwrap_1_st1<T: Tr1<As1: Tr2>>(x: St1<T>) -> (T, T::As1, <T::As1 as Tr2>::As2) {
    (x.outest, x.outer, x.inner)
}

fn unwrap_2_st1<T>(x: St1<T>) -> (T, T::As1, <T::As1 as Tr2>::As2)
where
    T: Tr1,
    T::As1: Tr2,
{
    unwrap_1_st1(x)
}

struct St2<T: Tr1<As1: Tr2<As2: Tr3>>> {
    outest: T,
    outer: T::As1,
    inner: <T::As1 as Tr2>::As2,
}

struct St3<T: Tr1<As1: 'static>> {
    outest: T,
    outer: &'static T::As1,
}

struct St4<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l>>> {
    f1: &'x1 <T::As1 as Tr4<'x1>>::As4,
    f2: &'x2 <T::As1 as Tr4<'x2>>::As4,
}

struct St5<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l, As4: Copy>>> {
    f1: &'x1 <T::As1 as Tr4<'x1>>::As4,
    f2: &'x2 <T::As1 as Tr4<'x2>>::As4,
}

struct St6<T>
where
    T: Tr1<As1: Tr2 + 'static + Tr5>,
{
    f0: T,
    f1: <T::As1 as Tr2>::As2,
    f2: &'static T::As1,
    f3: <T::As1 as Tr5>::As5,
}

struct St7<'a, 'b, T> // `<T::As1 as Tr2>::As2: 'a` is implied.
where
    T: Tr1<As1: Tr2>,
{
    f0: &'a T,
    f1: &'b <T::As1 as Tr2>::As2,
}

fn _use_st7<'a, 'b, T>(x: St7<'a, 'b, T>)
where
    T: Tr1,
    T::As1: Tr2,
{
    let _: &'a T = &x.f0;
}

struct StSelf<T> where Self: Tr1<As1: Tr2> {
    f2: <<Self as Tr1>::As1 as Tr2>::As2,
}

impl Tr1 for StSelf<&'static str> { type As1 = bool; }

fn main() {
    let st1 = St1 { outest: "foo", outer: true, inner: 42u8 };
    assert_eq!(("foo", true, 42), unwrap_1_st1(st1));

    let _ = St2 { outest: "foo", outer: true, inner: 42u8 };

    let _ = St3 { outest: "foo", outer: &true };

    let f1 = (1,);
    let f2 = (2,);
    let st4 = St4::<()> { f1: &f1.0, f2: &f2.0, };
    assert_eq!((&1, &2), (st4.f1, st4.f2));

    // FIXME: requires lazy normalization.
    /*
    let f1 = (1,);
    let f2 = (2,);
    let st5 = St5::<()> { f1: &f1.0, f2: &f2.0, };
    assert_eq!((&1, &2), (st5.f1, st5.f2));
    */

    let st6 = St6 { f0: "bar", f1: 24u8, f2: &true, f3: 12u16, };
    assert_eq!(("bar", 24, &true, 12), (st6.f0, st6.f1, st6.f2, st6.f3));

    let stself = StSelf::<&'static str> { f2: 42u8 };
    assert_eq!(stself.f2, 42u8);
}
