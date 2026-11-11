//@ run-pass

#![allow(dead_code)]

trait Tr1 { type As1; }
trait Tr2 { type As2; }
trait Tr3 { type As3; }
trait Tr4<'a> { type As4; }
trait Tr5 { type As5; }

impl Tr1 for &str { type As1 = bool; }
impl Tr2 for bool { type As2 = u8; }
impl Tr3 for u8 { type As3 = fn() -> u8; }
impl Tr1 for () { type As1 = (usize,); }
impl<'a> Tr4<'a> for (usize,) { type As4 = u8; }
impl Tr5 for bool { type As5 = u16; }

enum En1<T: Tr1<As1: Tr2>> {
    Outest(T),
    Outer(T::As1),
    Inner(<T::As1 as Tr2>::As2),
}

fn wrap_en1_1<T>(x: T) -> En1<T> where T: Tr1, T::As1: Tr2 {
    En1::Outest(x)
}

fn wrap_en1_2<T>(x: T::As1) -> En1<T> where T: Tr1, T::As1: Tr2 {
    En1::Outer(x)
}

fn wrap_en1_3<T>(x: <T::As1 as Tr2>::As2) -> En1<T> where T: Tr1, T::As1: Tr2 {
    En1::Inner(x)
}

enum En2<T: Tr1<As1: Tr2<As2: Tr3>>> {
    V0(T),
    V1(T::As1),
    V2(<T::As1 as Tr2>::As2),
    V3(<<T::As1 as Tr2>::As2 as Tr3>::As3),
}

enum En3<T: Tr1<As1: 'static>> {
    V0(T),
    V1(&'static T::As1),
}

enum En4<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l>>> {
    V0(&'x1 <T::As1 as Tr4<'x1>>::As4),
    V1(&'x2 <T::As1 as Tr4<'x2>>::As4),
}

enum _En5<'x1, 'x2, T: Tr1<As1: for<'l> Tr4<'l, As4: Copy>>> {
    _V0(&'x1 <T::As1 as Tr4<'x1>>::As4),
    _V1(&'x2 <T::As1 as Tr4<'x2>>::As4),
}

enum En6<T>
where
    T: Tr1<As1: Tr2 + 'static + Tr5>,
{
    V0(T),
    V1(<T::As1 as Tr2>::As2),
    V2(&'static T::As1),
    V3(<T::As1 as Tr5>::As5),
}

enum _En7<'a, 'b, T> // `<T::As1 as Tr2>::As2: 'a` is implied.
where
    T: Tr1<As1: Tr2>,
{
    V0(&'a T),
    V1(&'b <T::As1 as Tr2>::As2),
}

fn _make_en7<'a, 'b, T>(x: _En7<'a, 'b, T>)
where
    T: Tr1<As1: Tr2>,
{
    match x {
        _En7::V0(x) => {
            let _: &'a T = &x;
        },
        _En7::V1(_) => {},
    }
}

enum EnSelf<T> where Self: Tr1<As1: Tr2> {
    V0(T),
    V1(<Self as Tr1>::As1),
    V2(<<Self as Tr1>::As1 as Tr2>::As2),
}

impl Tr1 for EnSelf<&'static str> { type As1 = bool; }

fn main() {
    if let En1::Outest("foo") = wrap_en1_1::<_>("foo") {} else { panic!() };
    if let En1::Outer(true) = wrap_en1_2::<&str>(true) {} else { panic!() };
    if let En1::Inner(24u8) = wrap_en1_3::<&str>(24u8) {} else { panic!() };

    let _ = En2::<_>::V0("151571");
    let _ = En2::<&str>::V1(false);
    let _ = En2::<&str>::V2(42u8);
    let _ = En2::<&str>::V3(|| 12u8);

    let _ = En3::<_>::V0("deadbeef");
    let _ = En3::<&str>::V1(&true);

    let f1 = (1,);
    let f2 = (2,);
    let _ = En4::<()>::V0(&f1.0);
    let _ = En4::<()>::V1(&f2.0);

    let _ = En6::<_>::V0("bar");
    let _ = En6::<&str>::V1(24u8);
    let _ = En6::<&str>::V2(&false);
    let _ = En6::<&str>::V3(12u16);

    let _ = EnSelf::<_>::V0("foo");
    let _ = EnSelf::<&'static str>::V1(true);
    let _ = EnSelf::<&'static str>::V2(24u8);
}
