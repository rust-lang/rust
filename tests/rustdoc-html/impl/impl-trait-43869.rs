// https://github.com/rust-lang/rust/issues/43869
#![crate_name="foo"]

pub fn g() -> impl Iterator<Item=u8> {
    Some(1u8).into_iter()
}

#[allow(unused_parens)]
pub fn h() -> (impl Iterator<Item=u8>) {
    Some(1u8).into_iter()
}

pub fn i() -> impl Iterator<Item=u8> + 'static {
    Some(1u8).into_iter()
}

pub fn j() -> impl Iterator<Item=u8> + Clone {
    Some(1u8).into_iter()
}

pub fn k() -> [impl Clone; 2] {
    [123u32, 456u32]
}

pub fn l() -> (impl Clone, impl Default) {
    (789u32, -123i32)
}

pub fn m() -> &'static impl Clone {
    &1u8
}

pub fn n() -> *const impl Clone {
    &1u8
}

pub fn o() -> &'static [impl Clone] {
    b":)"
}

// issue #44731
pub fn test_44731_0() -> Box<impl Iterator<Item=u8>> {
    Box::new(g())
}

pub fn test_44731_1() -> Result<Box<impl Clone>, ()> {
    Ok(Box::new(j()))
}

// NOTE these involve Fn sugar, where impl Trait is disallowed for now, see issue #45994
//
//pub fn test_44731_2() -> Box<Fn(impl Clone)> {
//    Box::new(|_: u32| {})
//}
//
//pub fn test_44731_3() -> Box<Fn() -> impl Clone> {
//    Box::new(|| 0u32)
//}

pub fn test_44731_4() -> Box<Iterator<Item=impl Clone>> {
    Box::new(g())
}

//@ has foo/fn.g.html
//@ has foo/fn.h.html
//@ has foo/fn.i.html
//@ has foo/fn.j.html
//@ has foo/fn.k.html
//@ has foo/fn.l.html
//@ has foo/fn.m.html
//@ has foo/fn.n.html
//@ has foo/fn.o.html
//@ has foo/fn.test_44731_0.html
//@ has foo/fn.test_44731_1.html
//@ has foo/fn.test_44731_4.html
