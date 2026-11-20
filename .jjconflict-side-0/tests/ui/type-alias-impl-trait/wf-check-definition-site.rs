//@ check-pass

// Regression test for #114572, We were inferring an ill-formed type:
//
// `Opaque<'a> = Static<&'a str>`, vs
// `Opaque<'a> = Static<&'static str>`.
//
// The hidden type of the opaque ends up as `Static<'?0 str>`. When
// computing member constraints we end up choosing `'a` for `?0` unless
// `?0` is already required to outlive `'a`. We achieve this by checking
// that `Static<'?0 str>` is well-formed.
#![feature(type_alias_impl_trait)]

struct Static<T: 'static>(T);

type OpaqueRet<'a> = impl Sized + 'a;
#[define_opaque(OpaqueRet)]
fn test_return<'a>(msg: Static<&'static u8>) -> OpaqueRet<'a> {
    msg
}

fn test_rpit<'a>(msg: Static<&'static u8>) -> impl Sized + 'a {
    msg
}

type OpaqueAssign<'a> = impl Sized + 'a;
#[define_opaque(OpaqueAssign)]
fn test_assign<'a>(msg: Static<&'static u8>) {
    let _: OpaqueAssign<'a> = msg;
}

// `OpaqueRef<'a, T> = Ref<'a, T>`, vs
// `OpaqueRef<'a, T> = Ref<'static, T>`.
trait RefAt<'a>: 'a {}
struct Ref<'a, T: RefAt<'a>>(&'a T);
type OpaqueRef<'a, T: RefAt<'static>> = impl Sized + 'a;
#[define_opaque(OpaqueRef)]
fn test_trait<'a, T: RefAt<'static>>(msg: Ref<'static, T>) -> OpaqueRef<'a, T> {
    msg
}

fn main() {}
