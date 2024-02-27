// Regression test for #114572, We were inferring an ill-formed type:
//
// `Opaque<'a> = Static<&'a str>`, vs
// `Opaque<'a> = Static<&'static str>`.
#![feature(type_alias_impl_trait)]

struct Static<T: 'static>(T);

type OpaqueRet<'a> = impl Sized + 'a;
//~^ ERROR the type `&'a u8` does not fulfill the required lifetime
fn test_return<'a>(msg: Static<&'static u8>) -> OpaqueRet<'a> {
    msg
}

fn test_rpit<'a>(msg: Static<&'static u8>) -> impl Sized + 'a {
    //~^ ERROR the type `&'a u8` does not fulfill the required lifetime
    msg
}

type OpaqueAssign<'a> = impl Sized + 'a;
//~^ ERROR the type `&'a u8` does not fulfill the required lifetime
fn test_assign<'a>(msg: Static<&'static u8>) -> Option<OpaqueAssign<'a>> {
    let _: OpaqueAssign<'a> = msg;
    None
}

// `OpaqueRef<'a, T> = Ref<'a, T>`, vs
// `OpaqueRef<'a, T> = Ref<'static, T>`.
trait RefAt<'a>: 'a {}
struct Ref<'a, T: RefAt<'a>>(&'a T);
type OpaqueRef<'a, T: RefAt<'static>> = impl Sized + 'a;
//~^ ERROR mismatched types
fn test_trait<'a, T: RefAt<'static>>(msg: Ref<'static, T>) -> OpaqueRef<'a, T> {
    msg
}

fn main() {}
