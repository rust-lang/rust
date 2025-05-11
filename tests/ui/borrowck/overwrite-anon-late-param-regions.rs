// A regression test for #124021. When liberating the late bound regions here
// we encounter multiple `LateBoundRegion::Anon`. These ended up resulting in
// distinct nll vars, but mapped to the same `RegionKind::LateParam`. This
// then caused an ICE when trying to fetch lazily computed information for the
// nll var of an overwritten liberated bound region.
#![feature(type_alias_impl_trait)]
type Opaque2<'a> = impl Sized + 'a;

#[define_opaque(Opaque2)]
fn test2() -> impl for<'a, 'b> Fn((&'a str, &'b str)) -> (Opaque2<'a>, Opaque2<'a>) {
    |x| x
    //~^ ERROR lifetime may not live long enough
    //~| ERROR expected generic lifetime parameter, found `'a`
}

fn main() {}
