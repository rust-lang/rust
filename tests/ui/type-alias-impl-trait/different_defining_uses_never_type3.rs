#![feature(type_alias_impl_trait)]

type Tait = impl Sized;

struct One;
#[define_opaques(Tait)]
fn one() -> Tait { One }

struct Two<T>(T);
#[define_opaques(Tait)]
fn two() -> Tait { Two::<()>(todo!()) }
//~^ ERROR concrete type differs from previous defining opaque type use

fn main() {}
