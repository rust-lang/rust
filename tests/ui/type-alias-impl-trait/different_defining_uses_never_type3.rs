#![feature(type_alias_impl_trait)]

type Tait = impl Sized;

struct One;
#[defines(Tait)]
fn one() -> Tait { One }

struct Two<T>(T);
#[defines(Tait)]
fn two() -> Tait { Two::<()>(todo!()) }
//~^ ERROR concrete type differs from previous defining opaque type use

fn main() {}
