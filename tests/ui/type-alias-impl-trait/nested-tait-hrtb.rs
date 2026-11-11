#![feature(type_alias_impl_trait)]

trait Trait<'a> { type Assoc; }
impl<'a> Trait<'a> for () { type Assoc = &'a str; }

type WithoutLt = impl Sized;
#[define_opaque(WithoutLt)]
fn without_lt() -> impl for<'a> Trait<'a, Assoc = WithoutLt> {}
//~^ ERROR captures lifetime that does not appear in bounds

type WithLt<'a> = impl Sized + 'a;

#[define_opaque(WithLt)]
fn with_lt() -> impl for<'a> Trait<'a, Assoc = WithLt<'a>> {}
//~^ ERROR expected generic lifetime parameter, found `'a`

fn main() {}
