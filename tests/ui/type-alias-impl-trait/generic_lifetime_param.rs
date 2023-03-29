// check-pass

#![feature(type_alias_impl_trait)]

fn main() {}

type Region<'a> = impl std::fmt::Debug + 'a;

#[defines(Region<'b>)]
fn region<'b>(a: &'b ()) -> Region<'b> {
    a
}
