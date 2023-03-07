// check-pass

#![feature(type_alias_impl_trait)]

fn main() {}

type Region<'a> = impl std::fmt::Debug + 'a;


fn region<'b>(a: &'b ()) -> Region<'b> {
    a
}
