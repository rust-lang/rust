#![feature(type_alias_impl_trait)]

type Bar<'a, 'b> = impl PartialEq<Bar<'b, 'a>> + std::fmt::Debug;

fn bar<'a, 'b>(i: &'a i32) -> Bar<'a, 'b> {
    i //~ ERROR can't compare `&i32` with `impl PartialEq<Opaque
}

fn main() {
    let meh = 42;
    let muh = 69;
    assert_eq!(bar(&meh), bar(&meh));
}
