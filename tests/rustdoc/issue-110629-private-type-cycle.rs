//@ compile-flags: --document-private-items

#![feature(type_alias_impl_trait)]

type Bar<'a, 'b> = impl PartialEq<Bar<'a, 'b>> + std::fmt::Debug;

// @has issue_110629_private_type_cycle/type.Bar.html
// @has - '//pre[@class="rust item-decl"]' \
//     "pub(crate) type Bar<'a, 'b> = impl PartialEq<Bar<'a, 'b>> + Debug;"

fn bar<'a, 'b>(i: &'a i32) -> Bar<'a, 'b> {
    i
}

fn main() {
    let meh = 42;
    let muh = 42;
    assert_eq!(bar(&meh), bar(&muh));
}
