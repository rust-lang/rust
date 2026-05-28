// Make sure that we escape the arguments of the GAT projection even if we fail to compute
// the href of the corresponding trait (in this case it is private).
// Further, test that we also linkify the GAT arguments.
// https://github.com/rust-lang/rust/issues/94683
#![crate_name="foo"]

//@ has 'foo/type.A.html'
//@ has - '//pre[@class="rust item-decl"]' '<S as Tr>::P<Option<i32>>'
//@ has - '//pre[@class="rust item-decl"]//a[@class="enum"]/@href' '{{channel}}/core/option/enum.Option.html'
pub type A = <S as Tr>::P<Option<i32>>;

/*private*/ trait Tr {
    type P<T>;
}

pub struct S;

impl Tr for S {
    type P<T> = ();
}
