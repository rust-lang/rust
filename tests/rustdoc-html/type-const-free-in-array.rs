#![crate_name = "foo"]
#![feature(gca_min_const_items, gca_macroless_args)]
#![expect(incomplete_features)]

use std::gca;

const N: usize = gca!(2);

//@ has 'foo/trait.CollectArray.html'
//@ has - '//pre[@class="rust item-decl"]/code' '[A; N]'
pub trait CollectArray<A> {
    fn inner_array(&mut self) -> [A; N];
}
