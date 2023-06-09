#![feature(rustc_attrs)]

use std::borrow::Cow;

#[rustc_layout(debug)]
type Edges<'a, E> = Cow<'a, [E]>;
//~^ 6:1: 6:18: unable to determine layout for `<[E] as ToOwned>::Owned` because `<[E] as ToOwned>::Owned` cannot be normalized

fn main() {}
