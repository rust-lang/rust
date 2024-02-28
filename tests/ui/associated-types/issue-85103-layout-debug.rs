#![feature(rustc_attrs)]

use std::borrow::Cow;

#[rustc_layout(debug)]
type Edges<'a, E> = Cow<'a, [E]>;
//~^ ERROR trait `ToOwned` is not implemented for `[E]`

fn main() {}
