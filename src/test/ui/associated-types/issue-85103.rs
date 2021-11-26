#![feature(rustc_attrs)]

use std::borrow::Cow;

#[rustc_layout(debug)]
type Edges<'a, E> = Cow<'a, [E]>;
//~^ ERROR layout error: NormalizationFailure

fn main() {}
