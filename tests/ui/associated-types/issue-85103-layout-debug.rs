#![feature(rustc_attrs)]

use std::borrow::Cow;

#[rustc_layout(debug)]
type Edges<'a, E> = Cow<'a, [E]>;
//~^ the trait bound `[E]: ToOwned` is not satisfied

fn main() {}
