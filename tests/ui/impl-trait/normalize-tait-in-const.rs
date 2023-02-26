// known-bug: #103507
// failure-status: 101
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
// normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
// rustc-env:RUST_BACKTRACE=0

#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl)]
#![feature(const_refs_to_cell)]
#![feature(inline_const)]

use std::marker::Destruct;

trait T {
    type Item;
}

type Alias<'a> = impl T<Item = &'a ()>;

struct S;
impl<'a> T for &'a S {
    type Item = &'a ();
}

const fn filter_positive<'a>() -> &'a Alias<'a> {
    &&S
}

const fn with_positive<F: ~const for<'a> Fn(&'a Alias<'a>) + ~const Destruct>(fun: F) {
    fun(filter_positive());
}

const fn foo(_: &Alias<'_>) {}

const BAR: () = {
    with_positive(foo);
};

fn main() {}
