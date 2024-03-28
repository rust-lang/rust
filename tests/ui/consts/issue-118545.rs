#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Checked<const F: fn()>;
//~^ERROR: using function pointers as const generic parameters is forbidden

fn foo() {}
const _: Checked<foo> = Checked::<foo>;

fn main() {}
