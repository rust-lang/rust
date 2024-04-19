//@ known-bug: #118545
#![feature(generic_const_exprs)]

struct Checked<const F: fn()>;

fn foo() {}
const _: Checked<foo> = Checked::<foo>;
pub fn main() {}
