#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
pub struct S<const N: usize>;
// this is a directly represented anon const... it's a bit weird.
// imagine `S<{ (2, const { 1 + 1 }) }>`. a directly represented tuple, containing an anon const.
// now, replace `(2, _)` with `_`. it's a directly represented anon const.
// this is different from const argument lowering failing to represent an argument directly and
// falling back to representing it as an anon const instead.
pub fn f() -> S<{ const { 1 + 1 } }> {
    S
}
