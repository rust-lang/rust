// Regression test for #85461: MSVC sometimes fail to link with dead code and #[inline(always)]

extern crate inline_always_with_dead_code;

use inline_always_with_dead_code::{bar, baz};

fn main() {
    bar::call_me();
    baz::call_me();
}
