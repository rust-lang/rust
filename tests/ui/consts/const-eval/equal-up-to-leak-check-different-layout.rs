//@ check-pass

// Regression test for #155477. We previously didn't detect higher-ranked region
// errors when checking whether types are equal in `rustc_const_eval::relate_types`, that
// then caused us to trigger an assert that equal types have the same layout.

use std::mem::transmute;

type F1 = for<'a> fn(&'a ());
type F2 = fn(&'static ());

trait Trait {
    type Assoc;
}

impl Trait for F1 {
    type Assoc = i64;
}
#[expect(coherence_leak_check)]
impl Trait for F2 {
    type Assoc = [i32; 2];
}

struct Thing<T: Trait>(T::Assoc);

const fn foo(x: Thing<F1>) -> Thing<F2> {
    // This transmute is legal: The two types have the same size.
    unsafe { transmute(x) }
}

fn main() {
    const { foo(Thing(1i64)); }
}
