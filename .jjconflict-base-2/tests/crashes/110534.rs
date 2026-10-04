//@ known-bug: #110534
//@ edition:2021
use core::cell::Ref;

struct System;

trait IntoSystem {
    fn into_system(self) -> System;
}

impl IntoSystem for fn(Ref<'_, u32>) {
    fn into_system(self) -> System { System }
}

impl<A> IntoSystem for fn(A)
where
    // n.b. No `Ref<'_, u32>` can satisfy this bound
    A: 'static + for<'x> MaybeBorrowed<'x, Output = A>,
{
    fn into_system(self) -> System { System }
}

//---------------------------------------------------

trait MaybeBorrowed<'a> {
    type Output: 'a;
}

// If you comment this out you'll see the compiler chose to look at the
// fn(A) implementation of IntoSystem above
impl<'a, 'b> MaybeBorrowed<'a> for Ref<'b, u32> {
    type Output = Ref<'a, u32>;
}

// ---------------------------------------------

fn main() {
    fn sys_ref(_age: Ref<u32>) {}
    let _sys_c = (sys_ref as fn(_)).into_system();
    // properly fails
    // let _sys_c = (sys_ref as fn(Ref<'static, u32>)).into_system();
    // properly succeeds
    // let _sys_c = (sys_ref as fn(Ref<'_, u32>)).into_system();
}
