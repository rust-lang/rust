#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

existential type Two<T, U>: Debug;

fn two<T: Debug>(t: T) -> Two<T, u32> {
    //~^ ERROR defining existential type use does not fully define existential type
    (t, 4i8)
}

fn three<T: Debug, U>(t: T) -> Two<T, U> {
    (t, 5i8)
}

trait Bar {
    type Blub: Debug;
    const FOO: Self::Blub;
}

impl Bar for u32 {
    type Blub = i32;
    const FOO: i32 = 42;
}

// this should work! But it requires `two` and `three` not to be defining uses,
// just restricting uses
fn four<T: Debug, U: Bar>(t: T) -> Two<T, U> { //~ concrete type differs from previous
    (t, <U as Bar>::FOO)
}

fn is_sync<T: Sync>() {}

fn asdfl() {
    //FIXME(oli-obk): these currently cause cycle errors
    //is_sync::<Two<i32, u32>>();
    //is_sync::<Two<i32, *const i32>>();
}
