// check-pass

use std::ops::Deref;

fn main() {
    let _x: fn(&i32) -> <&i32 as Deref>::Target = unimplemented!();
}
