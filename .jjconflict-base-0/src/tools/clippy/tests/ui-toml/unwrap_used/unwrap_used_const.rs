#![warn(clippy::unwrap_used)]

fn main() {
    const SOME: Option<i32> = Some(3);
    const UNWRAPPED: i32 = SOME.unwrap();
    //~^ ERROR: used `unwrap()` on an `Option` value
    const {
        SOME.unwrap();
        //~^ ERROR: used `unwrap()` on an `Option` value
    }
}
