//@aux-build:proc_macros.rs
#![warn(clippy::unnecessary_map_or)]

#[macro_use]
extern crate proc_macros;

const TRUE: bool = true;
const FALSE: bool = false;

fn main() {
    let result = Ok::<i32, i32>(1);

    let _ = result.map_or(false, |_| true);
    //~^ unnecessary_map_or
    let _ = result.map_or(true, |_| false);
    //~^ unnecessary_map_or
    let _ = result.map_or(false, |_: i32| true);
    //~^ unnecessary_map_or
    let _ = result.map_or(!true, |_| TRUE);
    //~^ unnecessary_map_or

    let _ = result.map_or_else(|_| false, |_| true);
    //~^ unnecessary_map_or
    let _ = result.map_or_else(|_| true, |_| false);
    //~^ unnecessary_map_or
    let _ = result.map_or_else(|_: i32| false, |_: i32| true);
    //~^ unnecessary_map_or
    let _ = result.map_or_else(|_| FALSE, |_| !false);
    //~^ unnecessary_map_or

    // Calls in a closure body may have side effects. The lint does not inspect the callee body.
    let _ = result.map_or_else(
        |_| {
            std::hint::black_box(());
            false
        },
        |_| true,
    );
    let _ = result.map_or_else(|error| error > 0, |_| true);
    let _ = result.map_or_else(|_| false, |value| value > 0);

    external! {
        let _ = Ok::<i32, i32>(1).map_or(false, |_| true);
        let _ = Ok::<i32, i32>(1).map_or_else(|_| false, |_| true);
    }

    with_span! {
        let _ = Ok::<i32, i32>(1).map_or(false, |_| true);
        let _ = Ok::<i32, i32>(1).map_or_else(|_| false, |_| true);
    }
}
