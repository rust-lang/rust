//@ run-rustfix
//@ edition: 2024

// Replacing `await!(...)` must preserve grouping and any following `?`.

use std::future::ready;

async fn check() -> Result<(), ()> {
    let future = ready(1);
    let _: i32 = await!(future);
    //~^ ERROR incorrect use of `await`
    let _: i32 = await!(ready(1));
    //~^ ERROR incorrect use of `await`
    let _: i32 = await!(ready(Ok::<i32, ()>(1)))?;
    //~^ ERROR incorrect use of `await`
    let _: i32 = await!(/* keep */ &mut ready(1) /* keep */);
    //~^ ERROR incorrect use of `await`
    Ok(())
}

fn main() {
    let _ = check();
}
