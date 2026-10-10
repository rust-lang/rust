//@ edition: 2024

async fn once<O, F: AsyncFnOnce() -> O>(f: F) -> O {
    Ok::<(), ()>(())?;
    //~^ ERROR: the `?` operator can only be used in an async function that returns

    f()
    //~^ ERROR mismatched types
}

async fn fnref<O, F: AsyncFn() -> O>(f: F) -> O {
    Ok::<(), ()>(())?;
    //~^ ERROR: the `?` operator can only be used in an async function that returns

    f()
    //~^ ERROR mismatched types
}


async fn fnmut<O, F: AsyncFnMut() -> O>(f: F) -> O {
    Ok::<(), ()>(())?;
    //~^ ERROR: the `?` operator can only be used in an async function that returns

    f()
    //~^ ERROR mismatched types
}

fn main() {}
