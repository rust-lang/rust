#[cfg(any())]
fn test() {
    let x: unsafe<> ();
    //~^ ERROR unsafe binder types are experimental
}

fn main() {}
