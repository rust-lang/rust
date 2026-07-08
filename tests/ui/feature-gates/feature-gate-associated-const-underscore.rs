struct Thing {}

impl Thing {
    const _: () = {};
    //~^ ERROR: naming associated constants with `_` is unstable [E0658]
}

fn main() {
    let _ = Thing {};
}
