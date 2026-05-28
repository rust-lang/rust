fn whatever() -> i32 {
    opaque()
//~^ ERROR mismatched types
}

fn opaque() -> impl Fn() -> i32 {
    || 0
}

fn main() {
    let _ = whatever();
}
