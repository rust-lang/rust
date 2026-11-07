macro_rules! black_hole {
    ($($tt:tt)*) => {}
}

fn main() {
    black_hole! { '\u{FFFFFF}' }
    //~^ ERROR: invalid unicode character escape
    black_hole! { "this is surrogate: \u{DAAA}" }
    //~^ ERROR: invalid unicode character escape
}
