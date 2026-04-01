//! regression test for issue https://github.com/rust-lang/rust/issues/17933
pub static X: usize = 1;

fn main() {
    match 1 {
        self::X => {}
        //~^ ERROR expected unit struct, unit variant or constant, found static `self::X`
        _ => {}
    }
}
