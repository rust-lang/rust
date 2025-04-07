#[allow(clippy::no_effect, clippy::unnecessary_operation)]
#[warn(clippy::int_plus_one)]
fn main() {
    let x = 1i32;
    let y = 0i32;

    let _ = x >= y + 1;
    //~^ int_plus_one
    let _ = y + 1 <= x;
    //~^ int_plus_one

    let _ = x - 1 >= y;
    //~^ int_plus_one
    let _ = y <= x - 1;
    //~^ int_plus_one

    let _ = x > y; // should be ok
    let _ = y < x; // should be ok
}
