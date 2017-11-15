


#[allow(no_effect, unnecessary_operation)]
#[warn(int_plus_one)]
fn main() {
    let x = 1i32;
    let y = 0i32;
    
    x >= y + 1;
    y + 1 <= x;

    x - 1 >= y;
    y <= x - 1;

    x > y; // should be ok
    y < x; // should be ok
}
