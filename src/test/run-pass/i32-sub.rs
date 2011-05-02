// -*- rust -*-

fn main() {
    let i32 x = -400_i32;
    x = 0_i32 - x;
    check(x == 400_i32);
}

