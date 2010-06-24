// -*- rust -*-

fn main() {
    let i32 x = i32(-400);
    x = i32(0) - x;
    check(x == i32(400));
}

