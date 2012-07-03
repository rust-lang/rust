fn f(f: fn()) -> int {
    f(); 0
}

fn main() {
    // Testing that the old rule that statements (even control
    // structures) that have non-nil type be semi-terminated _no
    // longer_ is required
    do f {
    }
    if true { 0 } else { 0 }
    let _x = 0;
}