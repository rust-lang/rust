fn test_simple() {
    let r = alt true { true { true } false { fail } };
    assert (r == true);
}

fn test_box() {
    let r = alt true { true { [10] } false { fail } };
    assert (r.(0) == 10);
}

fn main() { test_simple(); test_box(); }