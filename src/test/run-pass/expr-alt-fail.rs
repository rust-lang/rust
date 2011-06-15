


// xfail-stage0
fn test_simple() {
    auto r = alt (true) { case (true) { true } case (false) { fail } };
    assert (r == true);
}

fn test_box() {
    auto r = alt (true) { case (true) { [10] } case (false) { fail } };
    assert (r.(0) == 10);
}

fn main() { test_simple(); test_box(); }