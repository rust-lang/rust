fn test_simple() {
    let r = match true { true => { true } false => { fail } };
    assert (r == true);
}

fn test_box() {
    let r = match true { true => { ~[10] } false => { fail } };
    assert (r[0] == 10);
}

fn main() { test_simple(); test_box(); }
