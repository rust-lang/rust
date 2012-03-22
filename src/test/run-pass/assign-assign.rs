// Issue 483 - Assignment expressions result in nil
fn test_assign() {
    let mut x: int;
    let mut y: () = x = 10;
    assert (x == 10);
    let mut z = x = 11;
    assert (x == 11);
    z = x = 12;
    assert (x == 12);
}

fn test_assign_op() {
    let mut x: int = 0;
    let mut y: () = x += 10;
    assert (x == 10);
    let mut z = x += 11;
    assert (x == 21);
    z = x += 12;
    assert (x == 33);
}

fn main() { test_assign(); test_assign_op(); }
