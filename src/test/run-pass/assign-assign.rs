// Issue 483 - Assignment expressions result in nil
fn test_assign() {
    let x: int;
    let y: () = x = 10;
    assert (x == 10);
    let z = x = 11;
    assert (x == 11);
    z = x = 12;
    assert (x == 12);
}

fn test_assign_op() {
    let x: int = 0;
    let y: () = x += 10;
    assert (x == 10);
    let z = x += 11;
    assert (x == 21);
    z = x += 12;
    assert (x == 33);
}

fn main() { test_assign(); test_assign_op(); }
