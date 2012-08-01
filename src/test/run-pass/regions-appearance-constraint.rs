/* Tests conditional rooting of the box y */

fn testfn(cond: bool) {
    let mut x = @3;
    let mut y = @4;

    let mut a = &*x;

    let mut exp = 3;
    if cond {
        a = &*y;

        exp = 4;
    }

    x = @5;
    y = @6;
    assert *a == exp;
}

fn main() {
}
