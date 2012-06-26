


// -*- rust -*-
fn two(it: fn(int)) { it(0); it(1); }

fn main() {
    let a: ~[mut int] = ~[mut -1, -1, -1, -1];
    let mut p: int = 0;
    do two {|i|
        do two {|j| a[p] = 10 * i + j; p += 1; }
    }
    assert (a[0] == 0);
    assert (a[1] == 1);
    assert (a[2] == 10);
    assert (a[3] == 11);
}
