// xfail-stage0
tag int_fn { f(fn(int) -> int ); }
tag int_box_fn { fb(@fn(int) -> int ); }
fn add1(i: int) -> int { ret i + 1; }
fn main() {
    let g = f(add1);
    assert (g(4) == 5);
    assert (f(add1)(5) == 6);
    assert ((@f(add1))(5) == 6);
    assert (fb(@add1)(7) == 8);
}